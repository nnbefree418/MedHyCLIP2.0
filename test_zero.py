# -*- coding: utf-8 -*-
import os  # 操作系统路径等相关操作
import argparse  # 命令行参数解析库
import random  # Python 自带随机数库
import math  # 数学函数库
import numpy as np  # 数值计算库
import torch  # PyTorch 主库
from torch import nn  # 神经网络模块
from torch.nn import functional as F  # 常用函数接口（卷积、插值等）
from tqdm import tqdm  # 进度条显示
from sklearn.metrics import roc_auc_score  # ROC-AUC 评价指标
from scipy.ndimage import gaussian_filter  # 高斯滤波（本文件未使用）
from dataset.medical_zero import MedTestDataset, MedTrainDataset  # 零样本训练/测试数据集定义
from CLIP.clip import create_model  # 创建 CLIP 模型
from CLIP.tokenizer import tokenize  # CLIP 文本 tokenizer（本文件未直接使用）
from CLIP.adapter import CLIP_Inplanted  # 在 CLIP 中插入适配器的封装模型
from PIL import Image  # 图像处理（本文件未直接使用）
from sklearn.metrics import precision_recall_curve  # PR 曲线（本文件未使用）
from loss import FocalLoss, BinaryDiceLoss  # 自定义 Focal loss 和 Dice loss
from utils import augment, encode_text_with_prompt_ensemble, encode_text_with_hyperbolic_adjustment  # 数据增强和文本编码工具
from prompt import REAL_NAME  # 各任务真实名称字典，用于文本 prompt
import geoopt  # 双曲几何库（用于 Hyper-MVFA）

import warnings  # 警告过滤
warnings.filterwarnings("ignore")  # 忽略所有警告，避免日志过多

# 检测是否有可用 GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")  # 有 GPU 则用 cuda:0，否则用 CPU

# 各数据集任务到整数索引的映射（>0 表示有像素级标注，<=0 表示只做图像级）
CLASS_INDEX = {'Brain':3, 'Liver':2, 'Retina_RESC':1, 'Retina_OCT2017':-1, 'Chest':-2, 'Histopathology':-3}
# 反向映射：由索引恢复任务名称
CLASS_INDEX_INV = {3:'Brain', 2:'Liver', 1:'Retina_RESC', -1:'Retina_OCT2017', -2:'Chest', -3:'Histopathology'}


def setup_seed(seed):
    """
    设置随机种子，保证实验可复现
    """
    torch.manual_seed(seed)  # CPU 随机数种子
    torch.cuda.manual_seed_all(seed)  # 所有 GPU 随机数种子
    np.random.seed(seed)  # numpy 随机数种子
    random.seed(seed)  # Python 内置随机数种子
    torch.backends.cudnn.deterministic = True  # cuDNN 使用确定性算法
    torch.backends.cudnn.benchmark = False  # 关闭 benchmark，避免非确定性行为


def main():
    """
    零样本测试脚本主入口：
    1）加载预训练好的 zero-shot 适配器权重
    2）在指定数据集上进行图像级 / 像素级异常检测评估
    """
    parser = argparse.ArgumentParser(description='Testing')  # 命令行参数解析器
    parser.add_argument('--model_name', type=str, default='ViT-L-14-336', help="ViT-B-16-plus-240, ViT-L-14-336")  # CLIP backbone 名称
    parser.add_argument('--pretrain', type=str, default='openai', help="laion400m, openai")  # 预训练权重来源
    parser.add_argument('--obj', type=str, default='Retina_RESC')  # 当前测试的数据集/任务
    parser.add_argument('--data_path', type=str, default='./data/')  # 数据路径
    parser.add_argument('--batch_size', type=int, default=1)  # batch 大小
    parser.add_argument('--img_size', type=int, default=240)  # 输入图像尺寸
    parser.add_argument('--save_path', type=str, default=None, help='checkpoint dir')  # 预训练 zero-shot checkpoint 保存路径
    parser.add_argument("--epoch", type=int, default=50, help="epochs")  # 保留一致性
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument('--seed', type=int, default=111)  # 随机种子
    # Hyper-MVFA 双曲模式参数
    parser.add_argument('--use_hyperbolic', action='store_true', help='Use hyperbolic adapters and distances')
    parser.add_argument('--hyperbolic_c', type=float, default=0.1, help='Curvature of Poincare ball')
    parser.add_argument('--scale_normal', type=float, default=0.1, help='Radius scale for normal text embeddings')
    parser.add_argument('--scale_abnormal', type=float, default=0.8, help='Radius scale for abnormal text embeddings')
    parser.add_argument('--temperature', type=float, default=20.0, help='Temperature for scaling hyperbolic distances to logits')
    parser.add_argument('--tag', type=str, default=None,
                        help='Optional tag for checkpoint naming, should match train_zero.py')
    parser.add_argument('--patience', type=int, default=10,
                        help='(unused in test, kept for CLI compatibility with train_zero.py)')
    args = parser.parse_args()  # 解析命令行参数

    # ===== 自动根据是否使用双曲模式选择默认读取目录 =====
    if args.save_path is None:
        mode_tag = "zero-shot-hyper" if args.use_hyperbolic else "zero-shot-euclid"
        args.save_path = os.path.join("./ckpt", mode_tag)

    # 设置随机种子
    setup_seed(args.seed)
    
    # 固定特征提取器：创建预训练 CLIP 模型
    clip_model = create_model(model_name=args.model_name,
                              img_size=args.img_size,
                              device=device,
                              pretrained=args.pretrain,
                              require_pretrained=True)
    clip_model.eval()

    # 在 CLIP 基础上插入适配器，构建 MVFA 模型
    model = CLIP_Inplanted(clip_model=clip_model, 
                           features=args.features_list,
                           use_hyperbolic=args.use_hyperbolic,
                           hyperbolic_c=args.hyperbolic_c).to(device)
    model.eval()

    # ========= 新增：根据 tag 决定读取哪个 ckpt 文件名 =========
    if args.tag is None:
        ckpt_name = f"{args.obj}.pth"
    else:
        ckpt_name = f"{args.obj}_{args.tag}.pth"
    ckpt_path = os.path.join(args.save_path, ckpt_name)
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    # ======================================================

    model.seg_adapters.load_state_dict(checkpoint["seg_adapters"])
    model.det_adapters.load_state_dict(checkpoint["det_adapters"])

    # 保留但不使用优化器（保持与 train_zero 结构一致）
    seg_optimizer = torch.optim.Adam(list(model.seg_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
    det_optimizer = torch.optim.Adam(list(model.det_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))

    # 加载数据集
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    test_dataset = MedTestDataset(args.data_path, args.obj, args.img_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)

    # 文本特征列表
    text_feature_list = [0]
    ball_list = [None]
    with torch.cuda.amp.autocast(), torch.no_grad():
        for i in [1,2,3,-3,-2,-1]:
            if args.use_hyperbolic:
                text_feature, ball = encode_text_with_hyperbolic_adjustment(
                    clip_model,
                    REAL_NAME[CLASS_INDEX_INV[i]],
                    device,
                    use_hyperbolic=True,
                    c=args.hyperbolic_c,
                    scale_normal=args.scale_normal,
                    scale_abnormal=args.scale_abnormal
                )
                ball_list.append(ball)
            else:
                text_feature = encode_text_with_prompt_ensemble(
                    clip_model, REAL_NAME[CLASS_INDEX_INV[i]], device
                )
                ball_list.append(None)
            text_feature_list.append(text_feature)

    # 用当前任务的文本特征做评估
    _ = test(args, model, test_loader,
             text_feature_list[CLASS_INDEX[args.obj]],
             ball_list[CLASS_INDEX[args.obj]],
             args.temperature)



def normalize_map_per_image(x, eps=1e-8):
    """Per-image min-max normalization over spatial dimensions (numpy array)."""
    if x.ndim == 2:
        x_min = x.min(keepdims=True)
        x_max = x.max(keepdims=True)
    elif x.ndim == 3:
        x_min = x.min(axis=(1, 2), keepdims=True)
        x_max = x.max(axis=(1, 2), keepdims=True)
    elif x.ndim == 4:
        x_min = x.min(axis=(1, 2, 3), keepdims=True)
        x_max = x.max(axis=(1, 2, 3), keepdims=True)
    else:
        raise ValueError(f"Unsupported map shape: {x.shape}")
    return (x - x_min) / (x_max - x_min + eps)

def test(args, seg_model, test_loader, text_features, ball=None, temperature=20.0):
    """
    在测试集上评估 zero-shot 模型
    """
    gt_list = []
    gt_mask_list = []
    image_scores = []
    segment_scores = []
    
    for (image, y, mask) in tqdm(test_loader):
        image = image.to(device)
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

        with torch.no_grad(), torch.cuda.amp.autocast():
            _, ori_seg_patch_tokens, ori_det_patch_tokens = seg_model(image)
            
            batch_size_current = image.shape[0]
            for batch_idx in range(batch_size_current):
                ori_seg_patch_tokens_single = [p[batch_idx, 1:, :] for p in ori_seg_patch_tokens]
                ori_det_patch_tokens_single = [p[batch_idx, 1:, :] for p in ori_det_patch_tokens]
                
                # ------------------ 图像级分数 ------------------
                anomaly_score = 0
                patch_tokens = ori_det_patch_tokens_single.copy()
                for layer in range(len(patch_tokens)):
                    if args.use_hyperbolic:
                        L, C = patch_tokens[layer].shape
                        text_h = text_features.T
                        normal_text = text_h[0]
                        abnormal_text = text_h[1]
                        dist_normal = ball.dist(patch_tokens[layer], normal_text)
                        dist_abnormal = ball.dist(patch_tokens[layer], abnormal_text)
                        logits_normal = -temperature * dist_normal
                        logits_abnormal = -temperature * dist_abnormal
                        anomaly_map = torch.stack([logits_normal, logits_abnormal], dim=-1).unsqueeze(0)
                        anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                        anomaly_score += anomaly_map.mean()
                    else:
                        patch_tokens[layer] /= patch_tokens[layer].norm(dim=-1, keepdim=True)
                        anomaly_map = (100.0 * patch_tokens[layer] @ text_features).unsqueeze(0)
                        anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                        anomaly_score += anomaly_map.mean()
                anomaly_score = anomaly_score / len(patch_tokens)
                image_scores.append(anomaly_score.cpu())

                # ------------------ 像素级分数 ------------------
                patch_tokens = ori_seg_patch_tokens_single
                anomaly_maps = []
                for layer in range(len(patch_tokens)):
                    if args.use_hyperbolic:
                        L, C = patch_tokens[layer].shape
                        H = int(np.sqrt(L))
                        text_h = text_features.T
                        normal_text = text_h[0]
                        abnormal_text = text_h[1]
                        dist_normal = ball.dist(patch_tokens[layer], normal_text)
                        dist_abnormal = ball.dist(patch_tokens[layer], abnormal_text)
                        logits_normal = -temperature * dist_normal
                        logits_abnormal = -temperature * dist_abnormal
                        anomaly_map = torch.stack([logits_normal, logits_abnormal], dim=-1).unsqueeze(0)
                        B = 1
                        anomaly_map = F.interpolate(
                            anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                            size=args.img_size, mode='bilinear', align_corners=True
                        )
                        anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
                        anomaly_maps.append(anomaly_map.cpu().numpy())
                    else:
                        patch_tokens[layer] /= patch_tokens[layer].norm(dim=-1, keepdim=True)
                        anomaly_map = (100.0 * patch_tokens[layer] @ text_features).unsqueeze(0)
                        B, L, C = anomaly_map.shape
                        H = int(np.sqrt(L))
                        anomaly_map = F.interpolate(
                            anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                            size=args.img_size, mode='bilinear', align_corners=True
                        )
                        anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
                        anomaly_maps.append(anomaly_map.cpu().numpy())
                final_score_map = np.mean(anomaly_maps, axis=0)
                
                gt_mask_list.append(mask[batch_idx].squeeze().cpu().detach().numpy())
                gt_list.extend(y[batch_idx:batch_idx+1].cpu().detach().numpy())
                segment_scores.append(final_score_map)
    
    gt_list = np.array(gt_list)
    gt_mask_list = np.asarray(gt_mask_list)
    gt_mask_list = (gt_mask_list>0).astype(np.int_)

    segment_scores = np.array(segment_scores)
    image_scores = np.array(image_scores)

    segment_scores = normalize_map_per_image(segment_scores)
    image_scores = (image_scores - image_scores.min()) / (image_scores.max() - image_scores.min())

    img_roc_auc_det = roc_auc_score(gt_list, image_scores)
    print(f'{args.obj} AUC : {round(img_roc_auc_det,4)}')

    if CLASS_INDEX[args.obj] > 0:
        seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), segment_scores.flatten())
        print(f'{args.obj} pAUC : {round(seg_roc_auc,4)}')
        return seg_roc_auc + img_roc_auc_det
    else:
        return img_roc_auc_det


if __name__ == '__main__':
    main()
