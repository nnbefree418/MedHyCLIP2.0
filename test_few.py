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
from scipy.ndimage import gaussian_filter  # 高斯滤波（本文件未使用）
from dataset.medical_few import MedDataset  # few-shot 医疗数据集
from CLIP.clip import create_model  # 创建 CLIP 模型
from CLIP.tokenizer import tokenize  # CLIP 文本 tokenizer（本文件未直接使用）
from CLIP.adapter import CLIP_Inplanted  # 在 CLIP 中插入适配器的封装模型
from PIL import Image  # 图像处理库（本文件未直接使用）
from sklearn.metrics import roc_auc_score, precision_recall_curve, pairwise  # 评价指标（pairwise 本文件未使用）
from loss import FocalLoss, BinaryDiceLoss  # 自定义 Focal loss 和 Dice loss
from utils import augment, cos_sim, encode_text_with_prompt_ensemble, encode_text_with_hyperbolic_adjustment, hyperbolic_distance_batch  # 数据增强、余弦相似度和文本编码工具
from prompt import REAL_NAME  # 各任务真实名称字典，用于构造文本 prompt
import geoopt  # 双曲几何库（用于 Hyper-MVFA）
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用 tokenizer 并行，避免多进程冲突

import warnings  # 警告过滤
warnings.filterwarnings("ignore")  # 忽略所有警告，避免日志过多

# 判断是否有可用 GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")  # 有 GPU 使用 cuda:0，否则使用 CPU

# 各数据集任务对应的索引（>0 表示有像素级标注，仅分割；<=0 表示只做图像级检测）
CLASS_INDEX = {'Brain':3, 'Liver':2, 'Retina_RESC':1, 'Retina_OCT2017':-1, 'Chest':-2, 'Histopathology':-3}

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
    few-shot 测试脚本主入口：
    1）加载预训练好的 few-shot 适配器权重
    2）基于同样的 few-shot 增强策略构建 memory bank
    3）在测试集上进行 zero-shot + few-shot 融合评估
    """
    parser = argparse.ArgumentParser(description='Testing')  # 命令行参数解析器
    parser.add_argument('--model_name', type=str, default='ViT-L-14-336', help="ViT-B-16-plus-240, ViT-L-14-336")  # CLIP backbone 名称
    parser.add_argument('--pretrain', type=str, default='openai', help="laion400m, openai")  # 预训练权重来源
    parser.add_argument('--obj', type=str, default='Liver')  # 当前实验对象（数据集/任务）
    parser.add_argument('--data_path', type=str, default='./data/')  # 数据路径
    parser.add_argument('--batch_size', type=int, default=1)  # 测试 batch size
    parser.add_argument('--save_model', type=int, default=1)  # 是否保存模型（此脚本不再训练，基本无用）
    parser.add_argument('--save_path', type=str, default=None, help='checkpoint dir')  # few-shot 模型保存/加载路径
    parser.add_argument('--img_size', type=int, default=240)  # 输入图像尺寸
    parser.add_argument("--epoch", type=int, default=50, help="epochs")  # 训练轮数参数（测试脚本中未用）
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")  # 学习率（用于定义优化器，但此脚本不训练）
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")  # 使用 CLIP 中哪些层的特征
    parser.add_argument('--seed', type=int, default=111)  # 随机种子
    parser.add_argument('--shot', type=int, default=4)  # few-shot 支持样本数量
    parser.add_argument('--iterate', type=int, default=0)  # 是否使用不同的 few-shot 组合（与数据集定义相关）
    # Hyper-MVFA 双曲模式参数
    parser.add_argument('--use_hyperbolic', action='store_true', help='Use hyperbolic adapters and distances')
    parser.add_argument('--hyperbolic_c', type=float, default=0.1, help='Curvature of Poincare ball')
    parser.add_argument('--scale_normal', type=float, default=0.1, help='Radius scale for normal text embeddings')
    parser.add_argument('--scale_abnormal', type=float, default=0.8, help='Radius scale for abnormal text embeddings')
    parser.add_argument('--temperature', type=float, default=20.0, help='Temperature for scaling hyperbolic distances to logits')
    # ========= 新增：tag 参数，用于区分不同实验版本的 few-shot checkpoint =========
    parser.add_argument('--tag', type=str, default=None,
                        help='Optional tag for checkpoint naming, should match train_few.py')
    parser.add_argument('--patience', type=int, default=10,
                        help='(unused in test, kept for CLI compatibility with train_few.py)')
    # ============================================================
    args = parser.parse_args()  # 解析命令行参数

    # ===== 自动根据是否使用双曲模式选择默认读取目录 =====
    if args.save_path is None:
        mode_tag = "few-shot-hyper" if args.use_hyperbolic else "few-shot-euclid"
        args.save_path = os.path.join("./ckpt", mode_tag)

    # ===== 根据是否指定 tag 决定要读取的模型文件名 =====
    if args.tag is None:
        ckpt_name = f'{args.obj}.pth'
    else:
        ckpt_name = f'{args.obj}_{args.tag}.pth'
    ckpt_path = os.path.join(args.save_path, ckpt_name)
    print(f"Loading few-shot checkpoint from: {ckpt_path}")
    # ======================================================

    # 设置随机种子
    setup_seed(args.seed)
    
    # 固定特征提取器：创建预训练 CLIP 模型
    clip_model = create_model(model_name=args.model_name,  # backbone 名称
                              img_size=args.img_size,      # 输入图片大小
                              device=device,               # 设备
                              pretrained=args.pretrain,    # 预训练权重来源
                              require_pretrained=True)     # 强制需要预训练权重
    clip_model.eval()  # CLIP 模型设为 eval 模式（不训练 CLIP 本体）

    # 在 CLIP 基础上插入适配器，构建 MVFA 模型
    model = CLIP_Inplanted(clip_model=clip_model, 
                           features=args.features_list,
                           use_hyperbolic=args.use_hyperbolic,
                           hyperbolic_c=args.hyperbolic_c).to(device)
    model.eval()  # 初始设为 eval 模式

    # 加载 already 训练好的 few-shot 适配器权重（支持 tag）
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.seg_adapters.load_state_dict(checkpoint["seg_adapters"])  # 加载分割适配器权重
    model.det_adapters.load_state_dict(checkpoint["det_adapters"])  # 加载检测适配器权重

    # 允许所有参数求梯度（虽然测试阶段不会用到）
    for name, param in model.named_parameters():
        param.requires_grad = True

    # 仅为适配器构建优化器（本脚本不再训练，只是保持结构一致）
    seg_optimizer = torch.optim.Adam(list(model.seg_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))  # 分割适配器优化器
    det_optimizer = torch.optim.Adam(list(model.det_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))  # 检测适配器优化器



    # 加载 few-shot 测试数据集
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}  # 设置为 0 避免 /tmp 空间不足问题
    test_dataset = MedDataset(args.data_path, args.obj, args.img_size, args.shot, args.iterate)  # 含 few-shot 支持样本的数据集
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)  # 测试 DataLoader


    # few-shot 图像增强：基于少量支持样本构造训练/支持图像
    augment_abnorm_img, augment_abnorm_mask = augment(test_dataset.fewshot_abnorm_img, test_dataset.fewshot_abnorm_mask)  # 增强异常图像及其 mask
    augment_normal_img, augment_normal_mask = augment(test_dataset.fewshot_norm_img)  # 增强正常图像（无显式 mask）

    # 拼接增强后的异常和正常 few-shot 图像及其 mask
    augment_fewshot_img = torch.cat([augment_abnorm_img, augment_normal_img], dim=0)
    augment_fewshot_mask = torch.cat([augment_abnorm_mask, augment_normal_mask], dim=0)
    
    # 为增强后的 few-shot 样本构造图像级标签：异常为 1，正常为 0
    augment_fewshot_label = torch.cat([torch.Tensor([1] * len(augment_abnorm_img)), torch.Tensor([0] * len(augment_normal_img))], dim=0)

    # 基于增强后的 few-shot 样本构造“训练数据集”（本脚本中未再训练，只是保持一致）
    train_dataset = torch.utils.data.TensorDataset(augment_fewshot_img, augment_fewshot_mask, augment_fewshot_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)  # few-shot 训练 DataLoader（此处不训练）


    # memory bank 构建：使用增强后的正常图像作为支持集
    support_dataset = torch.utils.data.TensorDataset(augment_normal_img)  # 仅包含正常图像
    support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=1, shuffle=True, **kwargs)  # 支持集 DataLoader


    # 定义损失函数（本脚本不会反向更新，只是与 train_few 保持一致）
    loss_focal = FocalLoss()  # 像素级 Focal Loss
    loss_dice = BinaryDiceLoss()  # 像素级 Dice Loss
    loss_bce = torch.nn.BCELoss()  # 图像级 BCE Loss（输入为概率）


    # 文本 prompt 编码：得到该任务对应的文本特征（[dim, 2]）
    with torch.cuda.amp.autocast(), torch.no_grad():  # 使用混合精度并关闭梯度
        if args.use_hyperbolic:
            # 使用双曲模式：调用双曲文本编码 + 半径调整
            text_features, ball = encode_text_with_hyperbolic_adjustment(
                clip_model,
                REAL_NAME[args.obj],
                device,
                use_hyperbolic=True,
                c=args.hyperbolic_c,
                scale_normal=args.scale_normal,
                scale_abnormal=args.scale_abnormal
            )
        else:
            # 使用欧氏模式：调用原始文本编码
            text_features = encode_text_with_prompt_ensemble(clip_model, REAL_NAME[args.obj], device)
            ball = None

    best_result = 0  # 记录最优结果（此脚本只测试一次，主要保持接口一致）

    # -------------------- 支持集特征提取并构建 memory bank --------------------
    seg_features = []  # 存储分割头特征（按样本、按层）
    det_features = []  # 存储检测头特征（按样本、按层）
    for image in support_loader:
        image = image[0].to(device)  # support_loader 返回 (img,)，取出第 0 个元素
        with torch.no_grad():  # 构建 memory bank 不需要梯度
            _, seg_patch_tokens, det_patch_tokens = model(image)
            # 每层 seg_patch_tokens 形状 [1, L+1, C]，去掉 CLS token（索引 0），只保留 patch tokens
            seg_patch_tokens = [p[0, 1:, :].contiguous() for p in seg_patch_tokens]
            det_patch_tokens = [p[0, 1:, :].contiguous() for p in det_patch_tokens]
            seg_features.append(seg_patch_tokens)  # 每个样本一个 list（按层）
            det_features.append(det_patch_tokens)
    # 对所有支持样本在“样本维度”上拼接，得到每层的 memory 特征
    seg_mem_features = [torch.cat([seg_features[j][i] for j in range(len(seg_features))], dim=0) for i in range(len(seg_features[0]))]
    det_mem_features = [torch.cat([det_features[j][i] for j in range(len(det_features))], dim=0) for i in range(len(det_features[0]))]
    

    # 调用 test 函数，在测试集上进行 zero-shot + few-shot 融合评估
    result = test(args, model, test_loader, text_features, seg_mem_features, det_mem_features, ball, args.temperature)



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

def test(args, model, test_loader, text_features, seg_mem_features, det_mem_features, ball=None, temperature=1.0):
    """
    few-shot 测试：
    - 对有像素级标注的任务：使用 seg head 做 zero-shot + few-shot 融合
    - 对只有图像级标注的任务：使用 det head 做 zero-shot + few-shot 融合
    注意：temperature 参数应与训练时保持一致，默认值 1.0 与 args.temperature 默认值相同
    
    Args:
        args: 命令行参数
        model: 模型
        test_loader: 测试数据加载器
        text_features: 文本特征 [C, 2]
        seg_mem_features: 分割 memory bank 特征列表
        det_mem_features: 检测 memory bank 特征列表
        ball: PoincareBall 对象（双曲模式时使用）
        temperature: 温度系数，用于缩放双曲距离到 logits
    """
    gt_list = []  # 图像级 GT 标签
    gt_mask_list = []  # 像素级 GT mask

    det_image_scores_zero = []  # zero-shot 检测头图像级分数
    det_image_scores_few = []  # few-shot 检测头图像级分数
    
    seg_score_map_zero = []  # zero-shot 分割头像素级分数图
    seg_score_map_few= []  # few-shot 分割头像素级分数图

    # 遍历测试集
    for (image, y, mask) in tqdm(test_loader):
        image = image.to(device)  # 图像送至设备
        # 将 mask 二值化：>0.5 为 1，其余为 0
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

        # 测试阶段不计算梯度，使用混合精度
        with torch.no_grad(), torch.cuda.amp.autocast():
            # 前向传播：获取分割/检测 patch tokens
            _, seg_patch_tokens, det_patch_tokens = model(image)
            
            # 遍历 batch 中的每个样本
            batch_size_current = image.shape[0]
            for batch_idx in range(batch_size_current):
                # 提取当前样本的 tokens，去掉 CLS token（索引 0）
                seg_patch_tokens_single = [p[batch_idx, 1:, :] for p in seg_patch_tokens]
                det_patch_tokens_single = [p[batch_idx, 1:, :] for p in det_patch_tokens]

                # 若当前任务有像素级标注（如 Liver, Brain, Retina_RESC），使用分割头融合
                if CLASS_INDEX[args.obj] > 0:

                    # ---------------- few-shot, seg head：基于 memory bank 的像素级异常图 ----------------
                    anomaly_maps_few_shot = []
                    for idx, p in enumerate(seg_patch_tokens_single):
                        # seg_mem_features[idx]: 该层的支持样本特征 [N_mem, C]
                        # p: 当前测试图像对应层 patch 特征 [L, C]
                        if args.use_hyperbolic:
                            # ===== 双曲模式：使用向量化双曲距离 =====
                            dist = hyperbolic_distance_batch(seg_mem_features[idx], p, ball)  # [N_mem, L]
                            height = int(np.sqrt(dist.shape[1]))  # patch 数量 L = H*H
                            # 距离越小越正常，取最近邻（最小距离）作为该 patch 的异常分数，应用温度系数缩放
                            anomaly_map_few_shot = torch.min(temperature * dist, dim=0)[0].reshape(1, 1, height, height)
                            # 插值到原始图像大小
                            anomaly_map_few_shot = F.interpolate(anomaly_map_few_shot,
                                                                    size=args.img_size, mode='bilinear', align_corners=True)
                            anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())
                        else:
                            # ===== 欧氏模式：使用余弦相似度 =====
                            cos = cos_sim(seg_mem_features[idx], p)  # 计算余弦相似度 [N_mem, L]
                            height = int(np.sqrt(cos.shape[1]))  # patch 网格大小 H（L = H*H）
                            # 使用 1 - cos 作为距离，并取每个 patch 在 memory bank 中最小相似度（最大距离）作为异常度
                            anomaly_map_few_shot = torch.min((1 - cos), 0)[0].reshape(1, 1, height, height)
                            # 插值到原始图像大小
                            anomaly_map_few_shot = F.interpolate(torch.tensor(anomaly_map_few_shot),
                                                                    size=args.img_size, mode='bilinear', align_corners=True)
                            anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())
                    # 对所有层的 few-shot anomaly map 求和（few-shot 多层融合保持 sum）
                    score_map_few = np.sum(anomaly_maps_few_shot, axis=0)
                    seg_score_map_few.append(score_map_few)

                    # ---------------- zero-shot, seg head：基于文本头的像素级异常图 ----------------
                    anomaly_maps = []
                    for layer in range(len(seg_patch_tokens_single)):
                        if args.use_hyperbolic:
                            # ===== 双曲模式 =====
                            L, C = seg_patch_tokens_single[layer].shape
                            H = int(np.sqrt(L))  # patch 网格尺寸 H x H
                            
                            # 文本特征：[C, 2] -> [2, C]
                            text_h = text_features.T  # [2, C]
                            normal_text = text_h[0]  # [C]
                            abnormal_text = text_h[1]  # [C]
                            
                            # 向量化计算距离
                            dist_normal = ball.dist(seg_patch_tokens_single[layer], normal_text)  # [L]
                            dist_abnormal = ball.dist(seg_patch_tokens_single[layer], abnormal_text)  # [L]
                            
                            # 距离转 logits
                            logits_normal = -temperature * dist_normal
                            logits_abnormal = -temperature * dist_abnormal
                            
                            # Stack 成 [L, 2]
                            anomaly_map = torch.stack([logits_normal, logits_abnormal], dim=-1).unsqueeze(0)  # [1, L, 2]
                            B = 1
                            # 将 [1, L, 2] 变形并插值到 img_size×img_size
                            anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                        size=args.img_size, mode='bilinear', align_corners=True)
                            # 对类别维 softmax 取异常通道（索引 1）
                            anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
                            # 转 numpy 存入列表
                            anomaly_maps.append(anomaly_map.cpu().numpy())
                        else:
                            # ===== 欧氏模式 =====
                            # L2 归一化特征
                            seg_patch_tokens_single[layer] /= seg_patch_tokens_single[layer].norm(dim=-1, keepdim=True)
                            # 与文本特征相乘，得到 [L, 2] logits
                            anomaly_map = (100.0 * seg_patch_tokens_single[layer] @ text_features).unsqueeze(0)
                            B, L, C = anomaly_map.shape  # B: batch, L: patch 数，C: 类别数
                            H = int(np.sqrt(L))  # patch 网格尺寸 H x H
                            # reshape 为 [B, 2, H, H] 并插值到 img_size x img_size
                            anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                        size=args.img_size, mode='bilinear', align_corners=True)
                            # 对类别维 softmax 取异常通道（索引 1）
                            anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
                            # 转 numpy 存入列表
                            anomaly_maps.append(anomaly_map.cpu().numpy())
                    # 对所有层 zero-shot anomaly map 取均值（多层平均融合）
                    score_map_zero = np.mean(anomaly_maps, axis=0)
                    seg_score_map_zero.append(score_map_zero)
                


                else:
                    # 无像素级标注的任务（如 Chest, Histopathology 等），在检测头上做 few-shot + zero-shot 融合

                    # ---------------- few-shot, det head：基于 memory bank 的图像级异常分数 ----------------
                    anomaly_maps_few_shot = []
                    for idx, p in enumerate(det_patch_tokens_single):
                        # det_mem_features[idx]：该层支持样本特征 [N_mem, C]
                        # p：当前图像该层 patch 特征 [L, C]
                        if args.use_hyperbolic:
                            # ===== 双曲模式：使用向量化双曲距离 =====
                            dist = hyperbolic_distance_batch(det_mem_features[idx], p, ball)  # [N_mem, L]
                            height = int(np.sqrt(dist.shape[1]))  # patch 网格大小 H
                            # 取最近邻（最小距离）作为该 patch 的异常分数，应用温度系数缩放
                            anomaly_map_few_shot = torch.min(temperature * dist, dim=0)[0].reshape(1, 1, height, height)
                            anomaly_map_few_shot = F.interpolate(anomaly_map_few_shot,
                                                                    size=args.img_size, mode='bilinear', align_corners=True)
                            anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())
                        else:
                            # ===== 欧氏模式：使用余弦相似度 =====
                            cos = cos_sim(det_mem_features[idx], p)  # [N_mem, L]
                            height = int(np.sqrt(cos.shape[1]))  # patch 网格大小 H（L = H*H）
                            # 1 - cos 作为距离，并取最小相似度（最大距离）作为 patch 异常度
                            anomaly_map_few_shot = torch.min((1 - cos), 0)[0].reshape(1, 1, height, height)
                            anomaly_map_few_shot = F.interpolate(torch.tensor(anomaly_map_few_shot),
                                                                    size=args.img_size, mode='bilinear', align_corners=True)
                            anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())
                    # 对所有层 few-shot anomaly map 求和
                    anomaly_map_few_shot = np.sum(anomaly_maps_few_shot, axis=0)
                    # 图像级 few-shot anomaly score 取整图平均值
                    score_few_det = anomaly_map_few_shot.mean()
                    det_image_scores_few.append(score_few_det)

                    # ---------------- zero-shot, det head：基于文本头的图像级异常分数 ----------------
                    anomaly_score = 0
                    for layer in range(len(det_patch_tokens_single)):
                        if args.use_hyperbolic:
                            # ===== 双曲模式 =====
                            L, C = det_patch_tokens_single[layer].shape
                            
                            # 文本特征：[C, 2] -> [2, C]
                            text_h = text_features.T  # [2, C]
                            normal_text = text_h[0]  # [C]
                            abnormal_text = text_h[1]  # [C]
                            
                            # 向量化计算距离
                            dist_normal = ball.dist(det_patch_tokens_single[layer], normal_text)  # [L]
                            dist_abnormal = ball.dist(det_patch_tokens_single[layer], abnormal_text)  # [L]
                            
                            # 距离转 logits
                            logits_normal = -temperature * dist_normal
                            logits_abnormal = -temperature * dist_abnormal
                            
                            # Stack 成 [L, 2]
                            anomaly_map = torch.stack([logits_normal, logits_abnormal], dim=-1).unsqueeze(0)  # [1, L, 2]
                            # softmax 后取异常类（索引 1）概率
                            anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                            # 对所有 patch 求平均并累加
                            anomaly_score += anomaly_map.mean()
                        else:
                            # ===== 欧氏模式 =====
                            # 特征 L2 归一化
                            det_patch_tokens_single[layer] /= det_patch_tokens_single[layer].norm(dim=-1, keepdim=True)
                            # 与文本特征相乘，得到 [L, 2] logits
                            anomaly_map = (100.0 * det_patch_tokens_single[layer] @ text_features).unsqueeze(0)
                            # softmax 后取异常类（索引 1）概率
                            anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                            # 对所有 patch 求平均并累加
                            anomaly_score += anomaly_map.mean()
                    # 多层平均：除以层数，保存 zero-shot 图像级得分
                    anomaly_score = anomaly_score / len(det_patch_tokens_single)
                    det_image_scores_zero.append(anomaly_score.cpu().numpy())

                
                # 收集当前样本的 GT mask 和图像级 GT 标签
                gt_mask_list.append(mask[batch_idx].squeeze().cpu().detach().numpy())
                gt_list.extend(y[batch_idx:batch_idx+1].cpu().detach().numpy())
            

    # 将 GT 转为 numpy 数组
    gt_list = np.array(gt_list)
    gt_mask_list = np.asarray(gt_mask_list)
    gt_mask_list = (gt_mask_list>0).astype(np.int_)  # 再次确保为 0/1


    # 若当前任务有像素级标注，则基于分割头的 zero-shot + few-shot score map 计算 AUC
    if CLASS_INDEX[args.obj] > 0:

        seg_score_map_zero = np.array(seg_score_map_zero)  # zero-shot 得分图
        seg_score_map_few = np.array(seg_score_map_few)    # few-shot 得分图

        # 统一到 (N, H, W)，避免维度广播导致超大数组分配
        def _to_nhw(x):
            while x.ndim > 3 and x.shape[1] == 1:
                x = np.squeeze(x, axis=1)
            return x

        seg_score_map_zero = _to_nhw(seg_score_map_zero)
        seg_score_map_few = _to_nhw(seg_score_map_few)
        if seg_score_map_zero.shape != seg_score_map_few.shape:
            raise RuntimeError(
                f"Shape mismatch before fusion: zero={seg_score_map_zero.shape}, few={seg_score_map_few.shape}"
            )

        # 对 zero-shot 与 few-shot 得分图分别做逐图像 min-max 归一化
        seg_score_map_zero = normalize_map_per_image(seg_score_map_zero)
        seg_score_map_few = normalize_map_per_image(seg_score_map_few)
    
        # 融合像素级得分：0.5 * zero-shot + 0.5 * few-shot
        segment_scores = 0.5 * seg_score_map_zero + 0.5 * seg_score_map_few
        # 像素级 ROC AUC（展开所有像素）
        seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), segment_scores.flatten())
        print(f'{args.obj} pAUC : {round(seg_roc_auc,4)}')

        # 将每张图像的 score map 展平，取最大值作为图像级 anomaly score
        segment_scores_flatten = segment_scores.reshape(segment_scores.shape[0], -1)
        roc_auc_im = roc_auc_score(gt_list, np.max(segment_scores_flatten, axis=1))  # 图像级 ROC AUC
        print(f'{args.obj} AUC : {round(roc_auc_im,4)}')

        # 返回像素级 AUC + 图像级 AUC 作为综合指标
        return seg_roc_auc + roc_auc_im

    else:
        # 若当前任务没有像素级标注，则在检测头上做 zero-shot + few-shot 图像级融合

        det_image_scores_zero = np.array(det_image_scores_zero)  # zero-shot 图像级得分
        det_image_scores_few = np.array(det_image_scores_few)    # few-shot 图像级得分

        # 各自做 min-max 归一化
        det_image_scores_zero = (det_image_scores_zero - det_image_scores_zero.min()) / (det_image_scores_zero.max() - det_image_scores_zero.min())
        det_image_scores_few = (det_image_scores_few - det_image_scores_few.min()) / (det_image_scores_few.max() - det_image_scores_few.min())
    
        # 图像级得分融合：0.5 * zero-shot + 0.5 * few-shot
        image_scores = 0.5 * det_image_scores_zero + 0.5 * det_image_scores_few
        # 计算图像级 ROC AUC
        img_roc_auc_det = roc_auc_score(gt_list, image_scores)
        print(f'{args.obj} AUC : {round(img_roc_auc_det,4)}')

        # 返回图像级 AUC
        return img_roc_auc_det





if __name__ == '__main__':
    # 当该文件作为脚本直接运行时，执行 main()
    main()
