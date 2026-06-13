# -*- coding: utf-8 -*-
import os  # 操作系统相关库
import argparse  # 命令行参数解析
import random  # Python 自带随机数库
import math  # 数学函数库
import numpy as np  # 数值计算库
import torch  # PyTorch 主库
from torch import nn  # 神经网络相关模块
from torch.nn import functional as F  # 一些常用函数接口（卷积、插值等）
from tqdm import tqdm  # 进度条显示
from scipy.ndimage import gaussian_filter  # 高斯滤波（本文件未使用）
from dataset.medical_few import MedDataset  # few-shot 医疗数据集
from CLIP.clip import create_model  # 创建 CLIP 模型
from CLIP.tokenizer import tokenize  # CLIP 文本 tokenizer（本文件未直接使用）
from CLIP.adapter import CLIP_Inplanted  # 在 CLIP 中插入适配器的封装模型
from PIL import Image  # 图像读写库（本文件未直接使用）
from sklearn.metrics import roc_auc_score, precision_recall_curve, pairwise  # 评价指标（pairwise 未使用）
from loss import FocalLoss, BinaryDiceLoss  # 自定义 Focal loss 和 Dice loss
from utils import augment, cos_sim, encode_text_with_prompt_ensemble, encode_text_with_hyperbolic_adjustment, hyperbolic_distance_batch  # 数据增强、余弦相似度、文本编码工具
from prompt import REAL_NAME  # 各任务对应的真实名称，用于文本 prompt
import geoopt  # 双曲几何库（用于 Hyper-MVFA）
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 关闭 tokenizer 并行，避免多进程冲突

import warnings  # 警告信息处理
warnings.filterwarnings("ignore")  # 忽略所有警告，清爽日志输出

# 判断是否有可用 GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")  # 有 GPU 就用 cuda:0，否则用 CPU

# 各数据集任务对应的索引（>0 表示有像素级标注，<=0 表示只有图像级标注）
CLASS_INDEX = {'Brain':3, 'Liver':2, 'Retina_RESC':1, 'Retina_OCT2017':-1, 'Chest':-2, 'Histopathology':-3}

def setup_seed(seed):
    """
    设置随机种子，保证实验可复现
    """
    torch.manual_seed(seed)  # CPU 随机种子
    torch.cuda.manual_seed_all(seed)  # 所有 GPU 的随机种子
    np.random.seed(seed)  # numpy 随机种子
    random.seed(seed)  # Python random 随机种子
    torch.backends.cudnn.deterministic = True  # cuDNN 使用确定性算法
    torch.backends.cudnn.benchmark = False  # 关闭 benchmark，避免非确定性行为



def main():
    """
    few-shot 训练主入口：
    1）构建 CLIP+适配器模型
    2）基于少量标注样本做数据增强得到训练集
    3）构建支持集特征内存库（memory bank）
    4）训练适配器并在测试集上评估
    """
    parser = argparse.ArgumentParser(description='Testing')  # 命令行参数解析器
    parser.add_argument('--model_name', type=str, default='ViT-L-14-336', help="ViT-B-16-plus-240, ViT-L-14-336")  # CLIP Backbone 名称
    parser.add_argument('--pretrain', type=str, default='openai', help="laion400m, openai")  # 预训练权重来源
    parser.add_argument('--obj', type=str, default='Liver')  # 当前实验对象（数据集/任务）
    parser.add_argument('--data_path', type=str, default='./data/')  # 数据路径
    parser.add_argument('--batch_size', type=int, default=1)  # 测试时的 batch size（few-shot 训练内部固定为 1）
    parser.add_argument('--save_model', type=int, default=1)  # 是否保存模型（1 表示保存）
    parser.add_argument('--save_path', type=str, default=None, help='checkpoint save dir')  # few-shot 模型保存路径
    parser.add_argument('--img_size', type=int, default=240)  # 输入图像的尺寸
    parser.add_argument("--epoch", type=int, default=50, help="epochs")  # 训练轮数
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")  # 学习率
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")  # 选择 CLIP 的哪些层作为特征
    parser.add_argument('--seed', type=int, default=111)  # 随机种子
    parser.add_argument('--shot', type=int, default=4)  # few-shot 样本数量（每类/每任务中的支持样本数）
    parser.add_argument('--iterate', type=int, default=0)  # 是否循环使用不同的 few-shot 组合（与数据集定义相关）
    # Hyper-MVFA 双曲模式参数
    parser.add_argument('--use_hyperbolic', action='store_true', help='Use hyperbolic adapters and distances')
    parser.add_argument('--hyperbolic_c', type=float, default=0.1, help='Curvature of Poincare ball')
    parser.add_argument('--scale_normal', type=float, default=0.1, help='Radius scale for normal text embeddings')
    parser.add_argument('--scale_abnormal', type=float, default=0.8, help='Radius scale for abnormal text embeddings')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for scaling hyperbolic distances to logits')
    # ========= 新增：tag，用于区分不同实验版本的 checkpoint =========
    parser.add_argument('--tag', type=str, default=None,
                        help='Optional tag for checkpoint naming, should match few-shot test script')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (epochs without improvement before stopping)')
    # ============================================================
    args = parser.parse_args()  # 解析命令行参数

    # ===== 自动根据是否使用双曲模式选择默认保存目录 =====
    if args.save_path is None:
        mode_tag = "few-shot-hyper" if args.use_hyperbolic else "few-shot-euclid"
        args.save_path = os.path.join("./ckpt", mode_tag)
    os.makedirs(args.save_path, exist_ok=True)

    # 设置随机种子，保证可复现
    setup_seed(args.seed)
    
    # 固定特征提取器：创建预训练 CLIP 模型（视觉编码器部分）
    clip_model = create_model(model_name=args.model_name,  # backbone 类型
                              img_size=args.img_size,      # 输入图片大小
                              device=device,               # 设备
                              pretrained=args.pretrain,    # 预训练权重来源
                              require_pretrained=True)     # 强制需要预训练权重
    clip_model.eval()  # CLIP 模型置为 eval 模式（不训练 CLIP 本体）

    # 在 CLIP 基础上插入适配器，构建 MVFA 模型
    model = CLIP_Inplanted(clip_model=clip_model,          # 预训练 CLIP 模型
                           features=args.features_list,     # 使用的特征层
                           use_hyperbolic=args.use_hyperbolic,  # 是否使用双曲适配器
                           hyperbolic_c=args.hyperbolic_c).to(device)  # 双曲空间曲率
    model.eval()  # 初始设为 eval 模式（仅训练适配器参数）

    # 将所有参数的 requires_grad 设为 True（但优化器只会更新 adapters）
    for name, param in model.named_parameters():
        param.requires_grad = True

    # 只对分割和检测适配器建立优化器（即只训练 adapters）
    seg_optimizer = torch.optim.Adam(list(model.seg_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))  # 分割适配器优化器
    det_optimizer = torch.optim.Adam(list(model.det_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))  # 检测适配器优化器


    # 加载测试数据集（其中内部包含 few-shot 支持样本）
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}  # 若有 GPU，开启多线程和固定内存
    test_dataset = MedDataset(args.data_path, args.obj, args.img_size, args.shot, args.iterate)  # few-shot 医疗数据集
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)  # 测试集 DataLoader（测试时固定为 batch_size=1）


    # few-shot 图像增强：对少量异常样本和正常样本做增强，构造训练数据
    augment_abnorm_img, augment_abnorm_mask = augment(test_dataset.fewshot_abnorm_img, test_dataset.fewshot_abnorm_mask)  # 增强异常图像及其 mask
    augment_normal_img, augment_normal_mask = augment(test_dataset.fewshot_norm_img)  # 增强正常图像（无 mask 返回）

    # 将增强后的异常和正常样本拼接，得到 few-shot 训练图像
    augment_fewshot_img = torch.cat([augment_abnorm_img, augment_normal_img], dim=0)
    # mask 也相应拼接（正常图像的 mask 通常为全 0 或占位）
    augment_fewshot_mask = torch.cat([augment_abnorm_mask, augment_normal_mask], dim=0)
    
    # few-shot 样本对应的图像级标签：异常为 1，正常为 0
    augment_fewshot_label = torch.cat([torch.Tensor([1] * len(augment_abnorm_img)), torch.Tensor([0] * len(augment_normal_img))], dim=0)

    # 基于增强后的 few-shot 图像、mask 和标签构造训练数据集
    train_dataset = torch.utils.data.TensorDataset(augment_fewshot_img, augment_fewshot_mask, augment_fewshot_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)  # few-shot 训练 DataLoader，batch_size=1


    # memory bank 构建：只用增强后的正常图像来构造支持集特征库
    support_dataset = torch.utils.data.TensorDataset(augment_normal_img)  # 只包含正常样本图像
    support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=1, shuffle=True, **kwargs)  # 支持集 DataLoader


    # 定义损失函数：像素级 Focal + Dice，用于分割；BCEWithLogits 用于图像级分类
    loss_focal = FocalLoss()  # 像素级 Focal Loss
    loss_dice = BinaryDiceLoss()  # 像素级 Dice Loss
    loss_bce = torch.nn.BCELoss()  # 图像级 BCE Loss（输入为 softmax 概率）


    # 编码文本 prompt，得到该任务对应的文本特征（形状 [dim, 2]）
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

    best_result = 0  # 记录最优结果，用于选择最好的 checkpoint
    no_improve_epochs = 0  # early stopping 计数器：记录连续未提升的 epoch 数

    # 开始训练多个 epoch
    for epoch in range(args.epoch):
        print('epoch ', epoch, ':')  # 打印当前 epoch

        loss_list = []  # 保存每个 batch 的 loss 用于统计
        for (image, gt, label) in train_loader:  # 遍历 few-shot 增强数据
            image = image.to(device)  # 将图像移到 device
            with torch.cuda.amp.autocast():  # 使用混合精度
                # 前向传播：得到全局输出（未使用）、分割 patch tokens 和检测 patch tokens
                _, seg_patch_tokens, det_patch_tokens = model(image)
                # 去掉 CLS token（索引 0），保留整个 batch，形状从 [B, L+1, C] 变为 [B, L, C]
                seg_patch_tokens = [p[:, 1:, :] for p in seg_patch_tokens]
                det_patch_tokens = [p[:, 1:, :] for p in det_patch_tokens]
                    
                # 图像级检测损失（det head）
                det_loss = 0
                image_label = label.to(device)  # few-shot 图像级标签（0/1），形状 [B]
                for layer in range(len(det_patch_tokens)):
                    if args.use_hyperbolic:
                        # ===== 双曲模式：使用向量化双曲距离计算 =====
                        B, L, C = det_patch_tokens[layer].shape
                        
                        # 将 patch 特征 reshape 为 [B*L, C]
                        patches_flat = det_patch_tokens[layer].reshape(B * L, C)
                        
                        # 文本特征：[C, 2] -> [2, C]
                        text_h = text_features.T  # [2, C]
                        normal_text = text_h[0]  # [C]
                        abnormal_text = text_h[1]  # [C]
                        
                        # 向量化计算距离
                        dist_normal = ball.dist(patches_flat, normal_text)  # [B*L]
                        dist_abnormal = ball.dist(patches_flat, abnormal_text)  # [B*L]
                        
                        # 距离转 logits
                        logits_normal = -args.temperature * dist_normal
                        logits_abnormal = -args.temperature * dist_abnormal
                        
                        # Stack 成 [B*L, 2]，然后 reshape 为 [B, L, 2]
                        anomaly_map = torch.stack([logits_normal, logits_abnormal], dim=-1)  # [B*L, 2]
                        anomaly_map = anomaly_map.view(B, L, 2)  # [B, L, 2]
                        
                        # 对类别维度做 softmax，取"异常"类（索引 1）的概率
                        anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]  # [B, L]
                        # 对所有 patch 取平均，得到图像级异常分数
                        anomaly_score = torch.mean(anomaly_map, dim=-1)  # [B]
                        # 数值保护，转 fp32（BCELoss 在 autocast 块内无条件报错，需在 autocast(enabled=False) 里调用）
                        anomaly_score = anomaly_score.float().clamp(1e-6, 1.0 - 1e-6)
                        with torch.cuda.amp.autocast(enabled=False):
                            det_loss += loss_bce(anomaly_score, image_label.float().view_as(anomaly_score))
                    else:
                        # ===== 欧氏模式（原版 MVFA）=====
                        # L2 归一化 patch 特征
                        det_patch_tokens[layer] = det_patch_tokens[layer] / det_patch_tokens[layer].norm(dim=-1, keepdim=True)
                        # 与文本特征相乘（相当于一个二分类 linear head），形状 [B, L, C] @ [C, 2] = [B, L, 2]
                        anomaly_map = 100.0 * det_patch_tokens[layer] @ text_features
                        # 对类别维度做 softmax，取"异常"类（索引 1）的概率，形状 [B, L]
                        anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                        # 对所有 patch 平均得到图像级异常分数，形状 [B]
                        anomaly_score = torch.mean(anomaly_map, dim=-1)
                        # 数值保护，转 fp32
                        anomaly_score = anomaly_score.float().clamp(1e-6, 1.0 - 1e-6)
                        with torch.cuda.amp.autocast(enabled=False):
                            det_loss += loss_bce(anomaly_score, image_label.float().view_as(anomaly_score))

                # 若该任务有像素级标注（CLASS_INDEX > 0），则训练分割头
                if CLASS_INDEX[args.obj] > 0:
                    # 像素级分割损失（seg head）
                    seg_loss = 0
                    mask = gt.squeeze(0).to(device)  # GT mask，形状 [1, B, 1, H, W] -> [B, 1, H, W]
                    # 二值化 mask：>0.5 置为 1，否则为 0
                    mask[mask > 0.5], mask[mask <= 0.5] = 1, 0
                    for layer in range(len(seg_patch_tokens)):
                        if args.use_hyperbolic:
                            # ===== 双曲模式：使用向量化双曲距离计算 =====
                            B, L, C = seg_patch_tokens[layer].shape
                            H = int(np.sqrt(L))  # patch 网格尺寸 H x H
                            
                            # 将 patch 特征 reshape 为 [B*L, C]
                            patches_flat = seg_patch_tokens[layer].reshape(B * L, C)
                            
                            # 文本特征：[C, 2] -> [2, C]
                            text_h = text_features.T  # [2, C]
                            normal_text = text_h[0]  # [C]
                            abnormal_text = text_h[1]  # [C]
                            
                            # 向量化计算距离
                            dist_normal = ball.dist(patches_flat, normal_text)  # [B*L]
                            dist_abnormal = ball.dist(patches_flat, abnormal_text)  # [B*L]
                            
                            # 距离转 logits
                            logits_normal = -args.temperature * dist_normal
                            logits_abnormal = -args.temperature * dist_abnormal
                            
                            # Stack 成 [B*L, 2]，然后 reshape 为 [B, L, 2]
                            anomaly_map = torch.stack([logits_normal, logits_abnormal], dim=-1)  # [B*L, 2]
                            anomaly_map = anomaly_map.view(B, L, 2)  # [B, L, 2]
                            
                            # 将 [B, L, 2] 变为 [B, 2, H, H] 的空间图，并插值到目标分辨率
                            anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                        size=args.img_size, mode='bilinear', align_corners=True)
                            # 像素级 softmax，得到每个像素属于两类的概率
                            anomaly_map = torch.softmax(anomaly_map, dim=1)
                            # 使用 Focal Loss 约束整个 2 通道概率图
                            seg_loss += loss_focal(anomaly_map, mask)
                            # 使用 DiceLoss 约束异常通道（通道 1）与 GT mask
                            seg_loss += loss_dice(anomaly_map[:, 1, :, :], mask)
                        else:
                            # ===== 欧氏模式（原版 MVFA）=====
                            # L2 归一化分割 patch 特征
                            seg_patch_tokens[layer] = seg_patch_tokens[layer] / seg_patch_tokens[layer].norm(dim=-1, keepdim=True)
                            # 计算与文本特征的相似度，形状 [B, L, C] @ [C, 2] = [B, L, 2]
                            anomaly_map = 100.0 * seg_patch_tokens[layer] @ text_features
                            B, L, C = anomaly_map.shape  # B: batch, L: patch 数量, C: 类别数=2
                            H = int(np.sqrt(L))  # 假定 patch 总数为 H*H，因此 H = sqrt(L)
                            # 将 [B, L, C] reshape 为 [B, C, H, H]，再插值到 img_size×img_size
                            anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                        size=args.img_size, mode='bilinear', align_corners=True)
                            # 对类别维度 softmax，得到每个像素属于两类的概率
                            anomaly_map = torch.softmax(anomaly_map, dim=1)
                            # Focal Loss 作用在 2 通道概率图上
                            seg_loss += loss_focal(anomaly_map, mask)
                            # Dice Loss 只作用在异常通道（通道 1）上
                            seg_loss += loss_dice(anomaly_map[:, 1, :, :], mask)
                    
                    # 总损失 = 像素级分割损失 + 图像级检测损失
                    loss = seg_loss + det_loss
                    loss.requires_grad_(True)  # 确保 loss 需要梯度（默认如此，此处保持原逻辑）

                    # 清空分割和检测两个优化器的梯度
                    seg_optimizer.zero_grad()
                    det_optimizer.zero_grad()
                    # 反向传播
                    loss.backward()
                    # 更新分割和检测适配器参数
                    seg_optimizer.step()
                    det_optimizer.step()

                else:
                    # 若该任务没有像素级标注，只训练检测头
                    loss = det_loss
                    loss.requires_grad_(True)  # 同样确保需要梯度
                    det_optimizer.zero_grad()  # 只更新检测适配器
                    loss.backward()
                    det_optimizer.step()

                # 记录本 batch 的损失
                loss_list.append(loss.item())

        # 输出当前 epoch 的平均训练损失
        print("Loss: ", np.mean(loss_list))


        # 构建 memory bank：基于支持集（正常样本）提取的特征
        seg_features = []  # 暂存每张支持图像的分割特征（多层）
        det_features = []  # 暂存每张支持图像的检测特征（多层）
        for image in support_loader:
            image = image[0].to(device)  # support_loader 输出是 (img,)，取第 0 个元素
            with torch.no_grad():  # 构建 memory bank 时不需要梯度
                _, seg_patch_tokens, det_patch_tokens = model(image)
                # 去除 CLS token（索引 0），保持与测试时一致（使用 p[0, 1:, :]）
                seg_patch_tokens = [p[0, 1:, :].contiguous() for p in seg_patch_tokens]
                det_patch_tokens = [p[0, 1:, :].contiguous() for p in det_patch_tokens]
                seg_features.append(seg_patch_tokens)  # list: 每张图像一个 list（按层）
                det_features.append(det_patch_tokens)

        # 将所有支持图像的特征在样本维度 concat，得到每层的 memory feature
        seg_mem_features = [torch.cat([seg_features[j][i] for j in range(len(seg_features))], dim=0) for i in range(len(seg_features[0]))]
        det_mem_features = [torch.cat([det_features[j][i] for j in range(len(det_features))], dim=0) for i in range(len(det_features[0]))]
        

        # 在测试集上评估当前模型性能（zero-shot + few-shot 结合）
        result = test(args, model, test_loader, text_features, seg_mem_features, det_mem_features, ball, args.temperature)
        if result >= best_result:
            # 若结果优于历史最优，则更新 best_result，并根据需要保存模型
            best_result = result
            no_improve_epochs = 0
            print(f'[EarlyStopping] Epoch {epoch}: score improved to {round(result, 4)}')
            if args.save_model == 1:
                # ========= 按照是否有 tag 决定保存的 ckpt 文件名 =========
                if args.tag is None:
                    ckpt_name = f'{args.obj}.pth'
                else:
                    ckpt_name = f'{args.obj}_{args.tag}.pth'
                ckp_path = os.path.join(args.save_path, ckpt_name)
                print(f"Saving checkpoint to: {ckp_path}")
                # ========================================================
                torch.save({'seg_adapters': model.seg_adapters.state_dict(),     # 保存分割适配器参数
                            'det_adapters': model.det_adapters.state_dict()},    # 保存检测适配器参数
                            ckp_path)
        else:
            no_improve_epochs += 1
            print(f'[EarlyStopping] Epoch {epoch}: no improvement ({no_improve_epochs}/{args.patience}), best={round(best_result, 4)}')
            if no_improve_epochs >= args.patience:
                print(f'[EarlyStopping] Patience {args.patience} reached, stopping early at epoch {epoch}.')
                break




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
    - pixel-level 任务（CLASS_INDEX > 0）：seg head few-shot + seg head zero-shot 融合
    - image-level 任务（CLASS_INDEX <= 0）：det head few-shot + det head zero-shot 融合
    注意：temperature 参数应与训练时保持一致，默认值 1.0 与 args.temperature 默认值相同
    """
    gt_list = []
    gt_mask_list = []

    det_image_scores_zero = []
    det_image_scores_few = []
    
    seg_score_map_zero = []
    seg_score_map_few = []

    for (image, y, mask) in tqdm(test_loader):
        image = image.to(device)
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

        with torch.no_grad(), torch.cuda.amp.autocast():
            _, seg_patch_tokens, det_patch_tokens = model(image)
            # 和原始 MVFA 一样：batch_size=1，直接取 [0, 1:, :]
            seg_patch_tokens = [p[0, 1:, :] for p in seg_patch_tokens]
            det_patch_tokens = [p[0, 1:, :] for p in det_patch_tokens]

            # ======================= 有像素级标注：分割头 =======================
            if CLASS_INDEX[args.obj] > 0:

                # ---------- few-shot, seg head ----------
                anomaly_maps_few_shot = []
                for idx, p in enumerate(seg_patch_tokens):
                    # seg_mem_features[idx]: [N_mem, C]
                    # p: [L, C]
                    if args.use_hyperbolic:
                        # 双曲 few-shot：用 hyperbolic_distance_batch
                        dist = hyperbolic_distance_batch(seg_mem_features[idx], p, ball)  # [N_mem, L]
                        height = int(np.sqrt(dist.shape[1]))
                        # 取最近邻（最小距离）作为异常分数
                        anomaly_map_few_shot = torch.min(temperature * dist, dim=0)[0].reshape(1, 1, height, height)
                        anomaly_map_few_shot = F.interpolate(
                            anomaly_map_few_shot,
                            size=args.img_size,
                            mode='bilinear',
                            align_corners=True
                        )
                    else:
                        # 欧氏 few-shot：用余弦相似度
                        cos = cos_sim(seg_mem_features[idx], p)  # [N_mem, L]
                        height = int(np.sqrt(cos.shape[1]))
                        anomaly_map_few_shot = torch.min((1 - cos), dim=0)[0].reshape(1, 1, height, height)
                        anomaly_map_few_shot = F.interpolate(
                            anomaly_map_few_shot,
                            size=args.img_size,
                            mode='bilinear',
                            align_corners=True
                        )
                    anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())

                score_map_few = np.sum(anomaly_maps_few_shot, axis=0)
                seg_score_map_few.append(score_map_few)

                # ---------- zero-shot, seg head ----------
                anomaly_maps = []
                for layer in range(len(seg_patch_tokens)):
                    if args.use_hyperbolic:
                        # 双曲 zero-shot：用文本 hyper 向量 + 温度
                        L, C = seg_patch_tokens[layer].shape
                        H = int(np.sqrt(L))

                        text_h = text_features.T  # [2, C]
                        normal_text = text_h[0]   # [C]
                        abnormal_text = text_h[1] # [C]

                        dist_normal = ball.dist(seg_patch_tokens[layer], normal_text)      # [L]
                        dist_abnormal = ball.dist(seg_patch_tokens[layer], abnormal_text)  # [L]

                        logits_normal = -temperature * dist_normal
                        logits_abnormal = -temperature * dist_abnormal

                        anomaly_map = torch.stack(
                            [logits_normal, logits_abnormal], dim=-1
                        ).unsqueeze(0)  # [1, L, 2]

                        B = 1
                        anomaly_map = F.interpolate(
                            anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                            size=args.img_size,
                            mode='bilinear',
                            align_corners=True
                        )
                        anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
                        anomaly_maps.append(anomaly_map.cpu().numpy())
                    else:
                        # 欧氏 zero-shot：保持原逻辑
                        seg_patch_tokens[layer] /= seg_patch_tokens[layer].norm(dim=-1, keepdim=True)
                        anomaly_map = (100.0 * seg_patch_tokens[layer] @ text_features)
                        L, C = anomaly_map.shape  # 修复：测试时没有 batch 维度
                        H = int(np.sqrt(L))
                        anomaly_map = F.interpolate(
                            anomaly_map.permute(1, 0).view(1, 2, H, H),  # 添加 batch 维度
                            size=args.img_size,
                            mode='bilinear',
                            align_corners=True
                        )
                        anomaly_map = torch.softmax(anomaly_map, dim=1)[0, 1, :, :]  # 取第 0 个 batch
                        anomaly_maps.append(anomaly_map.cpu().numpy())

                score_map_zero = np.mean(anomaly_maps, axis=0)
                seg_score_map_zero.append(score_map_zero)

            # ======================= 只有图像级标注：检测头 =======================
            else:
                # ---------- few-shot, det head ----------
                anomaly_maps_few_shot = []
                for idx, p in enumerate(det_patch_tokens):
                    if args.use_hyperbolic:
                        dist = hyperbolic_distance_batch(det_mem_features[idx], p, ball)  # [N_mem, L]
                        height = int(np.sqrt(dist.shape[1]))
                        # 取最近邻（最小距离）作为异常分数
                        anomaly_map_few_shot = torch.min(temperature * dist, dim=0)[0].reshape(1, 1, height, height)
                        anomaly_map_few_shot = F.interpolate(
                            anomaly_map_few_shot,
                            size=args.img_size,
                            mode='bilinear',
                            align_corners=True
                        )
                    else:
                        cos = cos_sim(det_mem_features[idx], p)
                        height = int(np.sqrt(cos.shape[1]))
                        anomaly_map_few_shot = torch.min((1 - cos), dim=0)[0].reshape(1, 1, height, height)
                        anomaly_map_few_shot = F.interpolate(
                            anomaly_map_few_shot,
                            size=args.img_size,
                            mode='bilinear',
                            align_corners=True
                        )
                    anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())

                anomaly_map_few_shot = np.sum(anomaly_maps_few_shot, axis=0)
                score_few_det = anomaly_map_few_shot.mean()
                det_image_scores_few.append(score_few_det)

                # ---------- zero-shot, det head ----------
                anomaly_score = 0
                for layer in range(len(det_patch_tokens)):
                    if args.use_hyperbolic:
                        L, C = det_patch_tokens[layer].shape

                        text_h = text_features.T
                        normal_text = text_h[0]
                        abnormal_text = text_h[1]

                        dist_normal = ball.dist(det_patch_tokens[layer], normal_text)     # [L]
                        dist_abnormal = ball.dist(det_patch_tokens[layer], abnormal_text) # [L]

                        logits_normal = -temperature * dist_normal
                        logits_abnormal = -temperature * dist_abnormal

                        anomaly_map = torch.stack(
                            [logits_normal, logits_abnormal], dim=-1
                        ).unsqueeze(0)  # [1, L, 2]
                        anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                        anomaly_score += anomaly_map.mean()
                    else:
                        det_patch_tokens[layer] /= det_patch_tokens[layer].norm(dim=-1, keepdim=True)
                        anomaly_map = (100.0 * det_patch_tokens[layer] @ text_features).unsqueeze(0)
                        anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                        anomaly_score += anomaly_map.mean()

                anomaly_score = anomaly_score / len(det_patch_tokens)
                det_image_scores_zero.append(anomaly_score.cpu().numpy())

        # ===== 每个 batch（这里就是一张图）收集 GT =====
        gt_mask_list.append(mask.squeeze().cpu().detach().numpy())
        gt_list.extend(y.cpu().detach().numpy())

    # ======================= 汇总与指标计算 =======================
    gt_list = np.array(gt_list)
    gt_mask_list = np.asarray(gt_mask_list)
    gt_mask_list = (gt_mask_list > 0).astype(np.int_)

    if CLASS_INDEX[args.obj] > 0:
        seg_score_map_zero = np.array(seg_score_map_zero)
        seg_score_map_few = np.array(seg_score_map_few)

        # 统一到 (N, H, W)，避免 (N,1,H,W) 与 (N,H,W) 广播成 (N,N,H,W) 导致内存爆炸
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

        seg_score_map_zero = normalize_map_per_image(seg_score_map_zero)
        seg_score_map_few = normalize_map_per_image(seg_score_map_few)

        segment_scores = 0.5 * seg_score_map_zero + 0.5 * seg_score_map_few
        seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), segment_scores.flatten())
        print(f'{args.obj} pAUC : {round(seg_roc_auc, 4)}')

        segment_scores_flatten = segment_scores.reshape(segment_scores.shape[0], -1)
        roc_auc_im = roc_auc_score(gt_list, np.max(segment_scores_flatten, axis=1))
        print(f'{args.obj} AUC : {round(roc_auc_im, 4)}')

        return seg_roc_auc + roc_auc_im
    else:
        det_image_scores_zero = np.array(det_image_scores_zero)
        det_image_scores_few = np.array(det_image_scores_few)

        det_image_scores_zero = (det_image_scores_zero - det_image_scores_zero.min()) / (
            det_image_scores_zero.max() - det_image_scores_zero.min() + 1e-8
        )
        det_image_scores_few = (det_image_scores_few - det_image_scores_few.min()) / (
            det_image_scores_few.max() - det_image_scores_few.min() + 1e-8
        )

        image_scores = 0.5 * det_image_scores_zero + 0.5 * det_image_scores_few
        img_roc_auc_det = roc_auc_score(gt_list, image_scores)
        print(f'{args.obj} AUC : {round(img_roc_auc_det, 4)}')

        return img_roc_auc_det




if __name__ == '__main__':
    # 当该文件作为脚本运行时，执行 main()
    main()
