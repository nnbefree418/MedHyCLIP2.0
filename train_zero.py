# -*- coding: utf-8 -*-
# 导入系统和常用库
import os
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter
from dataset.medical_zero import MedTestDataset, MedTrainDataset  # 零样本医疗数据集（训练/测试）
from CLIP.clip import create_model                               # 创建预训练 CLIP 模型
from CLIP.tokenizer import tokenize                              # CLIP 文本 tokenizer（此文件中未直接使用）
from CLIP.adapter import CLIP_Inplanted                          # 在 CLIP 中插入适配器的模型封装
from PIL import Image                                             # 图像读取/处理（此文件中未直接使用）
from sklearn.metrics import precision_recall_curve                # 精度-召回曲线（此文件中未直接使用）
from loss import FocalLoss, BinaryDiceLoss                        # 自定义 Focal loss 与 Dice loss（用于分割）
from utils import augment, encode_text_with_prompt_ensemble, encode_text_with_hyperbolic_adjustment       # 数据增强与文本编码工具
from prompt import REAL_NAME                                      # 各类医学数据集的真实名称字典（用于构造文本 prompt）
import geoopt  # 双曲几何库（用于 Hyper-MVFA）

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import warnings
warnings.filterwarnings("ignore")  # 忽略警告信息，避免日志过于冗长

# 判断是否有可用的 GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")  # 将在 main() 中被 DDP local_rank 覆盖

# 类别与索引之间的映射，用于对不同数据集/任务进行编码
CLASS_INDEX = {'Brain':3, 'Liver':2, 'Retina_RESC':1, 'Retina_OCT2017':-1, 'Chest':-2, 'Histopathology':-3}
# 反向映射：由索引恢复数据集名称
CLASS_INDEX_INV = {3:'Brain', 2:'Liver', 1:'Retina_RESC', -1:'Retina_OCT2017', -2:'Chest', -3:'Histopathology'}


def setup_seed(seed):
    """
    设置随机种子，保证实验可复现
    """
    torch.manual_seed(seed)                    # 设置 CPU 上的随机种子
    torch.cuda.manual_seed_all(seed)          # 设置所有 GPU 上的随机种子
    np.random.seed(seed)                      # numpy 随机种子
    random.seed(seed)                         # python 内置 random 随机种子
    torch.backends.cudnn.deterministic = True # 让 cuDNN 的结果可复现（牺牲少量性能）
    torch.backends.cudnn.benchmark = False    # 禁用 benchmark，避免不同输入形状带来的非确定性


def main():
    """
    训练入口：零样本 MVFA 适配器的训练与在线测试
    """
    # 构建命令行参数解析器
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_name', type=str, default='ViT-L-14-336', help="ViT-B-16-plus-240, ViT-L-14-336")  # CLIP backbone 名称
    parser.add_argument('--pretrain', type=str, default='openai', help="laion400m, openai")                        # 预训练权重来源
    parser.add_argument('--obj', type=str, default='Retina_RESC')                                                   # 当前任务/数据集名称
    parser.add_argument('--data_path', type=str, default='./data/')                                                 # 数据根路径
    parser.add_argument('--batch_size', type=int, default=16)                                                        # 训练 batch 大小（此脚本内又在 DataLoader 里固定为1）
    parser.add_argument('--img_size', type=int, default=240)                                                        # 输入图像缩放大小
    parser.add_argument("--epoch", type=int, default=50, help="epochs")                                             # 训练轮数
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="learning rate")                        # 学习率
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")      # 使用哪些层的 patch token
    parser.add_argument('--seed', type=int, default=111)                                                            # 随机种子
    # Hyper-MVFA 双曲模式参数
    parser.add_argument('--use_hyperbolic', action='store_true', help='Use hyperbolic adapters and distances')      # 是否启用双曲模式
    parser.add_argument('--hyperbolic_c', type=float, default=0.1, help='Curvature of Poincare ball')              # 双曲空间曲率
    parser.add_argument('--scale_normal', type=float, default=0.1, help='Radius scale for normal text embeddings')  # normal 文本半径缩放
    parser.add_argument('--scale_abnormal', type=float, default=0.8, help='Radius scale for abnormal text embeddings')  # abnormal 文本半径缩放
    parser.add_argument('--temperature', type=float, default=20.0, help='Temperature for scaling hyperbolic distances to logits')  # 双曲距离到 logits 的温度缩放因子
    parser.add_argument('--save_path', type=str, default=None, help='checkpoint save dir')                          # checkpoint 保存路径
    # ========= 新增：tag，用于区分不同实验版本的 zero-shot checkpoint =========
    parser.add_argument('--tag', type=str, default=None,
                        help='Optional tag for checkpoint naming, should match test_zero*.py')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (epochs without improvement before stopping)')
    # ==========================================================
    args = parser.parse_args()                                                                                      # 解析参数

    # ===== DDP 初始化：每个进程绑定自己的 GPU =====
    global device
    import datetime
    dist.init_process_group(backend='nccl',
                            timeout=datetime.timedelta(hours=2))  # 延长至 2h，防止大测试集 epoch-end 评估超时
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = dist.get_world_size()
    is_main = (dist.get_rank() == 0)
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    # ====================================================

    # ===== 自动根据是否使用双曲模式选择默认保存目录 =====
    if args.save_path is None:
        mode_tag = "zero-shot-hyper" if args.use_hyperbolic else "zero-shot-euclid"
        args.save_path = os.path.join("./ckpt", mode_tag)
    os.makedirs(args.save_path, exist_ok=True)

    # 设置随机种子，保证可复现性
    setup_seed(args.seed)
    
    # 构建并固定特征提取器（CLIP）
    clip_model = create_model(model_name=args.model_name,                      # 指定 CLIP backbone 结构
                              img_size=args.img_size,                          # 输入图像尺寸
                              device=device,                                   # 运行设备
                              pretrained=args.pretrain,                        # 预训练权重来源
                              require_pretrained=True)                         # 强制需要预训练权重
    clip_model.eval()                                                          # 将 CLIP 模型切换到推理模式

    # 在 CLIP 模型上插入适配器，构建 MVFA 的主模型
    model = CLIP_Inplanted(clip_model=clip_model,                              # 冻结/固定的 CLIP 模型
                           features=args.features_list,                         # 指定使用的中间层
                           use_hyperbolic=args.use_hyperbolic,                  # 是否使用双曲适配器
                           hyperbolic_c=args.hyperbolic_c).to(device)           # 双曲空间曲率
    model.eval()                                                               # 初始设为 eval 模式（后续只训练 adapters）

    # 遍历模型所有参数，将 requires_grad 置为 True（但优化器只会更新 adapters 的参数）
    for name, param in model.named_parameters():
        param.requires_grad = True

    # DDP 包裹：每张卡都有完整的模型副本，梯度自动跨卡同步
    # find_unused_parameters=True：seg/det 分支在不同批次可能不同时激活
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    raw_model = model.module  # 始终通过 raw_model 访问 adapters（不经过 DDP 包裹）
    if is_main:
        print(f'[DDP] Using {world_size} GPUs')

    # 仅对分割与检测适配器建立优化器（即只训练 adapters）
    seg_optimizer = torch.optim.Adam(list(raw_model.seg_adapters.parameters()),    # 分割适配器参数
                                     lr=args.learning_rate,                        # 学习率
                                     betas=(0.5, 0.999))                           # Adam 的动量参数
    det_optimizer = torch.optim.Adam(list(raw_model.det_adapters.parameters()),    # 检测适配器参数
                                     lr=args.learning_rate,
                                     betas=(0.5, 0.999))

    # 加载数据集和 DataLoader
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}        # 设置为 0 避免 /tmp 空间不足问题
    train_dataset = MedTrainDataset(args.data_path, args.obj, args.img_size, args.batch_size)  # 训练集
    # DDP：DistributedSampler 把 1640 个预批次均分给各卡，每卡处理 1640/world_size 个批次
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size,
                                       rank=dist.get_rank(), shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=1,
                                               sampler=train_sampler,         # 用 sampler 代替 shuffle=True
                                               **kwargs)

    test_dataset = MedTestDataset(args.data_path, args.obj, args.img_size)    # 测试集（仅 rank 0 使用）
    test_loader = torch.utils.data.DataLoader(test_dataset,                   # 测试 DataLoader
                                              batch_size=1,                   # 测试时也按单张图像处理
                                              shuffle=False,                  # 测试集不打乱
                                              **kwargs)

    # 定义损失函数
    loss_focal = FocalLoss()                                                  # 像素级 Focal Loss
    loss_dice = BinaryDiceLoss()                                              # 像素级 Dice Loss
    loss_bce = torch.nn.BCELoss()                                             # 图像级 BCE Loss（输入为 softmax 概率）

    # 文本特征列表，初始化一个占位元素（索引 0 不用）
    text_feature_list = [0]
    ball_list = [None]  # 保存每个任务对应的 ball 对象（双曲模式时使用）
    
    # 预先编码不同数据集/任务的文本 prompt（使用多提示集成）
    with torch.cuda.amp.autocast(), torch.no_grad():                          # 使用混合精度 + 不求梯度，提高效率
        for i in [1,2,3,-3,-2,-1]:                                            # 遍历 CLASS_INDEX 中用到的索引
            # 根据索引映射回任务名称，再从 REAL_NAME 中取真实名称列表进行 prompt 编码
            if args.use_hyperbolic:
                # 使用双曲模式：调用双曲文本编码 + 半径调整
                text_feature, ball = encode_text_with_hyperbolic_adjustment(
                    clip_model,                                               # CLIP 模型
                    REAL_NAME[CLASS_INDEX_INV[i]],                            # 对应任务的真实名称列表
                    device,                                                   # 运行设备
                    use_hyperbolic=True,                                      # 启用双曲模式
                    c=args.hyperbolic_c,                                      # 曲率
                    scale_normal=args.scale_normal,                           # normal 半径缩放
                    scale_abnormal=args.scale_abnormal                        # abnormal 半径缩放
                )
                ball_list.append(ball)                                        # 保存 ball 对象
            else:
                # 使用欧氏模式：调用原始文本编码
                text_feature = encode_text_with_prompt_ensemble(clip_model,   # CLIP 模型
                                                             REAL_NAME[CLASS_INDEX_INV[i]],  # 对应任务的真实名称列表
                                                                 device)      # 运行设备
                ball_list.append(None)                                        # 欧氏模式不需要 ball
            
            text_feature_list.append(text_feature)                            # 追加到特征列表中

    # 记录当前最优评分，用于保存最佳模型
    save_score = 0.0
    no_improve_epochs = 0  # early stopping 计数器：记录连续未提升的 epoch 数

    # 训练多个 epoch
    for epoch in range(args.epoch):
        # DDP：通知 sampler 当前 epoch，保证不同 epoch 间的 shuffle 差异
        train_sampler.set_epoch(epoch)

        if is_main:
            print('epoch', epoch, ':')                                        # 只由 rank 0 打印

        loss_list = []                                                        # 保存当前 epoch 的 batch loss
        for (image, image_label, mask, seg_idx) in tqdm(train_loader, disable=not is_main):

            image = image.squeeze(0).to(device)                               # 去掉 DataLoader 维度并移到 device
            seg_idx = seg_idx.item()                                         # seg_idx: 当前样本所属任务/类别索引

            # 前向传播部分使用混合精度
            with torch.cuda.amp.autocast():
                # 模型前向：返回全局输出、分割 patch tokens、检测 patch tokens
                _, seg_patch_tokens, det_patch_tokens = model(image)
                # 去掉 CLS token（索引 0），保留整个 batch，形状从 [B, L+1, C] 变为 [B, L, C]
                seg_patch_tokens = [p[:, 1:, :] for p in seg_patch_tokens]
                det_patch_tokens = [p[:, 1:, :] for p in det_patch_tokens]

                # 图像级损失（检测任务）
                det_loss = 0
                image_label = image_label.squeeze(0).to(device)              # 图像级标签，形状 [1, B] -> [B]

                # 遍历所有使用的层，对每层 patch token 计算图像级 anomaly score
                for layer in range(len(det_patch_tokens)):
                    if args.use_hyperbolic:
                        # ===== 双曲模式：使用向量化双曲距离计算 =====
                        # det_patch_tokens[layer]: [B, L, C] - 双曲空间特征
                        # text_feature_list[seg_idx]: [C, 2] - 双曲空间文本特征
                        B, L, C = det_patch_tokens[layer].shape
                        ball = ball_list[seg_idx]  # 获取对应的 ball 对象
                        
                        # 将 patch 特征 reshape 为 [B*L, C]
                        patches_flat = det_patch_tokens[layer].reshape(B * L, C)
                        
                        # 文本特征：[C, 2] -> [2, C]
                        text_h = text_feature_list[seg_idx].T  # [2, C]
                        normal_text = text_h[0]    # [C]
                        abnormal_text = text_h[1]  # [C]
                        
                        # 向量化计算距离（利用 ball.dist 的广播）
                        dist_normal = ball.dist(patches_flat, normal_text)      # [B*L]
                        dist_abnormal = ball.dist(patches_flat, abnormal_text)  # [B*L]
                        
                        # 距离转 logits：负距离（距离越小，logits 越大）
                        # 添加温度缩放因子，使 logits 范围与欧氏模式接近
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
                        # L2 归一化特征，便于与文本特征做余弦相似度
                        det_patch_tokens[layer] = det_patch_tokens[layer] / det_patch_tokens[layer].norm(dim=-1, keepdim=True)
                        # 与对应的文本特征相乘（相当于线性分类头），形状 [B, L, C] @ [C, 2] = [B, L, 2]
                        anomaly_map = 100.0 * det_patch_tokens[layer] @ text_feature_list[seg_idx]
                        # 对类别维度做 softmax，取第 2 类（索引 1，对应"异常"）的概率，形状 [B, L]
                        anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                        # 对所有 patch 取平均，得到图像级异常分数，形状 [B]
                        anomaly_score = torch.mean(anomaly_map, dim=-1)
                        # 数值保护，转 fp32
                        anomaly_score = anomaly_score.float().clamp(1e-6, 1.0 - 1e-6)
                        with torch.cuda.amp.autocast(enabled=False):
                            det_loss += loss_bce(anomaly_score, image_label.float().view_as(anomaly_score))

                # seg_idx > 0 表示该任务有像素级标注（如 Retina_RESC, Liver, Brain 等）
                if seg_idx > 0:
                    # 像素级损失（分割任务）
                    seg_loss = 0
                    mask = mask.squeeze(0).to(device)                         # 取出 mask，形状 [1, B, 1, H, W] -> [B, 1, H, W]
                    # 将 mask 二值化：>0.5 为 1，否则为 0
                    mask[mask > 0.5], mask[mask <= 0.5] = 1, 0
                    # 遍历各层的分割 patch token，生成像素级 anomaly map
                    for layer in range(len(seg_patch_tokens)):
                        if args.use_hyperbolic:
                            # ===== 双曲模式：使用向量化双曲距离计算 =====
                            # seg_patch_tokens[layer]: [B, L, C] - 双曲空间特征
                            B, L, C = seg_patch_tokens[layer].shape
                            ball = ball_list[seg_idx]  # 获取对应的 ball 对象
                            H = int(np.sqrt(L))  # patch 网格尺寸 H x H
                            
                            # 将 patch 特征 reshape 为 [B*L, C]
                            patches_flat = seg_patch_tokens[layer].reshape(B * L, C)
                            
                            # 文本特征：[C, 2] -> [2, C]
                            text_h = text_feature_list[seg_idx].T  # [2, C]
                            normal_text = text_h[0]    # [C]
                            abnormal_text = text_h[1]  # [C]
                            
                            # 向量化计算距离
                            dist_normal = ball.dist(patches_flat, normal_text)      # [B*L]
                            dist_abnormal = ball.dist(patches_flat, abnormal_text)  # [B*L]
                            
                            # 距离转 logits：负距离
                            # 添加温度缩放因子，使 logits 范围与欧氏模式接近
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
                            # 特征 L2 归一化
                            seg_patch_tokens[layer] = seg_patch_tokens[layer] / seg_patch_tokens[layer].norm(dim=-1, keepdim=True)
                            # 计算与文本特征的相似度，形状 [B, L, C] @ [C, 2] = [B, L, 2]
                            anomaly_map = 100.0 * seg_patch_tokens[layer] @ text_feature_list[seg_idx]
                            B, L, C = anomaly_map.shape                           # B: batch, L: patch 数量, C: 类别数(2)
                            H = int(np.sqrt(L))                                   # 假设 patch 数为 H*H，因此 H = sqrt(L)
                            # 将 [B, L, C] 变为 [B, C, H, H] 的空间图，并插值到目标分辨率 img_size
                            anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                        size=args.img_size, mode='bilinear', align_corners=True)
                            # 像素级 softmax，得到每个像素属于两类的概率
                            anomaly_map = torch.softmax(anomaly_map, dim=1)
                            # 使用 Focal Loss 约束整个 2 通道概率图
                            seg_loss += loss_focal(anomaly_map, mask)
                            # 使用 DiceLoss 约束异常通道（通道 1）与 GT mask
                            seg_loss += loss_dice(anomaly_map[:, 1, :, :], mask)
                    
                    # 总损失 = 像素级分割损失 + 图像级检测损失
                    loss = seg_loss + det_loss  # = focal(seg_out, mask) + bce(det_out, y)
                    loss.requires_grad_(True)                                   # 确保 loss 需要梯度（通常默认即可）

                    # 清空两个优化器的梯度
                    seg_optimizer.zero_grad()
                    det_optimizer.zero_grad()
                    # 反向传播
                    loss.backward()
                    # 分别更新分割与检测适配器
                    seg_optimizer.step()
                    det_optimizer.step()

                else:
                    # 如果 seg_idx <= 0，表示该任务没有像素级标注，只进行图像级优化
                    loss = det_loss
                    loss.requires_grad_(True)                                   # 确保 loss 需要梯度
                    det_optimizer.zero_grad()                                   # 只更新检测适配器
                    loss.backward()
                    det_optimizer.step()

                # 记录当前 batch 的损失，用于 epoch 结束时统计
                loss_list.append(loss.item())

        # DDP：DistributedSampler 负责每 epoch 的 shuffle（通过 set_epoch），无需手动重建 DataLoader

        if is_main:
            # 打印当前 epoch 的平均训练损失
            print("Loss: ", np.mean(loss_list))

            # ===== Early Stopping：epoch 结束时由 rank 0 做一次完整评估 =====
            epoch_score = test(args, raw_model, test_loader,
                               text_feature_list[CLASS_INDEX[args.obj]],
                               ball_list[CLASS_INDEX[args.obj]],
                               args.temperature)
            if epoch_score >= save_score:
                save_score = epoch_score
                no_improve_epochs = 0
                if args.tag is None:
                    ckpt_name = f'{args.obj}.pth'
                else:
                    ckpt_name = f'{args.obj}_{args.tag}.pth'
                ckp_path = os.path.join(args.save_path, ckpt_name)
                torch.save({'seg_adapters': raw_model.seg_adapters.state_dict(),
                            'det_adapters': raw_model.det_adapters.state_dict()},
                            ckp_path)
                print(f'[EarlyStopping] Epoch {epoch}: score improved to {round(epoch_score, 4)}, saved to {ckp_path}')
            else:
                no_improve_epochs += 1
                print(f'[EarlyStopping] Epoch {epoch}: no improvement ({no_improve_epochs}/{args.patience}), best={round(save_score, 4)}')
            # ======================================================

        # 将 no_improve_epochs 从 rank 0 广播到所有进程，统一 early stopping 决策
        stop_tensor = torch.tensor([no_improve_epochs], dtype=torch.long, device=device)
        dist.broadcast(stop_tensor, src=0)
        no_improve_epochs = stop_tensor.item()

        if no_improve_epochs >= args.patience:
            if is_main:
                print(f'[EarlyStopping] Patience {args.patience} reached, stopping early at epoch {epoch}.')
            break

        dist.barrier()  # 下一 epoch 开始前所有进程同步

    dist.destroy_process_group()

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
    在测试集上评估模型的图像级 AUC 和像素级 AUC（若有分割标注）
    
    Args:
        args: 命令行参数
        seg_model: 模型
        test_loader: 测试数据加载器
        text_features: 文本特征 [C, 2]
        ball: PoincareBall 对象（双曲模式时使用，欧氏模式为 None）
    """
    gt_list = []             # 存储图像级 GT 标签
    gt_mask_list = []        # 存储像素级 GT mask
    image_scores = []        # 存储图像级预测分数
    segment_scores = []      # 存储像素级预测分数（score map）
    
    # 遍历测试集
    for (image, y, mask) in tqdm(test_loader):
        image = image.to(device)                             # 将图像移动到 device
        # 将 mask 二值化：>0.5 为 1，否则为 0
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

        # 测试阶段不计算梯度，并开启混合精度
        with torch.no_grad(), torch.cuda.amp.autocast():
            # 前向得到分割与检测的 patch tokens
            _, ori_seg_patch_tokens, ori_det_patch_tokens = seg_model(image)
            
            # 遍历 batch 中的每个样本
            batch_size_current = image.shape[0]
            for batch_idx in range(batch_size_current):
                # 提取当前样本的 tokens，去掉 CLS token（索引 0）
                ori_seg_patch_tokens_single = [p[batch_idx, 1:, :] for p in ori_seg_patch_tokens]
                ori_det_patch_tokens_single = [p[batch_idx, 1:, :] for p in ori_det_patch_tokens]
                
                # 图像级分数计算
                anomaly_score = 0
                patch_tokens = ori_det_patch_tokens_single.copy()       # 复制一份检测 patch token
                for layer in range(len(patch_tokens)):
                    if args.use_hyperbolic:
                        # ===== 双曲模式 =====
                        # patch_tokens[layer]: [L, C] - 双曲空间特征
                        L, C = patch_tokens[layer].shape
                        
                        # 文本特征：[C, 2] -> [2, C]
                        text_h = text_features.T  # [2, C]
                        normal_text = text_h[0]  # [C]
                        abnormal_text = text_h[1]  # [C]
                        
                        # 向量化计算距离
                        dist_normal = ball.dist(patch_tokens[layer], normal_text)  # [L]
                        dist_abnormal = ball.dist(patch_tokens[layer], abnormal_text)  # [L]
                        
                        # 距离转 logits
                        logits_normal = -temperature * dist_normal
                        logits_abnormal = -temperature * dist_abnormal
                        
                        # Stack 成 [L, 2]
                        anomaly_map = torch.stack([logits_normal, logits_abnormal], dim=-1).unsqueeze(0)  # [1, L, 2]
                        # 对类别做 softmax，并取"异常"类别概率
                        anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                        # 对所有 patch 求平均，累加到 anomaly_score
                        anomaly_score += anomaly_map.mean()
                    else:
                        # ===== 欧氏模式 =====
                        # L2 归一化特征
                        patch_tokens[layer] /= patch_tokens[layer].norm(dim=-1, keepdim=True)
                        # 与文本特征做矩阵乘法，得到 [L, 2] logits
                        anomaly_map = (100.0 * patch_tokens[layer] @ text_features).unsqueeze(0)
                        # 对类别做 softmax，并取"异常"类别概率
                        anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                        # 对所有 patch 求平均，累加到 anomaly_score
                        anomaly_score += anomaly_map.mean()
                # 多层平均：除以层数后存入列表
                anomaly_score = anomaly_score / len(patch_tokens)
                image_scores.append(anomaly_score.cpu())

                # 像素级分数计算
                patch_tokens = ori_seg_patch_tokens_single              # 使用分割分支的 patch token
                anomaly_maps = []                                # 保存各层的 anomaly map
                for layer in range(len(patch_tokens)):
                    if args.use_hyperbolic:
                        # ===== 双曲模式 =====
                        # patch_tokens[layer]: [L, C] - 双曲空间特征
                        L, C = patch_tokens[layer].shape
                        H = int(np.sqrt(L))  # patch 网格尺寸 H x H
                        
                        # 文本特征：[C, 2] -> [2, C]
                        text_h = text_features.T  # [2, C]
                        normal_text = text_h[0]  # [C]
                        abnormal_text = text_h[1]  # [C]
                        
                        # 向量化计算距离
                        dist_normal = ball.dist(patch_tokens[layer], normal_text)  # [L]
                        dist_abnormal = ball.dist(patch_tokens[layer], abnormal_text)  # [L]
                        
                        # 距离转 logits
                        logits_normal = -temperature * dist_normal
                        logits_abnormal = -temperature * dist_abnormal
                        
                        # Stack 成 [L, 2]
                        anomaly_map = torch.stack([logits_normal, logits_abnormal], dim=-1).unsqueeze(0)  # [1, L, 2]
                        B = 1
                        # 将 [1, L, 2] 变形并插值到 img_size×img_size
                        anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                    size=args.img_size, mode='bilinear', align_corners=True)
                        # 对类别做 softmax，得到"异常"通道的概率 map
                        anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
                        # 转为 numpy 并加入 list
                        anomaly_maps.append(anomaly_map.cpu().numpy())
                    else:
                        # ===== 欧氏模式 =====
                        # L2 归一化
                        patch_tokens[layer] /= patch_tokens[layer].norm(dim=-1, keepdim=True)
                        # 与文本特征做矩阵乘法
                        anomaly_map = (100.0 * patch_tokens[layer] @ text_features).unsqueeze(0)
                        B, L, C = anomaly_map.shape                  # B: batch, L: patch 数, C: 类别数(2)
                        H = int(np.sqrt(L))                          # 假定 patch 数为 H*H
                        # 将 [B, L, C] 变形并插值到 img_size×img_size
                        anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                    size=args.img_size, mode='bilinear', align_corners=True)
                        # 对类别做 softmax，得到"异常"通道的概率 map
                        anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
                        # 转为 numpy 并加入 list
                        anomaly_maps.append(anomaly_map.cpu().numpy())
                # 将不同层的 anomaly map 取均值，得到最终像素级 score map
                final_score_map = np.mean(anomaly_maps, axis=0)
                
                # 收集当前样本的 GT mask（转为 numpy）、图像级 GT y 和像素级预测 score map
                gt_mask_list.append(mask[batch_idx].squeeze().cpu().detach().numpy())
                gt_list.extend(y[batch_idx:batch_idx+1].cpu().detach().numpy())
                segment_scores.append(final_score_map)
        
    # 将收集到的列表转换为 numpy 数组
    gt_list = np.array(gt_list)                            # 图像级 GT 标签数组
    gt_mask_list = np.asarray(gt_mask_list)                # 像素级 GT mask 数组
    gt_mask_list = (gt_mask_list>0).astype(np.int_)        # 再次保证为 0/1 整型

    segment_scores = np.array(segment_scores)              # 像素级预测分数数组
    image_scores = np.array(image_scores)                  # 图像级预测分数数组

    # 像素级逐图归一化，图像级全局归一化
    segment_scores = normalize_map_per_image(segment_scores)
    image_scores = (image_scores - image_scores.min()) / (image_scores.max() - image_scores.min())

    # 图像级 ROC AUC（检测）
    img_roc_auc_det = roc_auc_score(gt_list, image_scores)
    print(f'{args.obj} AUC : {round(img_roc_auc_det,4)}')

    # 若当前任务有像素级标注（对应的 CLASS_INDEX > 0），则同时计算像素级 AUC
    if CLASS_INDEX[args.obj] > 0:
        seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), segment_scores.flatten())
        print(f'{args.obj} pAUC : {round(seg_roc_auc,4)}')
        # 训练时用 “像素级 AUC + 图像级 AUC” 作为综合指标
        return seg_roc_auc + img_roc_auc_det
    else:
        # 若无像素级标注，则只返回图像级 AUC 作为指标
        return img_roc_auc_det

# 主入口：运行脚本时从这里开始执行
if __name__ == '__main__':
    main()
