"""
共享工具函数：用于两个可视化脚本的公共逻辑
包括：模型加载、数据集采样、patch 提取、特征提取等
"""
import os
import sys
import json
import random
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from dataset.medical_zero import MedTestDataset, CLASS_INDEX
from CLIP.clip import create_model
from CLIP.adapter import CLIP_Inplanted
from CLIP.tokenizer import tokenize
from utils import encode_text_with_prompt_ensemble, encode_text_with_hyperbolic_adjustment
from prompt import REAL_NAME
import geoopt


def setup_seed(seed):
    """设置随机种子以保证可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(args, device):
    """
    加载 MedHyCLIP 模型和 checkpoint
    
    Args:
        args: 命令行参数
        device: torch.device
        
    Returns:
        model: CLIP_Inplanted 模型
        clip_model: 原始 CLIP 模型
    """
    # 创建 CLIP 基础模型
    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained='openai',
        require_pretrained=True
    )
    clip_model.eval()
    
    # 创建带 adapter 的模型
    model = CLIP_Inplanted(
        clip_model=clip_model,
        features=args.features_list,
        use_hyperbolic=args.use_hyperbolic,
        hyperbolic_c=args.hyperbolic_c
    ).to(device)
    model.eval()
    
    # 加载 checkpoint
    ckpt_path = args.ckpt
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.seg_adapters.load_state_dict(checkpoint["seg_adapters"])
    model.det_adapters.load_state_dict(checkpoint["det_adapters"])
    
    return model, clip_model


def load_dataset(args):
    """
    加载数据集
    
    Args:
        args: 命令行参数
        
    Returns:
        dataset: MedTestDataset
        dataset_info: 数据集信息字典
    """
    dataset = MedTestDataset(
        dataset_path=args.data_path,
        class_name=args.dataset,
        resize=args.img_size
    )
    
    # 统计数据集信息
    normal_count = sum([1 for y in dataset.y if y == 0])
    abnormal_count = sum([1 for y in dataset.y if y == 1])
    has_masks = CLASS_INDEX[args.dataset] > 0
    
    dataset_info = {
        'name': args.dataset,
        'total_images': len(dataset),
        'normal_images': normal_count,
        'abnormal_images': abnormal_count,
        'has_pixel_masks': has_masks,
        'image_size': args.img_size
    }
    
    print(f"Dataset: {args.dataset}")
    print(f"  Total images: {len(dataset)}")
    print(f"  Normal: {normal_count}, Abnormal: {abnormal_count}")
    print(f"  Has pixel masks: {has_masks}")
    
    return dataset, dataset_info


def sample_images_from_dataset(dataset, args):
    """
    从数据集中采样图像
    
    Args:
        dataset: MedTestDataset
        args: 命令行参数
        
    Returns:
        sampled_indices: 采样的图像索引列表
        sampled_data: 采样的数据列表 [(img, label, mask, file_path), ...]
    """
    # 分别获取 normal 和 abnormal 的索引
    normal_indices = [i for i, y in enumerate(dataset.y) if y == 0]
    abnormal_indices = [i for i, y in enumerate(dataset.y) if y == 1]
    
    # 从每类中采样
    n_normal = min(args.n_images // 2, len(normal_indices))
    n_abnormal = min(args.n_images // 2, len(abnormal_indices))
    
    sampled_normal = random.sample(normal_indices, n_normal)
    sampled_abnormal = random.sample(abnormal_indices, n_abnormal)
    
    sampled_indices = sampled_normal + sampled_abnormal
    
    sampled_data = []
    for idx in sampled_indices:
        img_tensor, label, mask = dataset[idx]
        file_path = dataset.x[idx]
        sampled_data.append((img_tensor, label, mask, file_path))
    
    print(f"Sampled {len(sampled_indices)} images: {n_normal} normal + {n_abnormal} abnormal")
    
    return sampled_indices, sampled_data


def extract_patches_from_image(img_tensor, mask, label, patches_per_image, 
                                thr_pos=0.3, thr_neg=0.05, patch_size=24):
    """
    从单张图像中提取 patch 位置
    
    Args:
        img_tensor: [3, H, W] 图像张量
        mask: [1, H, W] mask 张量 (normal 图为全 0)
        label: 图像标签 (0=normal, 1=abnormal)
        patches_per_image: 每张图抽取的 patch 数量
        thr_pos: lesion patch 的阈值 (mask 占比 > thr_pos)
        thr_neg: non-lesion patch 的阈值 (mask 占比 < thr_neg)
        patch_size: patch grid 大小 (ViT-L-14-336 的 patch size 是 14, 336/14=24)
        
    Returns:
        patch_indices: 采样的 patch 索引列表 [(i, j), ...]
        patch_labels: patch 标签列表 [0 or 1, ...]  (0=normal/non-lesion, 1=lesion)
    """
    H, W = img_tensor.shape[1], img_tensor.shape[2]
    n_patches_h = patch_size
    n_patches_w = patch_size
    
    patch_h = H // n_patches_h
    patch_w = W // n_patches_w
    
    if label == 0:
        # Normal 图：随机抽取 patches
        all_patches = [(i, j) for i in range(n_patches_h) for j in range(n_patches_w)]
        sampled = random.sample(all_patches, min(patches_per_image, len(all_patches)))
        patch_indices = sampled
        patch_labels = [0] * len(sampled)
    else:
        # Abnormal 图：基于 mask 进行区分
        mask_np = mask.squeeze(0).numpy()  # [H, W]
        
        lesion_patches = []
        nonlesion_patches = []
        
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                patch_mask = mask_np[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
                mask_ratio = patch_mask.mean()
                
                if mask_ratio > thr_pos:
                    lesion_patches.append((i, j))
                elif mask_ratio < thr_neg:
                    nonlesion_patches.append((i, j))
        
        # 至少一半来自 lesion_patches
        n_lesion = max(patches_per_image // 2, 1)
        n_nonlesion = patches_per_image - n_lesion
        
        sampled_lesion = random.sample(lesion_patches, min(n_lesion, len(lesion_patches)))
        sampled_nonlesion = random.sample(nonlesion_patches, min(n_nonlesion, len(nonlesion_patches)))
        
        # 如果 lesion 不足，从 nonlesion 补充
        if len(sampled_lesion) < n_lesion and len(nonlesion_patches) > len(sampled_nonlesion):
            additional = min(n_lesion - len(sampled_lesion), 
                           len(nonlesion_patches) - len(sampled_nonlesion))
            remaining = [p for p in nonlesion_patches if p not in sampled_nonlesion]
            sampled_nonlesion.extend(random.sample(remaining, additional))
        
        patch_indices = sampled_lesion + sampled_nonlesion
        patch_labels = [1] * len(sampled_lesion) + [0] * len(sampled_nonlesion)
    
    return patch_indices, patch_labels


def extract_patch_features(model, clip_model, sampled_data, args, device):
    """
    提取所有 patch 的特征
    
    Args:
        model: CLIP_Inplanted 模型
        clip_model: 原始 CLIP 模型
        sampled_data: 采样的数据列表
        args: 命令行参数
        device: torch device
        
    Returns:
        all_patch_features: [N_patches, D] 所有 patch 的特征
        all_patch_labels: [N_patches] 所有 patch 的标签 (0=normal, 1=lesion)
        all_patch_info: patch 信息列表 [(img_idx, patch_i, patch_j, label), ...]
        sampling_stats: 采样统计信息
    """
    all_patch_features = []
    all_patch_labels = []
    all_patch_info = []
    
    normal_patch_count = 0
    lesion_patch_count = 0
    
    with torch.no_grad():
        for img_idx, (img_tensor, label, mask, file_path) in enumerate(sampled_data):
            # 提取图像的 patch features (先forward获取实际的grid大小)
            img_batch = img_tensor.unsqueeze(0).to(device)  # [1, 3, H, W]
            
            # Forward 获取 patch tokens
            pooled, seg_tokens, det_tokens = model(img_batch)
            
            # 使用指定层的 det_tokens
            # det_tokens 是列表，每个元素是 [1, L+1, C]
            # 我们取最后一层（或指定层）
            if args.layer >= 0 and args.layer < len(det_tokens):
                patch_tokens = det_tokens[args.layer]  # [1, L+1, C]
            else:
                patch_tokens = det_tokens[-1]  # 默认最后一层
            
            # patch_tokens: [1, L+1, C], 去掉 CLS token (第0个)
            patch_tokens = patch_tokens[:, 1:, :]  # [1, L, C]
            
            # 动态计算实际的 grid 大小
            num_patches = patch_tokens.shape[1]  # L (实际的patch数量)
            grid_size = int(num_patches ** 0.5)  # 假设是正方形grid
            
            if grid_size * grid_size != num_patches:
                print(f"Warning: num_patches={num_patches} is not a perfect square!")
                grid_size = int(num_patches ** 0.5)
            
            # 采样 patch 位置 (使用动态的grid_size)
            patch_indices, patch_labels = extract_patches_from_image(
                img_tensor, mask, label, 
                args.patches_per_image,
                thr_pos=args.thr_pos,
                thr_neg=args.thr_neg,
                patch_size=grid_size  # 使用实际的grid大小
            )
            
            # 提取指定 patch 的特征
            for (pi, pj), plabel in zip(patch_indices, patch_labels):
                patch_idx = pi * grid_size + pj  # 使用动态grid_size
                patch_feat = patch_tokens[0, patch_idx, :]  # [C]
                
                all_patch_features.append(patch_feat.cpu().numpy())
                all_patch_labels.append(plabel)
                all_patch_info.append((img_idx, pi, pj, plabel))
                
                if plabel == 0:
                    normal_patch_count += 1
                else:
                    lesion_patch_count += 1
    
    all_patch_features = np.array(all_patch_features)  # [N, 768]
    all_patch_labels = np.array(all_patch_labels)
    
    sampling_stats = {
        'total_patches': len(all_patch_labels),
        'normal_patches': normal_patch_count,
        'lesion_patches': lesion_patch_count,
        'thr_pos': args.thr_pos,
        'thr_neg': args.thr_neg
    }
    
    print(f"Extracted {len(all_patch_features)} patch features:")
    print(f"  Normal/non-lesion patches: {normal_patch_count}")
    print(f"  Lesion patches: {lesion_patch_count}")
    
    return all_patch_features, all_patch_labels, all_patch_info, sampling_stats


def encode_text_prompts(clip_model, args, device):
    """
    编码文本 prompts
    
    Args:
        clip_model: CLIP 模型
        args: 命令行参数
        device: torch device
        
    Returns:
        text_features_euc: [D, 2] Euclidean 文本特征
        text_features_hyp: [D, 2] or None Hyperbolic 文本特征 (如果使用双曲模式)
        ball: PoincareBall object or None
    """
    obj_name = REAL_NAME[args.dataset]
    
    with torch.no_grad():
        # 获取欧氏特征
        text_features_euc = encode_text_with_prompt_ensemble(
            clip_model, obj_name, device
        )  # [D, 2]
        
        # 如果使用双曲模式，也获取双曲特征
        if args.use_hyperbolic:
            text_features_hyp, ball = encode_text_with_hyperbolic_adjustment(
                clip_model, obj_name, device,
                use_hyperbolic=True,
                c=args.hyperbolic_c,
                scale_normal=args.scale_normal,
                scale_abnormal=args.scale_abnormal
            )
            return text_features_euc, text_features_hyp, ball
        else:
            return text_features_euc, None, None


def save_metadata(args, outdir, dataset_info, sampling_stats, sampled_indices, sampled_data):
    """
    保存实验 metadata 到 JSON 文件
    
    Args:
        args: 命令行参数
        outdir: 输出目录
        dataset_info: 数据集信息
        sampling_stats: 采样统计信息
        sampled_indices: 采样的图像索引
        sampled_data: 采样的数据
    """
    metadata = {
        'dataset': args.dataset,
        'checkpoint': args.ckpt,
        'seed': args.seed,
        'n_images': args.n_images,
        'patches_per_image': args.patches_per_image,
        'image_size': args.img_size,
        'model_name': args.model_name,
        'features_list': args.features_list,
        'layer': args.layer,
        'use_hyperbolic': args.use_hyperbolic,
        'hyperbolic_c': args.hyperbolic_c if args.use_hyperbolic else None,
        'scale_normal': getattr(args, 'scale_normal', None),
        'scale_abnormal': getattr(args, 'scale_abnormal', None),
        # 新增可视化映射参数
        'scale_x': getattr(args, 'scale_x', None),
        'scale_t': getattr(args, 'scale_t', None),
        'alpha_normal': getattr(args, 'alpha_normal', None),
        'alpha_lesion': getattr(args, 'alpha_lesion', None),
        'whiten_tangent': getattr(args, 'whiten_tangent', None),
        'pca_components': getattr(args, 'pca_components', None),
        'dataset_info': dataset_info,
        'sampling_stats': sampling_stats,
        'sampled_image_indices': sampled_indices,
        'sampled_image_files': [data[3] for data in sampled_data]
    }
    
    metadata_path = os.path.join(outdir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to: {metadata_path}")
