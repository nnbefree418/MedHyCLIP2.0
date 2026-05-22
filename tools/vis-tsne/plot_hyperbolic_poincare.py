#!/usr/bin/env python3
"""
Hyperbolic Poincaré Disk 可视化脚本

生成双曲空间的 Poincaré disk 图，展示：
- Normal patches (中心区域，绿色方块)
- Lesion patches (外圈区域，红色方块)
- Normal prompt (小半径，绿色大圆点)
- Abnormal prompt (大半径，红色大圆点)
- 单位圆边界
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.decomposition import PCA
import torch
import geoopt

# 导入共享工具
from vis_utils import (
    setup_seed, load_model, load_dataset, sample_images_from_dataset,
    extract_patch_features, encode_text_prompts, save_metadata
)


def hyperbolic_to_poincare_2d(features_euc, patch_labels, scale_x, scale_t, 
                               alpha_normal, alpha_lesion, whiten_tangent, 
                               pca_components, c=0.1):
    """
    将欧氏特征映射到 Poincaré disk (2D) - 增强版
    
    流程：
    1. Normalize 欧氏特征，并应用 scale_x/scale_t
    2. expmap0: 映射到高维 Poincaré ball
    3. 按类别应用 Möbius 径向缩放 (alpha_normal/alpha_lesion)
    4. logmap0: 映射回切空间 (欧氏)
    5. (可选) 白化切空间向量
    6. PCA: 降维并选择指定分量
    7. expmap0: 映射到 2D Poincaré disk
    
    Args:
        features_euc: [N, D] 欧氏特征 (前 N-2 是 patches，后 2 是 text)
        patch_labels: [N-2] patch 标签 (0=normal, 1=lesion)
        scale_x: patch embedding 的尺度系数
        scale_t: text embedding 的尺度系数
        alpha_normal: normal patches 的 Möbius 径向缩放
        alpha_lesion: lesion patches 的 Möbius 径向缩放
        whiten_tangent: 是否白化切空间
        pca_components: [comp1, comp2] PCA 分量索引
        c: Poincaré ball 曲率
        
    Returns:
        features_2d: [N, 2] 2D Poincaré disk 坐标
        ball_2d: geoopt.PoincareBall object
        stats: 统计信息字典
    """
    ball = geoopt.PoincareBall(c=c)
    N_total = features_euc.shape[0]
    N_patches = N_total - 2
    
    # 1. Normalize 并应用 scale
    features_norm = features_euc / (np.linalg.norm(features_euc, axis=1, keepdims=True) + 1e-8)
    
    # 分别对 patch 和 text 应用不同的 scale
    patch_scaled = features_norm[:N_patches] * scale_x
    text_scaled = features_norm[N_patches:] * scale_t
    features_scaled = np.vstack([patch_scaled, text_scaled])
    
    features_tensor = torch.from_numpy(features_scaled).float()
    
    print(f"Input scale: patch={scale_x}, text={scale_t}")
    print(f"Input norm range: [{np.linalg.norm(features_scaled, axis=1).min():.3f}, "
          f"{np.linalg.norm(features_scaled, axis=1).max():.3f}]")
    
    # 2. 映射到高维 Poincaré ball
    features_hyp = ball.expmap0(features_tensor)  # [N, D]
    
    # 检查 NaN/Inf
    if torch.isnan(features_hyp).any() or torch.isinf(features_hyp).any():
        print("WARNING: NaN/Inf detected after expmap0!")
    
    # 3. 按类别应用 Möbius 径向缩放（仅对 patches）
    if alpha_normal != 1.0 or alpha_lesion != 1.0:
        print(f"Applying Möbius scaling: normal={alpha_normal}, lesion={alpha_lesion}")
        
        for i in range(N_patches):
            if patch_labels[i] == 0:  # normal
                if alpha_normal != 1.0:
                    alpha_t = torch.tensor(alpha_normal, dtype=features_hyp.dtype)
                    features_hyp[i] = ball.mobius_scalar_mul(alpha_t, features_hyp[i])
            else:  # lesion
                if alpha_lesion != 1.0:
                    alpha_t = torch.tensor(alpha_lesion, dtype=features_hyp.dtype)
                    features_hyp[i] = ball.mobius_scalar_mul(alpha_t, features_hyp[i])
        
        # 检查缩放后的有效性
        if torch.isnan(features_hyp).any() or torch.isinf(features_hyp).any():
            print("WARNING: NaN/Inf detected after Möbius scaling!")
    
    # 计算高维双曲空间的 norm 统计
    hyp_norms = torch.norm(features_hyp, dim=1).cpu().numpy()
    print(f"Hyperbolic norm (high-dim): mean={hyp_norms.mean():.4f}, std={hyp_norms.std():.4f}, "
          f"min={hyp_norms.min():.4f}, max={hyp_norms.max():.4f}")
    
    # 4. 映射回切空间
    features_tangent = ball.logmap0(features_hyp)  # [N, D]
    features_tangent_np = features_tangent.cpu().numpy()
    
    # 5. (可选) 白化切空间
    if whiten_tangent:
        print("Applying tangent space whitening")
        mean = features_tangent_np.mean(axis=0, keepdims=True)
        std = features_tangent_np.std(axis=0, keepdims=True) + 1e-8
        features_tangent_np = (features_tangent_np - mean) / std
    
    # 6. PCA 降维并选择分量
    max_comp = max(pca_components) + 1
    pca = PCA(n_components=min(max_comp + 2, features_tangent_np.shape[1]), random_state=42)
    features_pca_all = pca.fit_transform(features_tangent_np)  # [N, n_components]
    
    print(f"PCA explained variance ratio (first 5): {pca.explained_variance_ratio_[:5]}")
    print(f"Selected PCA components: {pca_components}")
    
    # 选择指定的两个分量
    features_2d_tangent = features_pca_all[:, pca_components]  # [N, 2]
    
    # 7. 映射到 2D Poincaré disk
    features_2d_tensor = torch.from_numpy(features_2d_tangent).float()
    ball_2d = geoopt.PoincareBall(c=c)
    features_2d_hyp = ball_2d.expmap0(features_2d_tensor)  # [N, 2]
    
    features_2d = features_2d_hyp.cpu().numpy()
    
    # 检查最终结果
    if np.isnan(features_2d).any() or np.isinf(features_2d).any():
        print("WARNING: NaN/Inf in final 2D coordinates!")
    
    # 统计信息
    norms_2d = np.linalg.norm(features_2d, axis=1)
    stats = {
        'has_nan': bool(np.isnan(features_2d).any()),
        'has_inf': bool(np.isinf(features_2d).any()),
        'norm_mean': float(norms_2d.mean()),
        'norm_std': float(norms_2d.std()),
        'norm_min': float(norms_2d.min()),
        'norm_max': float(norms_2d.max()),
        'normal_norm_mean': float(norms_2d[:N_patches][patch_labels == 0].mean()) if (patch_labels == 0).any() else 0,
        'lesion_norm_mean': float(norms_2d[:N_patches][patch_labels == 1].mean()) if (patch_labels == 1).any() else 0
    }
    
    print(f"2D Poincaré norm: mean={stats['norm_mean']:.4f}, std={stats['norm_std']:.4f}, "
          f"min={stats['norm_min']:.4f}, max={stats['norm_max']:.4f}")
    print(f"  Normal patches norm: {stats['normal_norm_mean']:.4f}")
    print(f"  Lesion patches norm: {stats['lesion_norm_mean']:.4f}")
    
    return features_2d, ball_2d, stats


def adjust_text_radius_in_2d(text_feat_2d, scale_normal, scale_abnormal, c=0.1):
    """
    在 2D Poincaré disk 中调整 text prompts 的半径
    
    Args:
        text_feat_2d: [2, 2] 2D Poincaré disk 中的 text 坐标
        scale_normal: normal 半径缩放因子
        scale_abnormal: abnormal 半径缩放因子
        c: 曲率
        
    Returns:
        text_feat_adjusted: [2, 2] 调整后的坐标
    """
    ball_2d = geoopt.PoincareBall(c=c)
    
    text_tensor = torch.from_numpy(text_feat_2d).float()
    
    # 分离 normal 和 abnormal
    normal_feat = text_tensor[0]    # [2]
    abnormal_feat = text_tensor[1]  # [2]
    
    # Möbius scalar multiplication 调整半径
    scale_normal_t = torch.tensor(scale_normal, dtype=normal_feat.dtype)
    scale_abnormal_t = torch.tensor(scale_abnormal, dtype=abnormal_feat.dtype)
    
    normal_adjusted = ball_2d.mobius_scalar_mul(scale_normal_t, normal_feat)
    abnormal_adjusted = ball_2d.mobius_scalar_mul(scale_abnormal_t, abnormal_feat)
    
    text_adjusted = torch.stack([normal_adjusted, abnormal_adjusted], dim=0)
    
    return text_adjusted.cpu().numpy()


def plot_poincare_disk(patch_features_2d, patch_labels, text_features_2d, args, outdir, hyp_stats=None):
    """
    绘制 Poincaré disk 图
    
    Args:
        patch_features_2d: [N, 2] patch 在 2D Poincaré disk 的坐标
        patch_labels: [N] patch 标签
        text_features_2d: [2, 2] text 在 2D Poincaré disk 的坐标
        args: 命令行参数
        outdir: 输出目录
        hyp_stats: 双曲空间统计信息（可选）
    """
    print("\n=== Hyperbolic Poincaré Disk Visualization ===")
    
    # 打印统计信息
    if hyp_stats:
        print(f"Hyperbolic mapping statistics:")
        print(f"  Has NaN: {hyp_stats['has_nan']}")
        print(f"  Has Inf: {hyp_stats['has_inf']}")
        print(f"  Overall norm: mean={hyp_stats['norm_mean']:.4f}, std={hyp_stats['norm_std']:.4f}")
        print(f"  Normal patches mean norm: {hyp_stats['normal_norm_mean']:.4f}")
        print(f"  Lesion patches mean norm: {hyp_stats['lesion_norm_mean']:.4f}")
    
    # 绘图
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=150)
    
    # 绘制单位圆边界
    unit_circle = Circle((0, 0), 1.0, fill=False, edgecolor='black', linewidth=2, linestyle='--')
    ax.add_patch(unit_circle)
    
    # 绘制 normal patches (label=0)
    normal_mask = (patch_labels == 0)
    ax.scatter(
        patch_features_2d[normal_mask, 0],
        patch_features_2d[normal_mask, 1],
        c='green', marker='s', s=30, alpha=0.6,
        label=f'Normal patches ({normal_mask.sum()})'
    )
    
    # 绘制 lesion patches (label=1)
    lesion_mask = (patch_labels == 1)
    ax.scatter(
        patch_features_2d[lesion_mask, 0],
        patch_features_2d[lesion_mask, 1],
        c='red', marker='s', s=30, alpha=0.6,
        label=f'Lesion patches ({lesion_mask.sum()})'
    )
    
    # 绘制 text prompts (大圆点)
    ax.scatter(
        text_features_2d[0, 0], text_features_2d[0, 1],
        c='green', marker='o', s=200, edgecolors='black', linewidths=2,
        label='Normal prompt', zorder=10
    )
    ax.scatter(
        text_features_2d[1, 0], text_features_2d[1, 1],
        c='red', marker='o', s=200, edgecolors='black', linewidths=2,
        label='Abnormal prompt', zorder=10
    )
    
    # 计算并显示半径统计
    normal_radii = np.linalg.norm(patch_features_2d[normal_mask], axis=1) if normal_mask.any() else np.array([])
    lesion_radii = np.linalg.norm(patch_features_2d[lesion_mask], axis=1) if lesion_mask.any() else np.array([])
    
    print(f"\nFinal 2D visualization statistics:")
    print(f"  Total patches: {len(patch_labels)}, Normal: {normal_mask.sum()}, Lesion: {lesion_mask.sum()}")
    if len(normal_radii) > 0:
        print(f"  Normal patches radius: mean={normal_radii.mean():.4f}, std={normal_radii.std():.4f}, "
              f"min={normal_radii.min():.4f}, max={normal_radii.max():.4f}")
    if len(lesion_radii) > 0:
        print(f"  Lesion patches radius: mean={lesion_radii.mean():.4f}, std={lesion_radii.std():.4f}, "
              f"min={lesion_radii.min():.4f}, max={lesion_radii.max():.4f}")
    print(f"  Normal prompt radius: {np.linalg.norm(text_features_2d[0]):.4f}")
    print(f"  Abnormal prompt radius: {np.linalg.norm(text_features_2d[1]):.4f}")
    
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect('equal')
    ax.set_xlabel('Poincaré Disk X', fontsize=14)
    ax.set_ylabel('Poincaré Disk Y', fontsize=14)
    ax.set_title(f'Hyperbolic Poincaré Disk: {args.dataset} (c={args.hyperbolic_c})', fontsize=16)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 保存图片
    png_path = os.path.join(outdir, 'hyperbolic_poincare.png')
    pdf_path = os.path.join(outdir, 'hyperbolic_poincare.pdf')
    
    plt.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved Hyperbolic Poincaré plot:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description='Hyperbolic Poincaré Disk Visualization')
    
    # 数据集参数
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['Brain', 'Liver', 'Retina_RESC', 'Retina_OCT2017', 'Chest', 'Histopathology'],
                       help='Dataset name')
    parser.add_argument('--data_path', type=str, default='./data/',
                       help='Path to data directory')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Data split (only test is supported for now)')
    
    # 模型参数
    parser.add_argument('--ckpt', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--model_name', type=str, default='ViT-L-14-336',
                       help='CLIP model name')
    parser.add_argument('--img_size', type=int, default=240,
                       help='Image size')
    parser.add_argument('--features_list', type=int, nargs='+', default=[6, 12, 18, 24],
                       help='Adapter layer indices')
    parser.add_argument('--layer', type=int, default=-1,
                       help='Which adapter layer to use for patch features (-1=last)')
    
    # 采样参数
    parser.add_argument('--n_images', type=int, default=20,
                       help='Number of images to sample')
    parser.add_argument('--patches_per_image', type=int, default=10,
                       help='Number of patches per image')
    parser.add_argument('--thr_pos', type=float, default=0.3,
                       help='Threshold for lesion patches (mask ratio > thr_pos)')
    parser.add_argument('--thr_neg', type=float, default=0.05,
                       help='Threshold for non-lesion patches (mask ratio < thr_neg)')
    parser.add_argument('--max_samples', type=int, default=1000,
                       help='Maximum number of patches to prevent OOM')
    
    # Hyperbolic 参数
    parser.add_argument('--use_hyperbolic', action='store_true',
                       help='Whether the checkpoint uses hyperbolic adapters')
    parser.add_argument('--hyperbolic_c', type=float, default=0.1,
                       help='Curvature of Poincare ball')
    parser.add_argument('--scale_normal', type=float, default=0.2,
                       help='Radius scale for normal text (smaller = toward center)')
    parser.add_argument('--scale_abnormal', type=float, default=0.8,
                       help='Radius scale for abnormal text (larger = toward edge)')
    
    # 可视化映射参数（新增）
    parser.add_argument('--scale_x', type=float, default=1.0,
                       help='Scale factor for patch embeddings before expmap0')
    parser.add_argument('--scale_t', type=float, default=1.0,
                       help='Scale factor for text embeddings before expmap0')
    parser.add_argument('--alpha_normal', type=float, default=1.0,
                       help='Möbius radial scaling for normal patches (visualization only)')
    parser.add_argument('--alpha_lesion', type=float, default=1.0,
                       help='Möbius radial scaling for lesion patches (visualization only)')
    parser.add_argument('--whiten_tangent', action='store_true',
                       help='Apply whitening to tangent space before PCA')
    parser.add_argument('--pca_components', type=int, nargs=2, default=[0, 1],
                       help='PCA component indices to use for 2D projection (e.g., 0 1 or 0 2)')
    
    # 输出参数
    parser.add_argument('--outdir', type=str, required=True,
                       help='Output directory for plots and metadata')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # 设置随机种子
    setup_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.outdir, exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # 1. 加载模型
    print("\n=== Loading Model ===")
    model, clip_model = load_model(args, device)
    
    # 2. 加载数据集
    print("\n=== Loading Dataset ===")
    dataset, dataset_info = load_dataset(args)
    
    # 3. 采样图像
    print("\n=== Sampling Images ===")
    sampled_indices, sampled_data = sample_images_from_dataset(dataset, args)
    
    # 4. 提取 patch 特征 (欧氏空间)
    print("\n=== Extracting Patch Features ===")
    patch_features, patch_labels, patch_info, sampling_stats = extract_patch_features(
        model, clip_model, sampled_data, args, device
    )
    
    # 限制样本数量防止 OOM
    if len(patch_features) > args.max_samples:
        print(f"Warning: Too many patches ({len(patch_features)}), sampling {args.max_samples}")
        indices = np.random.choice(len(patch_features), args.max_samples, replace=False)
        patch_features = patch_features[indices]
        patch_labels = patch_labels[indices]
    
    # 5. 编码文本 prompts (欧氏空间)
    print("\n=== Encoding Text Prompts ===")
    text_features_euc, _, _ = encode_text_prompts(clip_model, args, device)
    text_feat_np = text_features_euc.cpu().numpy().T  # [2, D]
    
    # 6. 映射到双曲空间 (2D Poincaré disk)
    print("\n=== Mapping to Hyperbolic Space (2D Poincaré Disk) ===")
    print(f"Visualization parameters:")
    print(f"  scale_x={args.scale_x}, scale_t={args.scale_t}")
    print(f"  alpha_normal={args.alpha_normal}, alpha_lesion={args.alpha_lesion}")
    print(f"  whiten_tangent={args.whiten_tangent}")
    print(f"  pca_components={args.pca_components}")
    
    # 拼接 patch 和 text 特征
    all_features_euc = np.vstack([patch_features, text_feat_np])  # [N+2, D]
    
    # 映射到 2D Poincaré disk（使用新参数）
    all_features_2d, ball_2d, hyp_stats = hyperbolic_to_poincare_2d(
        all_features_euc, 
        patch_labels,
        scale_x=args.scale_x,
        scale_t=args.scale_t,
        alpha_normal=args.alpha_normal,
        alpha_lesion=args.alpha_lesion,
        whiten_tangent=args.whiten_tangent,
        pca_components=args.pca_components,
        c=args.hyperbolic_c
    )
    
    # 分离 patch 和 text
    patch_features_2d = all_features_2d[:-2, :]  # [N, 2]
    text_features_2d_raw = all_features_2d[-2:, :]  # [2, 2]
    
    # 调整 text prompts 的半径
    text_features_2d = adjust_text_radius_in_2d(
        text_features_2d_raw,
        scale_normal=args.scale_normal,
        scale_abnormal=args.scale_abnormal,
        c=args.hyperbolic_c
    )
    
    # 7. 绘制 Poincaré disk 图
    plot_poincare_disk(patch_features_2d, patch_labels, text_features_2d, args, args.outdir, hyp_stats)
    
    # 8. 保存 metadata
    print("\n=== Saving Metadata ===")
    save_metadata(args, args.outdir, dataset_info, sampling_stats, sampled_indices, sampled_data)
    
    print("\n=== Done! ===")


if __name__ == '__main__':
    main()

