#!/usr/bin/env python3
"""
Euclidean t-SNE 可视化脚本

生成欧氏空间的 t-SNE 散点图，展示：
- Normal patches (绿色方块)
- Lesion patches (红色方块)
- Normal prompt (绿色大圆点)
- Abnormal prompt (红色大圆点)
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch

# 导入共享工具
from vis_utils import (
    setup_seed, load_model, load_dataset, sample_images_from_dataset,
    extract_patch_features, encode_text_prompts, save_metadata
)


def plot_tsne_euclidean(patch_features, patch_labels, text_features, args, outdir):
    """
    绘制欧氏空间 t-SNE 图
    
    Args:
        patch_features: [N, D] patch 特征
        patch_labels: [N] patch 标签 (0=normal, 1=lesion)
        text_features: [D, 2] 文本特征 ([:, 0]=normal, [:, 1]=abnormal)
        args: 命令行参数
        outdir: 输出目录
    """
    print("\n=== Euclidean t-SNE Visualization ===")
    
    # 1. 准备数据：L2 normalize
    patch_features_norm = patch_features / (np.linalg.norm(patch_features, axis=1, keepdims=True) + 1e-8)
    
    # 文本特征转为 numpy 并 normalize
    text_feat_np = text_features.cpu().numpy().T  # [2, D] -> [D, 2]^T = [2, D]
    text_feat_norm = text_feat_np / (np.linalg.norm(text_feat_np, axis=1, keepdims=True) + 1e-8)
    
    # 2. 拼接所有数据
    all_features = np.vstack([patch_features_norm, text_feat_norm])  # [N+2, D]
    
    print(f"Running t-SNE on {all_features.shape[0]} points with {all_features.shape[1]} dimensions")
    
    # 3. t-SNE 降维
    tsne = TSNE(
        n_components=2,
        init='pca',
        random_state=args.seed,
        perplexity=min(args.perplexity, all_features.shape[0] - 1),
        n_iter=args.n_iter,
        verbose=1
    )
    embeddings_2d = tsne.fit_transform(all_features)  # [N+2, 2]
    
    # 4. 分离 patch 和 text embeddings
    patch_embeddings = embeddings_2d[:-2, :]  # [N, 2]
    text_embeddings = embeddings_2d[-2:, :]   # [2, 2]
    
    # 5. 绘图
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=150)
    
    # 绘制 normal patches (label=0)
    normal_mask = (patch_labels == 0)
    ax.scatter(
        patch_embeddings[normal_mask, 0],
        patch_embeddings[normal_mask, 1],
        c='green', marker='s', s=30, alpha=0.6,
        label=f'Normal patches ({normal_mask.sum()})'
    )
    
    # 绘制 lesion patches (label=1)
    lesion_mask = (patch_labels == 1)
    ax.scatter(
        patch_embeddings[lesion_mask, 0],
        patch_embeddings[lesion_mask, 1],
        c='red', marker='s', s=30, alpha=0.6,
        label=f'Lesion patches ({lesion_mask.sum()})'
    )
    
    # 绘制 text prompts (大圆点)
    ax.scatter(
        text_embeddings[0, 0], text_embeddings[0, 1],
        c='green', marker='o', s=200, edgecolors='black', linewidths=2,
        label='Normal prompt', zorder=10
    )
    ax.scatter(
        text_embeddings[1, 0], text_embeddings[1, 1],
        c='red', marker='o', s=200, edgecolors='black', linewidths=2,
        label='Abnormal prompt', zorder=10
    )
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=14)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=14)
    ax.set_title(f'Euclidean t-SNE: {args.dataset} (perplexity={args.perplexity})', fontsize=16)
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 6. 保存图片
    png_path = os.path.join(outdir, 'euclidean_tsne.png')
    pdf_path = os.path.join(outdir, 'euclidean_tsne.pdf')
    
    plt.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved Euclidean t-SNE plot:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description='Euclidean t-SNE Visualization')
    
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
    
    # t-SNE 参数
    parser.add_argument('--perplexity', type=int, default=30,
                       help='t-SNE perplexity')
    parser.add_argument('--n_iter', type=int, default=1000,
                       help='t-SNE iterations')
    
    # Hyperbolic 参数（虽然本脚本用欧氏特征，但需要加载模型时保持一致）
    parser.add_argument('--use_hyperbolic', action='store_true',
                       help='Whether the checkpoint uses hyperbolic adapters')
    parser.add_argument('--hyperbolic_c', type=float, default=0.1,
                       help='Curvature of Poincare ball')
    parser.add_argument('--scale_normal', type=float, default=0.1,
                       help='Radius scale for normal text')
    parser.add_argument('--scale_abnormal', type=float, default=0.8,
                       help='Radius scale for abnormal text')
    
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
    
    # 4. 提取 patch 特征
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
    
    # 5. 编码文本 prompts (使用欧氏特征)
    print("\n=== Encoding Text Prompts ===")
    text_features_euc, _, _ = encode_text_prompts(clip_model, args, device)
    
    # 6. 绘制 t-SNE 图
    plot_tsne_euclidean(patch_features, patch_labels, text_features_euc, args, args.outdir)
    
    # 7. 保存 metadata
    print("\n=== Saving Metadata ===")
    save_metadata(args, args.outdir, dataset_info, sampling_stats, sampled_indices, sampled_data)
    
    print("\n=== Done! ===")


if __name__ == '__main__':
    main()
