#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
边界假阳性（Boundary False Positive）错误模式热力图可视化

选取满足以下之一的样本并出图：
  - GT 无异常标注（正常图），但模型在组织边缘/强对比处大面积高亮
  - GT 异常区域很小，但模型响应集中在正常组织边界而非 GT 区域

输出：原图 | GT 叠加 | 热力图叠加 | 四联对比图（含结构边缘参考）
"""
import os
import sys
import argparse
import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.ndimage import gaussian_filter, sobel

# 项目根目录
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, ROOT)

from dataset.medical_zero import MedTestDataset, CLASS_INDEX, CLASS_NAMES
from CLIP.clip import create_model
from CLIP.adapter import CLIP_Inplanted
from utils import encode_text_with_prompt_ensemble, encode_text_with_hyperbolic_adjustment
from prompt import REAL_NAME


SEG_DATASETS = [name for name in CLASS_NAMES if CLASS_INDEX[name] > 0]


def setup_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_image_tensor(image_path, img_size):
    """从任意路径加载单张图（如 valid/good），返回 [3,H,W] 与原始 RGB uint8。"""
    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), Image.BICUBIC),
        transforms.ToTensor(),
    ])
    pil_img = Image.open(image_path).convert('RGB')
    rgb_uint8 = np.array(pil_img.resize((img_size, img_size), Image.BICUBIC), dtype=np.uint8)
    return transform(pil_img), rgb_uint8


def load_model_and_text(args, device):
    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained='openai',
        require_pretrained=True,
    )
    clip_model.eval()

    model = CLIP_Inplanted(
        clip_model=clip_model,
        features=args.features_list,
        use_hyperbolic=args.use_hyperbolic,
        hyperbolic_c=args.hyperbolic_c,
    ).to(device)
    model.eval()

    if args.tag is None:
        ckpt_name = f"{args.dataset}.pth"
    else:
        ckpt_name = f"{args.dataset}_{args.tag}.pth"
    ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.seg_adapters.load_state_dict(checkpoint["seg_adapters"])
    model.det_adapters.load_state_dict(checkpoint["det_adapters"])
    print(f"Loaded checkpoint: {ckpt_path}")

    with torch.no_grad(), torch.cuda.amp.autocast():
        if args.use_hyperbolic:
            text_features, ball = encode_text_with_hyperbolic_adjustment(
                clip_model,
                REAL_NAME[args.dataset],
                device,
                use_hyperbolic=True,
                c=args.hyperbolic_c,
                scale_normal=args.scale_normal,
                scale_abnormal=args.scale_abnormal,
            )
        else:
            text_features = encode_text_with_prompt_ensemble(
                clip_model, REAL_NAME[args.dataset], device
            )
            ball = None

    return model, text_features, ball


@torch.no_grad()
def compute_anomaly_map(model, image, text_features, ball, args, device):
    """与 test_zero.py 一致的 zero-shot 分割热力图（多层平均）。"""
    if image.dim() == 3:
        image = image.unsqueeze(0)
    elif image.dim() == 5:
        image = image.squeeze(0)
    image = image.to(device)
    with torch.cuda.amp.autocast():
        _, seg_patch_tokens, _ = model(image)
        seg_patch_tokens = [p[0, 1:, :] for p in seg_patch_tokens]

        anomaly_maps = []
        for layer in range(len(seg_patch_tokens)):
            if args.use_hyperbolic:
                L, _ = seg_patch_tokens[layer].shape
                H = int(np.sqrt(L))
                text_h = text_features.T
                dist_normal = ball.dist(seg_patch_tokens[layer], text_h[0])
                dist_abnormal = ball.dist(seg_patch_tokens[layer], text_h[1])
                logits_normal = -args.temperature * dist_normal
                logits_abnormal = -args.temperature * dist_abnormal
                anomaly_map = torch.stack(
                    [logits_normal, logits_abnormal], dim=-1
                ).unsqueeze(0)
                anomaly_map = F.interpolate(
                    anomaly_map.permute(0, 2, 1).view(1, 2, H, H),
                    size=args.img_size,
                    mode='bilinear',
                    align_corners=True,
                )
                anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
            else:
                tokens = seg_patch_tokens[layer]
                tokens = tokens / tokens.norm(dim=-1, keepdim=True)
                anomaly_map = (100.0 * tokens @ text_features).unsqueeze(0)
                L = anomaly_map.shape[1]
                H = int(np.sqrt(L))
                anomaly_map = F.interpolate(
                    anomaly_map.permute(0, 2, 1).view(1, 2, H, H),
                    size=args.img_size,
                    mode='bilinear',
                    align_corners=True,
                )
                anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
            anomaly_maps.append(anomaly_map[0].float().cpu().numpy())

    score_map = np.mean(anomaly_maps, axis=0)
    smin, smax = score_map.min(), score_map.max()
    score_map = (score_map - smin) / (smax - smin + 1e-8)
    return score_map


def structure_edge_map(rgb_uint8, sigma=1.0, percentile=82.0):
    """从原图梯度提取强对比/器官轮廓区域（用于量化边界假阳性）。"""
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray = gaussian_filter(gray, sigma=sigma)
    gx = sobel(gray, axis=1)
    gy = sobel(gray, axis=0)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    thr = np.percentile(grad, percentile)
    edges = (grad >= thr).astype(np.float32)
    return edges, grad


def boundary_false_positive_score(score_map, gt_mask, edge_map):
    """
    越高表示：GT 很小或为空，且高响应集中在 GT 外、尤其结构边缘处。
    """
    gt = (gt_mask > 0.5).astype(np.float32)
    gt_area = gt.mean()
    outside = 1.0 - gt

    outside_mean = (score_map * outside).sum() / (outside.sum() + 1e-8)
    inside_mean = (score_map * gt).sum() / (gt.sum() + 1e-8)
    misloc_ratio = outside_mean / (inside_mean + 1e-6)

    edge_outside = score_map * edge_map * outside
    edge_outside_mean = edge_outside.sum() / ((edge_map * outside).sum() + 1e-8)

    # 高响应像素中落在 GT 外的比例
    hot = score_map >= np.percentile(score_map, 90)
    hot_outside_frac = (hot * outside).sum() / (hot.sum() + 1e-8)

    # 正常图（无 GT）加权更高；小 GT 病灶也加权
    tiny_gt_boost = 1.0 + max(0.0, 0.05 - gt_area) * 20.0
    normal_boost = 2.5 if gt_area < 1e-6 else 1.0

    score = (
        misloc_ratio
        * edge_outside_mean
        * hot_outside_frac
        * tiny_gt_boost
        * normal_boost
    )
    metrics = {
        'gt_area_ratio': float(gt_area),
        'misloc_ratio': float(misloc_ratio),
        'edge_outside_mean': float(edge_outside_mean),
        'hot_outside_frac': float(hot_outside_frac),
        'boundary_fp_score': float(score),
    }
    return score, metrics


def tensor_to_rgb_uint8(img_tensor):
    """[3,H,W] float [0,1] -> uint8 RGB"""
    x = img_tensor.cpu().numpy().transpose(1, 2, 0)
    x = np.clip(x, 0, 1)
    if x.max() <= 1.0:
        x = (x * 255).astype(np.uint8)
    else:
        x = x.astype(np.uint8)
    return x


def apply_jet_overlay(rgb_uint8, score_map, alpha=0.55):
    sm = (score_map * 255).astype(np.uint8)
    heat = cv2.applyColorMap(sm, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    vis = cv2.addWeighted(heat, alpha, rgb_uint8, 1 - alpha, 0)
    return vis, heat


def draw_gt_overlay(rgb_uint8, gt_mask, color=(0, 0, 255)):
    vis = rgb_uint8.copy()
    gt = (gt_mask > 0.5)
    if gt.any():
        vis[gt] = (
            0.45 * vis[gt].astype(np.float32) + 0.55 * np.array(color, dtype=np.float32)
        ).astype(np.uint8)
    return vis


def _gt_area_ratio_from_path(mask_path, resize):
    """不跑模型，仅从 GT mask 文件估计面积占比。"""
    if mask_path is None:
        return 0.0
    from PIL import Image
    m = Image.open(mask_path).convert('L').resize((resize, resize), Image.NEAREST)
    arr = np.array(m, dtype=np.float32) / 255.0
    return float((arr > 0.5).mean())


def build_candidate_indices(dataset, args):
    """优先：正常图（无 GT）；其次：GT 面积很小的异常图。"""
    indices = []
    for idx, (y, mask_path) in enumerate(zip(dataset.y, dataset.mask)):
        if y == 0:
            indices.append(idx)
        elif mask_path is not None:
            area = _gt_area_ratio_from_path(mask_path, args.img_size)
            if area < args.max_gt_area_ratio:
                indices.append(idx)
    if args.max_scan > 0 and len(indices) > args.max_scan:
        # 保留全部正常图，其余随机子采样异常小病灶
        normal_idx = [i for i in indices if dataset.y[i] == 0]
        ab_idx = [i for i in indices if dataset.y[i] == 1]
        n_ab = max(0, args.max_scan - len(normal_idx))
        if len(ab_idx) > n_ab:
            rng = np.random.RandomState(args.seed)
            ab_idx = list(rng.choice(ab_idx, size=n_ab, replace=False))
        indices = normal_idx + ab_idx
    return indices


def scan_dataset(args, model, text_features, ball, device):
    dataset = MedTestDataset(args.data_path, args.dataset, args.img_size)
    candidate_indices = build_candidate_indices(dataset, args)
    print(f'  Candidate images to score: {len(candidate_indices)}')

    candidates = []
    for idx in tqdm(candidate_indices, desc=f'Scan {args.dataset}'):
        image, y, mask = dataset[idx]
        gt_np = mask.squeeze().numpy()
        gt_np = (gt_np > 0.5).astype(np.float32)

        score_map = compute_anomaly_map(model, image, text_features, ball, args, device)
        rgb = tensor_to_rgb_uint8(image)
        edge_map, _ = structure_edge_map(
            rgb, sigma=args.edge_sigma, percentile=args.edge_percentile
        )

        score, metrics = boundary_false_positive_score(score_map, gt_np, edge_map)
        metrics['index'] = idx
        metrics['label'] = int(y)
        metrics['path'] = dataset.x[idx]
        metrics['is_normal'] = metrics['gt_area_ratio'] < 1e-6
        candidates.append((score, metrics, score_map, gt_np, rgb, edge_map))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates


def save_visualization(
    rgb_uint8,
    gt_mask,
    score_map,
    edge_map,
    metrics,
    outdir,
    basename,
):
    os.makedirs(outdir, exist_ok=True)

    gt_vis = draw_gt_overlay(rgb_uint8, gt_mask)
    heat_vis, heat_raw = apply_jet_overlay(rgb_uint8, score_map, alpha=0.55)
    edge_vis = draw_gt_overlay(
        apply_jet_overlay(rgb_uint8, edge_map, alpha=0.35)[0],
        gt_mask,
        color=(0, 255, 0),
    )

    # 单独保存
    cv2.imwrite(
        os.path.join(outdir, f'{basename}_original.jpg'),
        cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        os.path.join(outdir, f'{basename}_gt.jpg'),
        cv2.cvtColor(gt_vis, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        os.path.join(outdir, f'{basename}_heatmap.jpg'),
        cv2.cvtColor(heat_vis, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        os.path.join(outdir, f'{basename}_heatmap_raw.jpg'),
        cv2.cvtColor(heat_raw, cv2.COLOR_RGB2BGR),
    )

    # 四联图（论文用）
    fig = plt.figure(figsize=(16, 4.2), dpi=150)
    gs = gridspec.GridSpec(1, 4, wspace=0.08)

    titles = [
        '(a) Input',
        '(b) GT (red)',
        '(c) Pred heatmap',
        '(d) Structure edges',
    ]
    panels = [rgb_uint8, gt_vis, heat_vis, edge_vis]

    for ax_idx, (title, img) in enumerate(zip(titles, panels)):
        ax = fig.add_subplot(gs[ax_idx])
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    mode = 'Normal (no GT lesion)' if metrics['is_normal'] else 'Small GT lesion'
    fig.suptitle(
        f'Boundary False Positive — {mode}\n'
        f'score={metrics["boundary_fp_score"]:.4f}  '
        f'GT area={100*metrics["gt_area_ratio"]:.2f}%  '
        f'hot outside GT={100*metrics["hot_outside_frac"]:.1f}%',
        fontsize=12,
        y=1.02,
    )
    panel_path = os.path.join(outdir, f'{basename}_panel.png')
    plt.savefig(panel_path, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)

    # 错误模式示意：GT 外高响应 vs GT 内
    fig2, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
    outside = 1.0 - (gt_mask > 0.5).astype(float)
    gt = (gt_mask > 0.5).astype(float)

    axes[0].imshow(rgb_uint8)
    axes[0].set_title('Input')
    axes[0].axis('off')

    im1 = axes[1].imshow(score_map * outside, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title('Heatmap × outside GT')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(score_map * edge_map * outside, cmap='hot', vmin=0, vmax=1)
    axes[2].set_title('Heatmap × edges × outside GT')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    fig2.suptitle('Mislocalization: response on normal boundaries, not GT', fontsize=11)
    diag_path = os.path.join(outdir, f'{basename}_mislocalization.png')
    plt.savefig(diag_path, bbox_inches='tight')
    plt.close(fig2)

    return panel_path, diag_path


def main():
    parser = argparse.ArgumentParser(
        description='Select and visualize boundary false-positive heatmaps'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        choices=SEG_DATASETS,
        help='Dataset with pixel masks. If omitted, scan all seg datasets and pick global best.',
    )
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt/zero-shot-hyper')
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='ViT-L-14-336')
    parser.add_argument('--img_size', type=int, default=240)
    parser.add_argument('--features_list', type=int, nargs='+', default=[6, 12, 18, 24])
    parser.add_argument('--use_hyperbolic', action='store_true', default=True)
    parser.add_argument('--no_hyperbolic', action='store_false', dest='use_hyperbolic')
    parser.add_argument('--hyperbolic_c', type=float, default=0.1)
    parser.add_argument('--scale_normal', type=float, default=0.1)
    parser.add_argument('--scale_abnormal', type=float, default=0.8)
    parser.add_argument('--temperature', type=float, default=20.0)
    parser.add_argument('--max_gt_area_ratio', type=float, default=0.05,
                        help='Max GT area ratio for abnormal samples (small lesion)')
    parser.add_argument('--edge_sigma', type=float, default=1.0)
    parser.add_argument('--edge_percentile', type=float, default=82.0)
    parser.add_argument('--index', type=int, default=None,
                        help='Force a specific dataset index (skip auto selection)')
    parser.add_argument('--image_path', type=str, default=None,
                        help='Direct path to one image (e.g. valid/good/00025_99.png); skips dataset scan')
    parser.add_argument('--outdir', type=str, default='./outputs/edge_false_positive_heatmap')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Save metadata for top-k candidates per dataset')
    parser.add_argument('--max_scan', type=int, default=0,
                        help='Max images to run model on (0=all candidates)')
    args = parser.parse_args()

    setup_seed(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    )
    print(f'Device: {device}')

    # ---------- 指定单张图片：直接出图 ----------
    if args.image_path:
        if args.dataset is None:
            args.dataset = 'Brain'
        image_path = os.path.abspath(args.image_path)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(image_path)

        print(f'\n=== Single image mode ===')
        print(f'  Image: {image_path}')
        print(f'  Dataset/checkpoint: {args.dataset}')

        model, text_features, ball = load_model_and_text(args, device)
        image_tensor, rgb = load_image_tensor(image_path, args.img_size)
        gt_np = np.zeros((args.img_size, args.img_size), dtype=np.float32)

        score_map = compute_anomaly_map(model, image_tensor, text_features, ball, args, device)
        edge_map, _ = structure_edge_map(
            rgb, sigma=args.edge_sigma, percentile=args.edge_percentile
        )
        _, metrics = boundary_false_positive_score(score_map, gt_np, edge_map)
        metrics['path'] = image_path
        metrics['label'] = 0
        metrics['index'] = -1
        metrics['is_normal'] = True

        stem = os.path.splitext(os.path.basename(image_path))[0]
        out_dir = os.path.join(args.outdir, args.dataset)
        save_visualization(rgb, gt_np, score_map, edge_map, metrics, out_dir, stem)

        summary = {
            'error_mode': 'boundary_false_positive',
            'image_path': image_path,
            'metrics': metrics,
            'output_dir': out_dir,
        }
        summary_path = os.path.join(out_dir, f'{stem}_summary.json')
        os.makedirs(out_dir, exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print('\n=== Done ===')
        print(json.dumps(metrics, indent=2))
        print(f'Output: {out_dir}')
        return

    datasets_to_scan = [args.dataset] if args.dataset else SEG_DATASETS
    global_best = None

    for ds_name in datasets_to_scan:
        args.dataset = ds_name
        print(f'\n========== {ds_name} ==========')
        model, text_features, ball = load_model_and_text(args, device)

        if args.index is not None:
            dataset = MedTestDataset(args.data_path, ds_name, args.img_size)
            image, y, mask = dataset[args.index]
            gt_np = (mask.squeeze().numpy() > 0.5).astype(np.float32)
            score_map = compute_anomaly_map(model, image, text_features, ball, args, device)
            rgb = tensor_to_rgb_uint8(image)
            edge_map, _ = structure_edge_map(rgb)
            _, metrics = boundary_false_positive_score(score_map, gt_np, edge_map)
            metrics['index'] = args.index
            metrics['path'] = dataset.x[args.index]
            metrics['label'] = int(y)
            metrics['is_normal'] = metrics['gt_area_ratio'] < 1e-6
            chosen = (0.0, metrics, score_map, gt_np, rgb, edge_map)
        else:
            candidates = scan_dataset(args, model, text_features, ball, device)
            if not candidates:
                print(f'  No candidates for {ds_name}')
                continue
            chosen = candidates[0]
            print(f'  Top candidate score={chosen[0]:.4f} idx={chosen[1]["index"]} path={chosen[1]["path"]}')

            # 记录 top-k
            top_meta = [c[1] for c in candidates[: args.top_k]]
            meta_path = os.path.join(args.outdir, ds_name, 'candidates_topk.json')
            os.makedirs(os.path.dirname(meta_path), exist_ok=True)
            with open(meta_path, 'w') as f:
                json.dump(top_meta, f, indent=2)

        score_val, metrics, score_map, gt_np, rgb, edge_map = chosen
        if global_best is None or metrics['boundary_fp_score'] > global_best[1]['boundary_fp_score']:
            global_best = (ds_name, metrics, score_map, gt_np, rgb, edge_map)

        if args.dataset and args.index is not None:
            # 单数据集 + 指定 index：直接出图
            basename = f'{ds_name}_idx{args.index:04d}'
            out_ds_dir = os.path.join(args.outdir, ds_name)
            save_visualization(rgb, gt_np, score_map, edge_map, metrics, out_ds_dir, basename)
            print(f'Saved to {out_ds_dir}')
            return

    if global_best is None:
        raise RuntimeError('No suitable sample found across datasets.')

    ds_name, metrics, score_map, gt_np, rgb, edge_map = global_best
    basename = f'{ds_name}_idx{metrics["index"]:04d}_boundary_fp'
    out_dir = os.path.join(args.outdir, ds_name)
    panel_path, diag_path = save_visualization(
        rgb, gt_np, score_map, edge_map, metrics, out_dir, basename
    )

    summary = {
        'error_mode': 'boundary_false_positive',
        'description': (
            'Model heatmap concentrates on normal tissue edges / strong contrast '
            'structures rather than GT lesion region.'
        ),
        'selected_dataset': ds_name,
        'metrics': metrics,
        'outputs': {
            'panel': panel_path,
            'mislocalization': diag_path,
        },
    }
    summary_path = os.path.join(args.outdir, 'selection_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n=== Selected sample ===')
    print(json.dumps(metrics, indent=2))
    print(f'Summary: {summary_path}')
    print(f'Figures: {out_dir}')


if __name__ == '__main__':
    main()
