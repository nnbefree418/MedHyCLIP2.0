# MedHyCLIP (Hyper-MVFA)

PyTorch implementation of **Hyper-MVFA**, built on top of the official [MVFA](https://arxiv.org/abs/2403.12570) codebase (CVPR 2024 Highlight). This repository extends MVFA with hyperbolic adapters and hyperbolic visual-language matching for medical anomaly detection.

**Code repository:** https://github.com/nnbefree418/MedHyCLIP2.0

---

## Overview

Hyper-MVFA keeps the lightweight multi-level CLIP adapter design from MVFA, and adds:

- Hyperbolic residual adapters in the Poincare ball
- Hyperbolic text embedding adjustment
- Hyperbolic distance-based anomaly scoring
- Unified zero-shot / few-shot train & test scripts with `--use_hyperbolic`

Supported datasets: **Brain**, **Liver**, **Retina_RESC**, **Retina_OCT2017**, **Chest**, **Histopathology**.

---

## Installation

```bash
pip install -r requirements.txt
```

Main dependencies: `torch`, `torchvision`, `geoopt`, `kornia`, `scikit-learn`, `opencv-python`, `ftfy`, `regex`.

---

## Data Preparation

Follow the original MVFA paper / [BMAD](https://github.com/DorisBao/BMAD) instructions, or download the pre-processed benchmarks linked in the original MVFA README. Place datasets under `data/`:

```
data/
├── Brain_AD/
├── Liver_AD/
├── Retina_RESC_AD/
├── Retina_OCT2017_AD/
├── Chest_AD/
└── Histopathology_AD/
```

Few-shot sample indices are provided under `dataset/fewshot_seed/`.

---

## Pretrained Weights

### 1. CLIP backbone (required)

Download OpenAI ViT-L/14@336 and place it at `CLIP/ckpt/ViT-L-14-336px.pt`:

https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt

### 2. Hyper-MVFA checkpoints (this project)

Download the two zip files from the latest GitHub Release (tag `v0.3.0`):

| File | Contents |
|------|----------|
| `zero-shot-hyper.zip` | 5 datasets: Brain, Chest, Histopathology, Liver, Retina_RESC |
| `few-shot-hyper.zip` | 6 datasets: Brain, Chest, Histopathology, Liver, Retina_OCT2017, Retina_RESC |

Unzip at the project root (creates `ckpt/zero-shot-hyper/` and `ckpt/few-shot-hyper/`):

```bash
unzip zero-shot-hyper.zip
unzip few-shot-hyper.zip
```

Expected layout after extraction:

```
ckpt/
├── zero-shot-hyper/
│   ├── Brain.pth
│   ├── Chest.pth
│   ├── Histopathology.pth
│   ├── Liver.pth
│   └── Retina_RESC.pth
└── few-shot-hyper/
    ├── Brain.pth
    ├── Chest.pth
    ├── Histopathology.pth
    ├── Liver.pth
    ├── Retina_OCT2017.pth
    └── Retina_RESC.pth
```

> Original MVFA (Euclidean) weights remain available from the [MVFA Google Drive links](https://arxiv.org/abs/2403.12570).

---

## Quick Start

### Zero-shot test (Hyper-MVFA)

```bash
python test_zero.py --obj Brain --use_hyperbolic
python test_zero.py --obj Liver --use_hyperbolic
python test_zero.py --obj Retina_RESC --use_hyperbolic
```

### Few-shot test (Hyper-MVFA, k=4)

```bash
python test_few.py --obj Brain --shot 4 --use_hyperbolic
python test_few.py --obj Liver --shot 4 --use_hyperbolic
```

### Training

```bash
# Zero-shot
python train_zero.py --obj Brain --use_hyperbolic --epoch 50 --patience 10

# Few-shot (k=4)
python train_few.py --obj Brain --shot 4 --use_hyperbolic --epoch 50 --patience 10
```

Checkpoints are saved to `ckpt/zero-shot-hyper/` or `ckpt/few-shot-hyper/` by default.

Multi-GPU DDP training is supported via `torchrun`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_zero.py \
  --obj Brain --use_hyperbolic --epoch 50 --patience 10
```

---

## Project Structure

```
MedHyCLIP/
├── train_zero.py          # Zero-shot training
├── test_zero.py           # Zero-shot evaluation
├── train_few.py           # Few-shot training
├── test_few.py            # Few-shot evaluation
├── loss.py                # Focal / Dice losses
├── prompt.py              # Dataset prompt names
├── utils.py               # Text encoding & augmentation
├── requirements.txt
├── CLIP/                  # CLIP model + adapters
│   ├── adapter.py         # Euclidean adapters
│   ├── hyperbolic_adapter.py  # Hyperbolic adapters
│   └── ...
├── dataset/
│   ├── medical_zero.py
│   ├── medical_few.py
│   └── fewshot_seed/
└── ckpt/                  # Downloaded / trained weights (not in git)
```

---

## Citation

If you use the original MVFA codebase, please cite:

```bibtex
@inproceedings{huang2024adapting,
  title={Adapting Visual-Language Models for Generalizable Anomaly Detection in Medical Images},
  author={Huang, Chaoqin and Jiang, Aofan and Feng, Jinghao and Zhang, Ya and Wang, Xinchao and Wang, Yanfeng},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```

If you use Hyper-MVFA, please also cite our work (update with your paper details):

```bibtex
@article{medhyclip2025,
  title={MedHyCLIP: Hyperbolic Visual-Language Adaptation for Medical Anomaly Detection},
  author={...},
  year={2025}
}
```

---

## License

This project is based on the official MVFA implementation (MIT License, Copyright (c) 2024 MediaBrain-SJTU). See [LICENSE](LICENSE) for details.

Hyper-MVFA extensions are released under the same MIT License. When using or distributing this code, please retain the original copyright notice and cite both MVFA and this work.

---

## Acknowledgements

- [MVFA](https://arxiv.org/abs/2403.12570) (CVPR 2024 Highlight)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [April-GAN](https://github.com/ByChelsea/VAND-APRIL-GAN)
- [geoopt](https://github.com/geoopt/geoopt) for hyperbolic geometry
