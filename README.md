# PMT: Plain Mask Transformer for Image and Video Segmentation with Frozen Vision Encoders
**CVPR 2026 Workshop** · [📄 Paper](https://arxiv.org/abs/2603.25398)

**[Niccolò Cavagnero](https://scholar.google.com/citations?user=Pr4XHRAAAAAJ), [Narges Norouzi](https://scholar.google.com/citations?user=q7sm490AAAAJ), [Gijs Dubbelman](https://scholar.google.nl/citations?user=wy57br8AAAAJ), [Daan de Geus](https://ddegeus.github.io)**

Eindhoven University of Technology

## Overview

We present the **Plain Mask Transformer (PMT)**, a fast Transformer-based segmentation model that operates on top of **frozen** Vision Foundation Model (VFM) features.

Encoder-only models like [EoMT](https://github.com/tue-mps/eomt) and [VidEoMT](https://github.com/tue-mps/videomt) achieve competitive accuracy with low latency but require **finetuning the full encoder**, preventing the VFM from being reused for other downstream tasks. 

PMT addresses this by introducing the **Plain Mask Decoder (PMD)**: a lightweight Transformer decoder that mimics the last encoder layers of EoMT, processing queries and frozen patch tokens jointly — without touching the encoder weights.

The result: a model that **keeps the encoder frozen and shareable** across tasks while matching the accuracy and speed of task-specific alternatives.

## Repository Structure

The codebase is organized by task domain. Image segmentation code is available now; video segmentation will be added in a future release.

```
pmt/
├── requirements.txt          # shared dependencies
├── image/                    # image segmentation
├── video/                    # video segmentation (coming soon)
├── model_zoo/                # pre-trained weight catalogues
│   ├── image/                # image model weights (DINOv3)
│   └── video/                # video model weights (coming soon)
└── docs/                     # project page
```

## Installation

If you don't have Conda installed, install Miniconda and restart your shell:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Then create the environment, activate it, and install the dependencies:

```bash
conda create -n pmt python==3.13.2
conda activate pmt
python3 -m pip install -r requirements.txt
```

[Weights & Biases](https://wandb.ai/) (wandb) is used for experiment logging and visualization. To enable wandb, log in to your account:

```bash
wandb login
```

## Data Preparation

- **Image datasets** (COCO, ADE20K, Cityscapes): follow the instructions in the [EoMT repository](https://github.com/tue-mps/eomt).
- **Video datasets** (YouTube-VIS, VIPSeg, VSPW): follow the instructions in the [VidEoMT repository](https://github.com/tue-mps/videomt).

## Image Segmentation

### Training

To train PMT from scratch, run:

```bash
python3 image/main.py fit \
  -c image/configs/coco/panoptic/pmt_l_640.yaml \
  --trainer.devices 4 \
  --data.batch_size 4 \
  --data.path /path/to/dataset
```

This trains `PMT-L` with a 640×640 input on COCO panoptic segmentation using 4 GPUs, for a total batch size of 16.

✅ Make sure the total batch size is `devices × batch_size = 16`  
🔧 Replace `/path/to/dataset` with the directory containing the dataset zip files.

> This configuration takes ~6 hours on 4×NVIDIA H100 GPUs, each using ~26GB VRAM.

To fine-tune a pre-trained PMT model, add:

```bash
  --model.ckpt_path /path/to/pytorch_model.bin \
  --model.load_ckpt_class_head False
```

🔧 Replace `/path/to/pytorch_model.bin` with the path to the checkpoint to fine-tune.  
> `--model.load_ckpt_class_head False` skips loading the classification head when fine-tuning on a dataset with different classes.

### Evaluating

To evaluate a pre-trained PMT model, run:

```bash
python3 image/main.py validate \
  -c image/configs/coco/panoptic/pmt_l_640.yaml \
  --model.network.masked_attn_enabled False \
  --trainer.devices 4 \
  --data.batch_size 4 \
  --data.path /path/to/dataset \
  --model.ckpt_path /path/to/pytorch_model.bin
```

🔧 Replace `/path/to/dataset` with the directory containing the dataset zip files.  
🔧 Replace `/path/to/pytorch_model.bin` with the path to the checkpoint to evaluate.

## Video Segmentation

Video segmentation code will be added in a future release. The `video/` directory is reserved for this purpose.

## Model Zoo

We provide pre-trained weights for PMT models with DINOv3 encoders.

- **[Image Models](model_zoo/image/dinov3.md)** - Image segmentation with DINOv3 encoder.
- **Video Models** - Coming soon.

## Citation

If you find this work useful in your research, please cite it using the BibTeX entry below:

```BibTeX
@inproceedings{cavagnero2026pmt,
  author    = {Cavagnero, Niccol\`{o} and Norouzi, Narges and Dubbelman, Gijs and {de Geus}, Daan},
  title     = {{PMT: Plain Mask Transformer for Image and Video Segmentation with Frozen Vision Encoders}},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year      = {2026},
}
```

## Acknowledgements

This project builds upon code from the following libraries and repositories:

- [EoMT](https://github.com/tue-mps/eomt) (MIT License)
- [VidEoMT](https://github.com/tue-mps/videomt) (MIT License)
- [Hugging Face Transformers](https://github.com/huggingface/transformers) (Apache-2.0 License)  
- [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models) (Apache-2.0 License)  
- [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) (Apache-2.0 License)  
- [TorchMetrics](https://github.com/Lightning-AI/torchmetrics) (Apache-2.0 License)  
- [Mask2Former](https://github.com/facebookresearch/Mask2Former) (Apache-2.0 License)
- [Detectron2](https://github.com/facebookresearch/detectron2) (Apache-2.0 License)
