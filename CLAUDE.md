# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **eTraM (Event-based Traffic Monitoring Dataset)** repository, which contains implementations for event-based traffic monitoring using deep learning models. The project includes two main components:

1. **RVT (Recurrent Vision Transformers)** - Modified version for event-based object detection
2. **Ultralytics YOLO** - Modified version for event-based data

## Key Commands

### RVT Component (rvt_eTram/)

**Environment Setup:**
```bash
# Create environment using conda/mamba
conda env create -f environment.yaml
conda activate rvt
```

**Data Preprocessing:**
```bash
# Preprocess eTraM dataset to required format
python scripts/genx/preprocess_dataset.py <DATA_IN_PATH> <DATA_OUT_PATH> \
  conf_preprocess/representation/stacked_hist.yaml \
  conf_preprocess/extraction/const_duration.yaml \
  conf_preprocess/filter_gen4.yaml -ds gen4 -np <N_PROCESSES>
```

**Training:**
```bash
# Train RVT model
python train.py model=rnndet dataset=gen4 dataset.path=<DATA_DIR> \
  wandb.project_name=<WANDB_NAME> wandb.group_name=<WAND_GRP> \
  +experiment/gen4="default.yaml" hardware.gpus=0 batch_size.train=6 \
  batch_size.eval=2 hardware.num_workers.train=4 hardware.num_workers.eval=3 \
  training.max_epochs=20 dataset.train.sampling=stream +model.head.num_classes=3
```

**Evaluation:**
```bash
# Evaluate model
python validation.py dataset=gen4 dataset.path=${DATA_DIR} checkpoint=${CKPT_PATH} \
  use_test_set=${USE_TEST} hardware.gpus=${GPU_ID} +experiment/gen4="${MDL_CFG}.yaml" \
  batch_size.eval=8 model.postprocess.confidence_threshold=0.001
```

### Ultralytics Component (ultralytics_eTram/)

**Environment Setup:**
```bash
# Install requirements
pip install -r requirements.txt
```

**Training/Inference:**
```bash
# Run YOLO training/inference
cd yolo_eTram
python main.py
```

## Architecture Overview

### RVT Architecture (rvt_eTram/)

- **Configuration System**: Uses Hydra for configuration management with YAML files in `config/`
- **Main Components**:
  - `train.py`: Main training script with PyTorch Lightning
  - `modules/detection.py`: Core detection module with RNN states management
  - `models/detection/`: Model architectures including YOLOX detector and recurrent backbones
  - `data/`: Data loading and preprocessing utilities
  - `callbacks/`: Custom callbacks for visualization and monitoring

- **Key Features**:
  - Supports streaming and random sampling modes
  - Uses W&B for experiment tracking
  - Implements confusion matrix evaluation
  - Multi-GPU training support with DDP strategy

### Model Configuration

The project uses hierarchical YAML configuration:
- `config/model/`: Model architectures (base, small, tiny variants)
- `config/dataset/`: Dataset configurations (gen1, gen4, etrap)
- `config/experiment/`: Experiment-specific configs combining model and dataset settings

### Data Pipeline

- **Event Representation**: Uses stacked histogram representation
- **Preprocessing**: Constant duration extraction with filtering
- **Classes**: 8 traffic participant classes (vehicles, pedestrians, micro-mobility)
- **Evaluation**: Uses Prophesee evaluation metrics and confusion matrix

## Development Guidelines

### Working with Configurations

- Model configurations are in `config/model/`
- Dataset paths and parameters are in `config/dataset/`
- Experiment configs combine model and dataset settings
- Use `+experiment/gen4="config_name.yaml"` to load experiment configs

### Training Workflow

1. Preprocess dataset using `preprocess_dataset.py`
2. Configure model, dataset, and experiment parameters
3. Run training with appropriate GPU and batch size settings
4. Monitor training through W&B (if enabled)
5. Evaluate using `validation.py`

### Model Variants

- **RVT-base**: Full model with maximum performance
- **RVT-small**: Reduced model size for faster training
- **RVT-tiny**: Minimal model for resource-constrained environments

### Data Handling

- Uses PyTorch Lightning data modules
- Supports both streaming and random sampling
- Implements custom data loaders for event-based data
- Includes data augmentation and padding utilities

## RVT Model Architecture Summary

### High-Level Architecture Overview

```
Event Data → Recurrent Backbone → Feature Pyramid → Detection Head → Predictions
    ↓              ↓                    ↓              ↓            ↓
Preprocessing   MaxViT+LSTM         YOLO PAFPN    YOLOX Head    Post-process
```

### File Structure & Component Mapping

```
rvt_eTram/
├── models/detection/
│   ├── yolox_extension/models/
│   │   ├── detector.py           # Main YoloXDetector class
│   │   ├── build.py             # Component builders
│   │   └── yolo_pafpn.py        # Feature Pyramid Network
│   ├── recurrent_backbone/
│   │   ├── base.py              # BaseDetector interface
│   │   └── maxvit_rnn.py        # RNN Backbone (Core)
│   └── yolox/models/
│       ├── yolo_head.py         # Detection head
│       ├── losses.py            # Loss functions
│       └── network_blocks.py    # Basic building blocks
├── modules/
│   └── detection.py             # PyTorch Lightning module
├── config/
│   ├── model/                   # Model configurations
│   └── experiment/              # Experiment configs
└── train.py                     # Main training script
```

### Visual Architecture Diagram

```
INPUT: Event Representation (N, C, H, W)
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RECURRENT BACKBONE                           │
│              maxvit_rnn.py (RNNDetector)                       │
├─────────────────────────────────────────────────────────────────┤
│  Stage 1: Patch=4  │ Stage 2: ↓2    │ Stage 3: ↓2    │ Stage 4: ↓2   │
│  Stride: 4         │ Stride: 8      │ Stride: 16     │ Stride: 32    │
│  Dim: 64           │ Dim: 128       │ Dim: 256       │ Dim: 512      │
│  ┌─────────────┐   │ ┌─────────────┐ │ ┌─────────────┐ │ ┌─────────────┐ │
│  │ Downsample  │   │ │ Downsample  │ │ │ Downsample  │ │ │ Downsample  │ │
│  │     ↓       │   │ │     ↓       │ │ │     ↓       │ │ │     ↓       │ │
│  │ MaxViT      │   │ │ MaxViT      │ │ │ MaxViT      │ │ │ MaxViT      │ │
│  │ Attention   │   │ │ Attention   │ │ │ Attention   │ │ │ Attention   │ │
│  │ (Window+Grid)│  │ │ (Window+Grid)│ │ │ (Window+Grid)│ │ │ (Window+Grid)│ │
│  │     ↓       │   │ │     ↓       │ │ │     ↓       │ │ │     ↓       │ │
│  │ ConvLSTM    │   │ │ ConvLSTM    │ │ │ ConvLSTM    │ │ │ ConvLSTM    │ │
│  │ (Recurrent) │   │ │ (Recurrent) │ │ │ (Recurrent) │ │ │ (Recurrent) │ │
│  └─────────────┘   │ └─────────────┘ │ └─────────────┘ │ └─────────────┘ │
│                    │                │                │                │
│     P1: H/4×W/4    │   P2: H/8×W/8  │  P3: H/16×W/16 │  P4: H/32×W/32 │
│     (not used)     │   (for FPN)    │   (for FPN)    │   (for FPN)    │
└─────────────────────────────────────────────────────────────────┘
          │                    │                │                │
          ▼                    ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                 FEATURE PYRAMID NETWORK                        │
│                 yolo_pafpn.py (YOLOPAFPN)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  P4 (512) ──1×1──→ 256 ──↑×2──┐                               │
│                              │                                │
│  P3 (256) ──────────────────→ ⊕ ──CSP──→ 256 ──1×1──→ 128 ──↑×2──┐ │
│                                                               │ │
│  P2 (128) ────────────────────────────────────────────────→ ⊕ ──CSP──→ N3 (128) │
│                                                               │         │
│                                              ┌────3×3↓2─────┘         │
│                                              │                         │
│  N4 (256) ←──CSP←──⊕←──────────────────────┘                         │
│              │                                                       │
│              └─────3×3↓2────→ ⊕ ────CSP───→ N5 (512)                 │
│                               │                                       │
│                               └───────────← P4                       │
│                                                                       │
│  OUTPUT: (N3: 128@H/8, N4: 256@H/16, N5: 512@H/32)                  │
└─────────────────────────────────────────────────────────────────┘
          │                    │                │
          ▼                    ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DETECTION HEAD                              │
│                 yolo_head.py (YOLOXHead)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  For each scale (stride 8, 16, 32):                           │
│                                                                 │
│  Feature ──1×1──→ Hidden (256) ──┬──→ Cls Conv ──→ Cls Pred    │
│                                  │    (3×3×2)      (1×1)       │
│                                  │                             │
│                                  └──→ Reg Conv ──┬──→ Reg Pred │
│                                       (3×3×2)    │    (1×1×4)  │
│                                                  │             │
│                                                  └──→ Obj Pred │
│                                                       (1×1×1)  │
│                                                                 │
│  Output: [Reg(4) + Obj(1) + Cls(num_classes)] per anchor      │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   POST-PROCESSING                              │
│                boxes.py (postprocess)                          │
├─────────────────────────────────────────────────────────────────┤
│  1. Confidence Thresholding                                    │
│  2. NMS (Non-Maximum Suppression)                              │
│  3. Coordinate Decoding                                        │
│  4. Final Detections                                           │
└─────────────────────────────────────────────────────────────────┘
```

### Component Details

#### 1. Main Detector (`detector.py:18`)
```python
class YoloXDetector(th.nn.Module):
    def __init__(self, model_cfg):
        self.backbone = build_recurrent_backbone(backbone_cfg)
        self.fpn = build_yolox_fpn(fpn_cfg, in_channels)
        self.yolox_head = build_yolox_head(head_cfg, in_channels, strides)
```

#### 2. Recurrent Backbone (`maxvit_rnn.py:23`)
```python
class RNNDetector(BaseDetector):
    # 4-stage hierarchical backbone
    # Each stage: Downsample → MaxViT Attention → ConvLSTM
    # Strides: [4, 8, 16, 32]
    # Dims: [64, 128, 256, 512] (configurable)
```

#### 3. Feature Pyramid (`yolo_pafpn.py:18`)
```python
class YOLOPAFPN(nn.Module):
    # Top-down + Bottom-up feature fusion
    # Input: P2, P3, P4 from backbone
    # Output: N3, N4, N5 for detection
```

#### 4. Detection Head (`yolo_head.py:21`)
```python
class YOLOXHead(nn.Module):
    # Per-scale processing: 3 scales (8, 16, 32)
    # Outputs: Classification + Regression + Objectness
    # Uses anchor-free detection with center-based assignment
```

### Data Flow Summary

1. **Input**: Event representation (stacked histograms) - `(B, C, H, W)`
2. **Backbone**: 4-stage processing with recurrent memory - `{P1, P2, P3, P4}`
3. **FPN**: Multi-scale feature fusion - `{N3, N4, N5}`
4. **Detection**: Per-scale predictions - `[reg, obj, cls]`
5. **Output**: Bounding boxes with confidence scores

### Key Configuration Files

- **Model Config**: `config/model/maxvit_yolox/default.yaml`
- **Training Config**: `config/experiment/gen4/default.yaml`
- **Dataset Config**: `config/dataset/gen4.yaml`

### Memory and Recurrence

The key innovation of RVT is the **recurrent processing** at each backbone stage:

```python
# From maxvit_rnn.py:180
h_c_tuple = self.lstm(x, h_and_c_previous)  # ConvLSTM with memory
```

Each stage maintains **hidden states** across time steps, enabling temporal modeling of event sequences. This is managed by:

- **RNNStates** (`modules/detection.py:22`): Manages LSTM states per training mode
- **LstmStates** type: List of hidden/cell state tuples for each backbone stage

### Current Limitations for Small Objects

1. **Missing P1 features** (stride 4) - highest resolution discarded
2. **FPN starts at P2** (stride 8) - limited fine detail
3. **Anchor assignment** optimized for medium/large objects
4. **Loss weighting** doesn't prioritize small object scales