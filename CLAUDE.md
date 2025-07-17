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

### ✅ PHASE 1 COMPLETED - P1 Feature Integration (2025-07-04)

**Successfully implemented 4-scale FPN for small object detection!**

**Modified Files:**
1. **`rvt_eTram/models/detection/yolox_extension/models/yolo_pafpn.py`**:
   - Extended YOLOPAFPN class to support both 3-scale and 4-scale configurations
   - Added adaptive layer creation based on `num_scales` parameter
   - Implemented 4-scale top-down pathway: P4→P3→P2→P1
   - Implemented 4-scale bottom-up pathway: N1→N2→N3→N4
   - Maintains backward compatibility with original 3-scale mode

2. **`rvt_eTram/config/model/maxvit_yolox/default.yaml`**:
   - Changed `in_stages: [2, 3, 4]` → `in_stages: [1, 2, 3, 4]`
   - Enables P1 features (stride 4) for small object detection

**Current Architecture Status:**
- ✅ **P1 features now utilized** (stride 4) - highest resolution features active
- ✅ **FPN supports 4 scales** (stride 4, 8, 16, 32) - enhanced fine detail
- ✅ **Detection head automatically supports 4 scales** - no changes needed
- 🔄 **Anchor assignment** still optimized for medium/large objects (Phase 2)
- 🔄 **Loss weighting** doesn't prioritize small object scales (Phase 2)

## Small Object Detection Enhancement Plan

### Strategy Overview

Event-based small object detection faces unique challenges:
- **Sparse Event Generation**: Small objects generate fewer events
- **Temporal Inconsistency**: Irregular movement patterns
- **Signal-to-Noise Ratio**: Difficulty distinguishing small objects from noise
- **Resolution Loss**: Current architecture discards high-resolution features

### ✅ Phase 1: High-Resolution Feature Integration (COMPLETED)

**Implementation Status: DONE ✅**

#### 1.1 Backbone Modification ✅
- **Status**: No changes needed - backbone already outputs all stages
- **Verification**: `maxvit_rnn.py:104` already returns `{1: P1, 2: P2, 3: P3, 4: P4}`

#### 1.2 FPN Extension ✅ 
- **File**: `rvt_eTram/models/detection/yolox_extension/models/yolo_pafpn.py`
- **Changes**:
  - Extended `__init__` to support 4-scale: `in_stages=[1,2,3,4], in_channels=[64,128,256,512]`
  - Added conditional layer creation for 4-scale vs 3-scale
  - Implemented P1 processing layers: `reduce_conv2`, `C3_p2`, `bu_conv3`, `C3_n2`
  - Updated `forward()` method for 4-scale pathway

#### 1.3 Detection Head ✅
- **Status**: No changes needed - YOLOXHead automatically supports variable scales
- **Verification**: Tested with 4 input scales, works correctly

**Testing Results:**
```
✓ 4-scale FPN initialization successful
✓ Forward pass successful - Output shapes:
  N1: torch.Size([2, 64, 80, 80]) (stride 4)
  N2: torch.Size([2, 128, 40, 40]) (stride 8)  
  N3: torch.Size([2, 256, 20, 20]) (stride 16)
  N4: torch.Size([2, 512, 10, 10]) (stride 32)
✓ Total detection scales: 4 (including stride 4 for small objects)
```

### Phase 2: Small Object Specialized Components (Medium Priority)

#### 2.1 Small Object Attention Module
```python
class SmallObjectAttention(nn.Module):
    def __init__(self, channels, scale_factor):
        self.motion_attention = MotionAwareAttention()
        self.spatial_attention = SpatialAttention()  
        self.scale_attention = ScaleAwareAttention(scale_factor)
```

#### 2.2 Temporal Feature Enhancement
```python
class TemporalFeatureEnhancer(nn.Module):
    def __init__(self):
        self.multi_temporal_fusion = MultiTemporalFusion()
        self.temporal_consistency = TemporalConsistency()
```

#### 2.3 Size-Aware Loss Function
```python
class SizeAwareLoss(nn.Module):
    def forward(self, pred, target, bbox_sizes):
        # Higher weight for smaller objects
        size_weights = torch.exp(-bbox_sizes / threshold)
        weighted_loss = base_loss * size_weights
```

### Phase 3: Training Strategy Optimization

#### 3.1 Adaptive Sampling
- Increase sampling ratio for small object samples
- Hard negative mining for difficult small object cases

#### 3.2 Multi-Scale Training  
- Various input resolutions during training
- Scale jittering for robustness improvement

### Implementation Priority

**High Priority (Immediate Implementation):**
1. P1 Feature Integration (Backbone + FPN + Head)
2. Size-Aware Loss Function
3. Training Strategy Improvements

**Medium Priority (Secondary Implementation):**
4. Small Object Attention Mechanisms
5. Temporal Enhancement Modules
6. Advanced Event Representations

**Low Priority (Experimental):**
7. Deformable Convolutions
8. Neural Architecture Search adaptations

### Expected Performance Improvements

- **Small Object AP**: +15-25% improvement expected
- **Overall mAP**: +5-10% improvement expected  
- **Temporal Consistency**: Significant improvement in tracking
- **False Positive Rate**: Reduction through better noise distinction

### File Modification Summary

**Core Files to Modify:**
- `rvt_eTram/models/detection/recurrent_backbone/maxvit_rnn.py` - Add P1 output
- `rvt_eTram/models/detection/yolox_extension/models/yolo_pafpn.py` - 4-scale FPN
- `rvt_eTram/models/detection/yolox/models/yolo_head.py` - 4-scale detection
- `rvt_eTram/models/detection/yolox/models/losses.py` - Size-aware loss
- `rvt_eTram/config/model/maxvit_yolox/default.yaml` - Configuration updates

**✅ Configuration Changes Applied:**
```yaml
# rvt_eTram/config/model/maxvit_yolox/default.yaml
fpn:
  in_stages: [1, 2, 3, 4]  # ✅ IMPLEMENTED - Include P1 for small objects
  # Note: in_channels automatically inferred from backbone: [64, 128, 256, 512]

# Detection head automatically inherits strides: [4, 8, 16, 32] ✅ WORKING
```

**🚀 Ready for Training:**
The implementation is complete and tested. You can now run training with enhanced small object detection:

```bash
# Train with P1 features enabled
python train.py model=rnndet dataset=gen4 dataset.path=<DATA_DIR> \
  +experiment/gen4="default.yaml" hardware.gpus=0 batch_size.train=6 \
  batch_size.eval=2 training.max_epochs=20
```

This systematic approach will significantly enhance small object detection performance while maintaining the RVT architecture's temporal modeling strengths.

## Small Object Detection Enhancement Experiments

### 🎯 실험 목표 및 설정

**데이터셋 (고정):**
- **메인 데이터셋**: `etram_cls8_sample`
- **경로**: `/home/oeoiewt/eTraM/rvt_eTram/data/etram_cls8_sample`
- **클래스 수**: 8개 (Car, Truck, Motorcycle, Bicycle, Pedestrian, Bus, Static, Other)
- **목표**: Small object detection 성능 향상 (클래스 2,3,4: Motorcycle, Bicycle, Pedestrian)

**필수 설정값 (로컬 메모리):**
- **클래스 수**: `+model.head.num_classes=8` (필수!)
- **훈련 스텝**: `training.max_steps=100000` (필수!)
- **Screen 사용**: 모든 훈련/validation에서 필수
- **데이터셋**: `dataset=gen4`
- **데이터 경로**: `dataset.path=/home/oeoiewt/eTraM/rvt_eTram/data/etram_cls8_sample`

### 📊 베이스라인 성능 (3-scale FPN)

**전체 성능:**
- **Overall mAP**: 34.02%
- **AP50**: 67.03%
- **AP75**: 30.79%

**크기별 성능:**
- **🔴 Small objects**: 17.28% mAP (클래스 2,3,4: Motorcycle, Bicycle, Pedestrian) ⚠️ **주요 개선 타겟**
- **🟡 Medium objects**: 34.03% mAP (클래스 0,1,5,6,7: Car, Truck, Bus, Static, Other)
- **🟢 Large objects**: 56.94% mAP (매우 큰 객체들)

### 📋 실험 한 사이클 표준 프로세스

#### Phase 1: 실험 설정 및 준비 (10분)

```bash
# 1. 실험 폴더 생성
EXPERIMENT_ID="4scale_enhanced_100k"  # 형식: {architecture}_{modification}_{steps}
mkdir -p experiments/${EXPERIMENT_ID}/{checkpoints,confusion_matrices,training_logs,validation_results}

# 2. 모델 설정 변경
# 파일: config/model/maxvit_yolox/default.yaml
# 3-scale: in_stages: [2, 3, 4]
# 4-scale: in_stages: [1, 2, 3, 4]  # P1 features 활성화

# 3. 설정 백업
cp config/model/maxvit_yolox/default.yaml experiments/${EXPERIMENT_ID}/model_config.yaml
```

#### Phase 2: 훈련 실행 (5-6시간)

```bash
# Screen에서 훈련 (필수!)
screen -dmS ${EXPERIMENT_ID}
screen -S ${EXPERIMENT_ID} -p 0 -X stuff "cd /home/oeoiewt/eTraM/rvt_eTram\n"
screen -S ${EXPERIMENT_ID} -p 0 -X stuff "python train.py model=rnndet dataset=gen4 dataset.path=/home/oeoiewt/eTraM/rvt_eTram/data/etram_cls8_sample +experiment/gen4='default.yaml' hardware.gpus=0 batch_size.train=6 batch_size.eval=2 hardware.num_workers.train=4 hardware.num_workers.eval=3 training.max_steps=100000 dataset.train.sampling=stream +model.head.num_classes=8 wandb.project_name=etram_enhanced wandb.group_name=${EXPERIMENT_ID}; echo 'Training completed! Press Enter to continue...'; read\n"
```

#### Phase 3: 결과 수집 및 정리 (30분)

```bash
# 1. 체크포인트 정리
cp dummy/${WANDB_ID}/checkpoints/epoch=*-step=100000-*.ckpt experiments/${EXPERIMENT_ID}/checkpoints/final_model.ckpt

# 2. Confusion Matrix 이동
mv confM/* experiments/${EXPERIMENT_ID}/confusion_matrices/

# 3. Screen 세션 정리
screen -r ${EXPERIMENT_ID}  # 완료 확인
```

#### Phase 4: Validation 및 상세 지표 (10분)

```bash
# Screen에서 Validation 실행 (필수!)
screen -dmS validation_${EXPERIMENT_ID}
screen -S validation_${EXPERIMENT_ID} -p 0 -X stuff "cd /home/oeoiewt/eTraM/rvt_eTram\n"
screen -S validation_${EXPERIMENT_ID} -p 0 -X stuff "python validation.py dataset=gen4 dataset.path=/home/oeoiewt/eTraM/rvt_eTram/data/etram_cls8_sample checkpoint=experiments/${EXPERIMENT_ID}/checkpoints/final_model.ckpt +experiment/gen4='default.yaml' hardware.gpus=0 batch_size.eval=8 +model.head.num_classes=8; echo 'Validation completed! Press Enter to continue...'; read\n"

# 결과 저장
mkdir -p validation_results/${EXPERIMENT_ID}
screen -r validation_${EXPERIMENT_ID} -X hardcopy /tmp/validation_${EXPERIMENT_ID}.txt
cp /tmp/validation_${EXPERIMENT_ID}.txt validation_results/${EXPERIMENT_ID}/validation_output.log
```

#### Phase 5: 결과 분석 및 문서화 (20분)

```bash
# 1. 성능 요약 파일 생성
# validation_results/${EXPERIMENT_ID}/metrics_summary.txt
# validation_results/${EXPERIMENT_ID}/evaluation_info.txt

# 2. 실험 결과 JSON 생성
# experiments/${EXPERIMENT_ID}/experiment_results.json
```

#### Phase 6: Git 관리 및 보존 (10분)

```bash
# 실험 결과 커밋
git add experiments/${EXPERIMENT_ID}/
git add validation_results/${EXPERIMENT_ID}/
git commit -m "feat: complete ${EXPERIMENT_ID} experiment

- Model: [구체적 아키텍처 설명]
- Performance: mAP X.XX% (+X.X% vs baseline)
- Small objects: X.XX% mAP (+X.X% improvement)
- Key findings: [주요 발견사항]

🤖 Generated with [Claude Code](https://claude.ai/code)
Co-Authored-By: Claude <noreply@anthropic.com>"

# Screen 세션 정리
screen -S ${EXPERIMENT_ID} -X quit
screen -S validation_${EXPERIMENT_ID} -X quit
```

### 🎯 성능 개선 목표

**Small Object Detection 목표:**
- **현재 베이스라인**: 17.28% mAP (Small objects)
- **4-scale FPN 목표**: 20-22% mAP (+15-25% 향상)
- **전체 성능 목표**: 36-38% mAP (+5-10% 향상)

**실험 시리즈:**
1. ✅ **3-scale Baseline** (완료): 34.02% mAP
2. 🔄 **4-scale Enhanced**: P1 features 추가 (진행 중)
3. 🔮 **Size-aware Loss**: Loss function 개선
4. 🔮 **Attention Modules**: Small object 전용 attention

### 📁 표준 파일 구조

```
experiments/{EXPERIMENT_ID}/
├── checkpoints/final_model.ckpt
├── confusion_matrices/*.png
├── model_config.yaml
├── training_logs/
├── validation_results/ → validation_results/{EXPERIMENT_ID}/
└── experiment_results.json

validation_results/{EXPERIMENT_ID}/
├── validation_output.log
├── metrics_summary.txt
└── evaluation_info.txt
```

### 🔬 실험 관리 원칙

1. **재현성**: 모든 설정을 Git으로 관리
2. **체계성**: 표준 폴더 구조 유지
3. **비교성**: 베이스라인 대비 성능 측정
4. **문서화**: 각 실험의 목적과 결과 명확히 기록

## 640×360 해상도 소형 객체 검출 혁신 전략

### 🎯 현재 성능 한계 및 목표

**현재 베이스라인 성능:**
- **Overall mAP**: 34.02%
- **Small objects mAP**: 17.28% (클래스 2,3,4: Motorcycle, Bicycle, Pedestrian)

**목표 성능:**
- **Small objects mAP**: 20-25% (+15-45% 향상)
- **Overall mAP**: 37-39% (+5-10% 향상)

### 📊 혁신적 접근법 우선순위

| 방법 | 예상 개선폭 | 구현 난이도 | 우선순위 |
|------|-------------|-------------|----------|
| **ConvLSTM + Temporal Attention** | +4-6% mAP | 중간 | 🔥 최고 |
| **Size-aware Loss v2** | +3-5% mAP | 낮음 | 🔥 최고 |
| **4-scale P1 최적화** | +2-3% mAP | 낮음 | ⚡ 높음 |
| **Deformable Conv + SE** | +2-4% mAP | 중간 | ⚡ 높음 |
| **VTEI + Advanced Aug** | +1-3% mAP | 높음 | 🎯 중간 |
| **Multi-res Training** | +3-5% mAP | 높음 | 🎯 중간 |

### 🚀 1단계: 고급 시간적 모델링 (이벤트 카메라 특화)

**이론적 근거**: 이벤트 카메라의 시간적 정보는 소형 객체의 motion pattern에서 핵심적 역할

#### A) ConvLSTM 강화 (Recurrent YOLOv8 기반)
```python
class EnhancedConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers=2):
        # Multi-temporal fusion으로 여러 시간 스케일 통합
        self.multi_temporal_fusion = MultiTemporalFusion()
        self.conv_lstm = ConvLSTM(input_channels, hidden_channels, num_layers)
```

#### B) Sparse Cross-Attention (ASTMNet 기반)
```python
class EventSparseAttention(nn.Module):
    def __init__(self, channels):
        # Event features와 backbone features 간 cross-attention
        self.cross_attention = SparseMultiHeadAttention()
        self.temporal_consistency = TemporalConsistencyModule()
```

#### C) Motion-Aware Feature Enhancement
```python
class MotionAwareEnhancer(nn.Module):
    def __init__(self):
        # Event polarity 기반 motion direction 예측
        self.motion_predictor = MotionPredictor()
        self.trajectory_tracker = TrajectoryTracker()
```

### ⚖️ 2단계: 적응적 Loss 함수 혁신

#### A) Size-Weighted Loss with Feedback
```python
class AdaptiveSizeAwareLoss(nn.Module):
    def forward(self, pred, target, bbox_sizes):
        # 동적 가중치 with feedback mechanism
        feedback_multiplier = self.compute_feedback(training_history)
        small_weight = torch.exp(-bbox_sizes / threshold) * feedback_multiplier
        return weighted_loss
```

#### B) Temporal Consistency Loss
```python
class TemporalConsistencyLoss(nn.Module):
    def forward(self, current_pred, previous_pred, motion_vectors):
        # 연속 프레임 간 small object tracking loss
        temporal_loss = self.consistency_penalty(current_pred, previous_pred)
        return temporal_loss
```

#### C) Hard Negative Mining for Small Objects
```python
class SmallObjectHardMining(nn.Module):
    def mine_hard_negatives(self, predictions, targets):
        # Small object 주변의 어려운 negative samples 강화 학습
        hard_negatives = self.select_hard_samples(predictions, targets)
        return hard_negatives
```

### 🔍 3단계: Multi-Scale Feature 혁신

#### A) 4-Scale FPN 최적화
```python
class OptimizedP1Features(nn.Module):
    def __init__(self):
        # P1 features를 small objects 전용으로 fine-tuning
        self.small_object_enhancer = SmallObjectEnhancer()
        self.scale_specific_norm = ScaleSpecificNormalization()
```

#### B) Squeeze-and-Excitation + Deformable Convolutions
```python
class AdaptiveFeatureModule(nn.Module):
    def __init__(self, channels):
        # 각 scale별 adaptive feature enhancement
        self.se_block = SEBlock(channels)
        self.deformable_conv = DeformableConv2d(channels, channels)
```

#### C) Adaptive Feature Fusion
```python
class EventDensityFusion(nn.Module):
    def __init__(self):
        # Event density에 따른 dynamic feature fusion
        self.density_estimator = EventDensityEstimator()
        self.adaptive_fusion = AdaptiveFusionLayer()
```

### 📡 4단계: Event Data 처리 혁신

#### A) Volume of Ternary Event Images (VTEI)
```python
class VTEIRepresentation(nn.Module):
    def __init__(self):
        # Positive/Negative/Zero states로 세분화
        self.ternary_encoder = TernaryEventEncoder()
        self.volume_processor = VolumeProcessor()
```

#### B) Random Polarity Suppression
```python
class EventAugmentation(nn.Module):
    def __init__(self):
        # Small objects에 특화된 augmentation strategies
        self.polarity_suppression = PolaritySuppressionAug()
        self.noise_injection = NoiseInjectionAug()
```

#### C) Sparse Data Optimization
```python
class SparseEventProcessor(nn.Module):
    def __init__(self):
        # Memory-efficient sparse tensor operations
        self.sparse_conv = SparseConv3d()
        self.sparse_attention = SparseAttentionLayer()
```

### 🏗️ 5단계: 아키텍처 수준 혁신

#### A) Multi-Resolution Training
```python
class MultiResolutionTraining:
    def __init__(self):
        # 640×360 + 1280×720 mixed training
        self.resolution_scheduler = ResolutionScheduler()
        self.scale_invariance_loss = ScaleInvarianceLoss()
```

#### B) Teacher-Student Distillation
```python
class SmallObjectDistillation(nn.Module):
    def __init__(self, teacher_model, student_model):
        # High-resolution teacher → Low-resolution student
        self.knowledge_transfer = KnowledgeTransferModule()
        self.feature_distillation = FeatureDistillationLoss()
```

#### C) Neural Architecture Search (NAS)
```python
class EventNAS:
    def __init__(self):
        # Small object detection에 특화된 architecture 자동 탐색
        self.search_space = EventBasedSearchSpace()
        self.performance_estimator = SmallObjectPerformanceEstimator()
```

### 🎯 실험 로드맵

#### Phase 1: 즉시 구현 (1-2주)
1. **Size-aware Loss v2**: Dynamic feedback mechanism 추가
2. **4-scale P1 최적화**: P1 features 전용 처리 모듈
3. **ConvLSTM 강화**: Multi-temporal fusion 구현

#### Phase 2: 중기 구현 (2-4주)
1. **Temporal Attention**: Event-specific attention mechanisms
2. **Deformable Convolutions**: Shape-adaptive feature extraction
3. **Advanced Augmentation**: Small object 특화 data augmentation

#### Phase 3: 장기 실험 (4-8주)
1. **Multi-resolution Training**: Scale invariance 강화
2. **Teacher-Student Distillation**: Knowledge transfer
3. **Neural Architecture Search**: Optimal architecture 탐색

### 📈 예상 성능 향상

**누적 개선 효과:**
- **Phase 1 완료**: 17.28% → 19-21% mAP (+10-20%)
- **Phase 2 완료**: 19-21% → 22-24% mAP (+15-25%)  
- **Phase 3 완료**: 22-24% → 25-27% mAP (+20-30%)

**최종 목표**: Small objects 25% mAP, Overall 38-40% mAP

## 표준 실험 문서화 프로세스

### 📋 필수 문서화 파일 구조

```
experiments/{EXPERIMENT_ID}/
├── experiment_hypothesis.txt      # 실험 가설 및 이론적 근거
├── modification_details.txt       # 코드 수정사항 상세 기록  
├── implementation_details.txt     # 구현 세부사항 및 아키텍처
├── experiment_config.yaml         # 실험 설정 파일 백업
├── code_changes_summary.txt       # 주요 변경사항 요약
├── training_command.txt           # 실제 사용한 훈련 명령어
├── memory_test_results.txt        # 메모리 테스트 및 최적화 결과
├── experiment_results.json        # 최종 성능 결과 및 분석
├── checkpoints/final_model.ckpt   # 최종 훈련된 모델
├── confusion_matrices/            # Confusion matrix 이미지들
├── training_logs/                 # 훈련 로그 파일들
└── validation_results/            # Validation 상세 결과
```

### 🔄 실험 단계별 체크리스트

#### 1. 실험 기획 단계 (Day 0)
- [ ] **실험 가설 수립**: `experiment_hypothesis.txt` 작성
- [ ] **이론적 근거 정리**: 기존 연구 및 예상 효과 분석
- [ ] **성공 기준 설정**: 정량적 성능 목표 설정
- [ ] **리스크 분석**: 잠재적 실패 원인 및 대응책

#### 2. 구현 단계 (Day 1-3)
- [ ] **코드 수정사항 기록**: `modification_details.txt` 실시간 업데이트
- [ ] **설정 파일 백업**: `experiment_config.yaml` 저장
- [ ] **Git 브랜치 생성**: 실험 전용 브랜치에서 작업
- [ ] **Unit Test 실행**: 개별 컴포넌트 기능 검증

#### 3. 메모리 최적화 단계 (Day 3-4)
- [ ] **메모리 테스트**: `memory_test_results.txt` 작성
- [ ] **Batch size 최적화**: OOM 방지 및 성능 균형
- [ ] **Hardware 설정 확인**: GPU 메모리 및 Workers 수 조정

#### 4. 훈련 실행 단계 (Day 4-5)
- [ ] **Screen 세션 생성**: 장시간 훈련 안정성 확보
- [ ] **훈련 명령어 기록**: `training_command.txt` 저장
- [ ] **실시간 모니터링**: WandB 또는 로그를 통한 진행상황 추적
- [ ] **중간 체크포인트 확인**: Overfitting 및 수렴성 모니터링

#### 5. 검증 및 분석 단계 (Day 5-6)
- [ ] **Validation 실행**: 표준 평가 스크립트 사용
- [ ] **성능 지표 수집**: mAP, AP50, AR 등 상세 메트릭
- [ ] **베이스라인 비교**: 기존 실험 대비 성능 변화 분석
- [ ] **Confusion Matrix 분석**: 클래스별 성능 개선/저하 확인

#### 6. 문서화 및 정리 단계 (Day 6-7)
- [ ] **실험 결과 종합**: `experiment_results.json` 완성
- [ ] **주요 발견사항 정리**: 성공/실패 원인 분석
- [ ] **다음 실험 방향 제시**: 개선점 및 후속 연구 계획
- [ ] **Git 커밋**: 모든 실험 결과물 저장

### 📊 성능 평가 표준

#### 정량적 지표
- **Overall mAP**: 전체 평균 정밀도
- **Small Objects mAP**: 클래스 2,3,4 (Motorcycle, Bicycle, Pedestrian) 성능
- **AP50/AP75**: IoU threshold별 성능
- **AR (Average Recall)**: 재현율 지표

#### 정성적 분석
- **훈련 안정성**: Loss 수렴 패턴 및 안정성
- **메모리 효율성**: 메모리 사용량 및 훈련 속도
- **실용성**: 실제 배포 가능성 및 계산 복잡도

#### 비교 기준
- **베이스라인**: 3-scale baseline (34.02% mAP, 17.28% small objects)
- **상대 성능**: 베이스라인 대비 개선/저하 정도
- **실험 시리즈**: 동일 카테고리 실험들 간 비교

### 🚨 일반적 실험 실패 원인 및 대응

#### 메모리 관련 실패
- **원인**: Batch size 과다, Workers 수 부적절
- **대응**: 체계적 메모리 테스트 및 단계적 조정

#### 수렴 관련 실패  
- **원인**: Learning rate 부적절, Loss 함수 불균형
- **대응**: 훈련 초기 상세 모니터링 및 조기 중단

#### 성능 저하
- **원인**: 과도한 복잡성, 데이터셋 미스매치
- **대응**: 단계적 복잡성 증가 및 Ablation study

#### 재현성 실패
- **원인**: 설정 파일 불일치, Random seed 미설정
- **대응**: 완전한 설정 백업 및 환경 고정

## 실험 성능 순위 및 주요 발견사항

### 📊 전체 실험 성능 순위 (Overall mAP)

| 순위 | 실험명 | Overall mAP | Small Objects mAP | 아키텍처 | 주요 특징 |
|------|--------|-------------|-------------------|----------|-----------|
| 🥇 | **3scale_sizeaware_100k** | **34.08%** | 13.53% | 3-scale + Size-aware Loss | 안정적 최고 성능 |
| 🥈 | **3scale_baseline** | **34.02%** | **17.28%** | 3-scale FPN | Small objects 최고 |
| 🥉 | **4scale_sizeaware_100k** | 32.23% | 12.75% | 4-scale + Size-aware | P1 features 활용 |
| 4 | **ABC_sod_basic_100k** | 31.7% | 14.8% | 4-scale + Multi-task | ABC 접근법 |
| 5 | **patch2_4scale_sizeaware_200k** | 31.24% | 14.92% | patch=2 + 4-scale | Memory 제약 |
| 6 | **4scale_enhanced_100k** | 30.93% | 14.83% | 4-scale FPN | P1 노이즈 문제 |
| 7 | **3scale_sizeaware_attention_100k** | 24.7% | TBD | 3-scale + Attention | 극심한 성능 저하 |

### 🔍 Small Objects 성능 순위

| 순위 | 실험명 | Small Objects mAP | 변화량 | 상태 |
|------|--------|------------------|--------|------|
| 🥇 | **3scale_baseline** | **17.28%** | 기준점 | ✅ 최고 성능 |
| 2 | **ABC_sod_basic_100k** | 14.8% | -2.5% | ❌ 하락 |
| 3 | **patch2_4scale_sizeaware_200k** | 14.92% | -2.36% | ❌ 하락 |
| 4 | **4scale_enhanced_100k** | 14.83% | -2.45% | ❌ 하락 |
| 5 | **3scale_sizeaware_100k** | 13.53% | -3.75% | ❌ 하락 |
| 6 | **4scale_sizeaware_100k** | 12.75% | -4.53% | ❌ 하락 |

### 🚨 핵심 발견사항

#### 1. **복잡성 역설** (Complexity Paradox)
**발견**: 모든 "개선" 시도가 베이스라인보다 성능 저하를 일으켰음

**관찰된 패턴**:
```
복잡성 순서: 3scale_baseline < ABC < 4scale < attention
성능 순서:   3scale_baseline > ABC > 4scale > attention
```

**결론**: 640×360 해상도에서 **단순함이 최고의 성능**을 보장

#### 2. **해상도 제약의 근본적 한계**
**발견**: 아키텍처 개선보다 **해상도 증가**가 더 중요

**증거**:
- P1 features (stride 4) 활용 시도들 모두 실패
- 고해상도 features의 노이즈 문제
- Small objects 정보 부족 (640×360 제약)

**제안**: 1280×720 해상도 우선 실험 필요

#### 3. **Multi-task Learning의 한계**
**ABC 실험 교훈**:
- Multi-task learning이 단일 task보다 어려움
- Gradient conflicts로 인한 최적화 문제
- Small dataset에서 복잡한 architecture의 과적합 위험

#### 4. **Small Object Detection의 근본 문제**
**관찰**: 모든 small object 개선 시도 실패

**원인 분석**:
- **데이터 품질**: Event-based data의 sparse nature
- **해상도 한계**: 640×360에서 small objects 정보 부족
- **클래스 불균형**: Motorcycle(16K) vs Bicycle(1K) instances

### 🎯 혁신적 접근법 우선순위 (업데이트)

| 접근법 | 기존 예상 | 실제 결과 | 수정된 우선순위 |
|--------|----------|----------|-----------------|
| **해상도 증가** | +3-5% mAP | **미실험** | 🔥 **최우선** |
| **Data-centric 개선** | +1-3% mAP | **미실험** | 🔥 **최우선** |
| **Simple Loss 개선** | +3-5% mAP | **부분 성공** | ⚡ 높음 |
| ~~Multi-task Learning~~ | +4-6% mAP | **-2.3% 실패** | ❌ 비추천 |
| ~~4-scale FPN~~ | +2-3% mAP | **-3% 실패** | ❌ 비추천 |
| ~~Attention Mechanisms~~ | +4-6% mAP | **-9.3% 실패** | ❌ 절대 금지 |

### 💡 실용적 권장사항

#### 즉시 실행할 방향
1. **해상도 증가**: 640×360 → 1280×720 실험
2. **Data augmentation**: 아키텍처 수정 대신 데이터 다양성
3. **3-scale baseline 최적화**: 검증된 아키텍처 세밀 조정

#### 피해야 할 접근법
1. **복잡한 아키텍처**: Attention, Multi-task learning
2. **P1 features 활용**: 노이즈 대비 이득 없음
3. **과도한 engineering**: Simple is better

### 📈 성능 개선 로드맵 (수정)

#### Phase 1: Resolution First (최우선)
- **목표**: 1280×720에서 베이스라인 재현
- **예상 효과**: Small objects 20-25% mAP 달성 가능

#### Phase 2: Data-Centric (고우선순위)
- **목표**: Advanced augmentation, 데이터 품질 개선
- **예상 효과**: 추가 2-3% mAP 향상

#### Phase 3: Focused Enhancement (중간 우선순위)
- **목표**: 검증된 simple modifications만
- **예상 효과**: 미세 조정으로 1-2% 추가 향상

**최종 목표**: Small objects 25% mAP, Overall 40% mAP (고해상도에서)

이러한 발견사항들은 Event-based Small Object Detection 연구에서 **"복잡함보다 기본에 충실"**하라는 중요한 교훈을 제공합니다.