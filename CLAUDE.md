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