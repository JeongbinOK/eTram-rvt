# Small Object Detection Enhancement for eTraM Dataset

**4-scale Feature Pyramid Network for Event-based Traffic Monitoring**

<div>
<a href="https://eventbasedvision.github.io/eTraM/">Original eTraM Dataset</a> |
<a href="./rvt_eTram/">Enhanced RVT Implementation</a> |
<a href="./results/">Experiment Results</a>
</div>

---

## 🎯 Project Overview

This repository contains an enhanced implementation of the eTraM (Event-based Traffic Monitoring) dataset focused on **improving small object detection performance**. We extend the original RVT (Recurrent Vision Transformers) architecture with a 4-scale Feature Pyramid Network to better detect small traffic participants such as motorcycles, bicycles, and pedestrians.

### Key Improvements

- ✅ **4-scale Feature Pyramid Network**: Extended from 3-scale to 4-scale FPN with P1 features (stride 4)
- ✅ **Detailed Metrics System**: Class-wise mAP, AP50, AP75, AP95 for all 8 traffic classes
- ✅ **Experiment Tracking**: JSON-based result storage with Git integration
- ✅ **Small Object Analysis**: Specialized evaluation for small traffic participants
- 🔄 **Size-aware Loss Function**: (In development)
- 🔄 **Temporal Enhancement**: (Future work)

## 🏗️ Architecture Enhancement

### Original vs Enhanced Architecture

| Component | Original (3-scale) | Enhanced (4-scale) | Result |
|-----------|-------------------|-------------------|---------|
| **FPN Scales** | 8, 16, 32 | **4, 8, 16, 32** | ✅ Successfully implemented |
| **P1 Features** | ❌ Not used | ✅ **Enabled for small objects** | ✅ Architecture works |
| **Small Object mAP** | **17.28%** | **14.83%** | ❌ **-2.45% decrease** |
| **Overall mAP** | **34.02%** | **30.93%** | ❌ **-3.09% decrease** |
| **Detection Resolution** | 1/8 minimum | **1/4 minimum** | ⚠️ **Higher res = worse performance** |

### Feature Pyramid Network Flow

```
Input Event Data (H×W)
    ↓
┌─────────────────────────────────────────────────┐
│            RVT Backbone (MaxViT + LSTM)        │
├─────────────────────────────────────────────────┤
│ P1: H/4×W/4   P2: H/8×W/8   P3: H/16×W/16   P4: H/32×W/32 │
│ (stride 4)    (stride 8)    (stride 16)     (stride 32)   │
│    NEW!         Original        Original        Original   │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│              4-scale FPN Network                │
├─────────────────────────────────────────────────┤
│ N1: Small     N2: Medium    N3: Large     N4: X-Large     │
│ (stride 4)    (stride 8)    (stride 16)   (stride 32)     │
│ Motorcycles   Cars          Trucks        Buses           │
│ Bicycles      Pedestrians                                  │
│ Small Peds                                                 │
└─────────────────────────────────────────────────┘
    ↓
Detection Head (YOLOX) → Final Predictions
```

## 📊 Performance Results

### Experimental Results Summary

| Experiment | Overall mAP | Small Objects mAP | AP50 | AP75 | Status |
|------------|-------------|-------------------|------|------|--------|
| **3-scale Baseline** | **34.02%** | **17.28%** | **67.03%** | **30.79%** | ✅ Completed |
| **4-scale Enhanced** | **30.93%** | **14.83%** | **62.34%** | **27.30%** | ✅ Completed |
| **Performance Change** | **-3.09%** | **-2.45%** | **-4.69%** | **-3.49%** | ⚠️ **Unexpected decrease** |

### Key Findings from 4-scale FPN Experiment

**❌ Unexpected Results**: The 4-scale FPN with P1 features actually **decreased** performance instead of improving it.

**Critical Analysis**:
- **Small Objects** (Motorcycle, Bicycle, Pedestrian): 17.28% → 14.83% **(-2.45%)**
- **Overall Performance**: All metrics showed degradation
- **Hypothesis**: P1 features alone are insufficient and may introduce noise

**Research Implications**:
- Adding high-resolution P1 features requires careful training strategy adjustments
- Model complexity increased without proportional performance gains
- Need for size-aware loss functions and specialized training approaches

## 🚀 Quick Start

### Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd eTraM

# Create conda environment
cd rvt_eTram
conda env create -f environment.yaml
conda activate rvt
```

### Training Commands

#### 1. Baseline 3-scale FPN (for comparison)
```bash
python train.py model=rnndet dataset=gen4 \
  dataset.path=/path/to/etram_cls8_sample \
  +experiment/gen4="default.yaml" \
  hardware.gpus=0 batch_size.train=6 batch_size.eval=2 \
  training.max_steps=100000 +model.head.num_classes=8 \
  wandb.project_name=etram_baseline wandb.group_name=3scale_baseline
```

#### 2. Enhanced 4-scale FPN
```bash
# First, enable 4-scale FPN in config
# config/model/maxvit_yolox/default.yaml: in_stages: [1, 2, 3, 4]

python train.py model=rnndet dataset=gen4 \
  dataset.path=/path/to/etram_cls8_sample \
  +experiment/gen4="default.yaml" \
  hardware.gpus=0 batch_size.train=6 batch_size.eval=2 \
  training.max_steps=100000 +model.head.num_classes=8 \
  wandb.project_name=etram_enhanced wandb.group_name=4scale_enhanced
```

### Evaluation

```bash
python validation.py dataset=gen4 \
  dataset.path=/path/to/etram_cls8_sample \
  checkpoint=/path/to/checkpoint.ckpt \
  +experiment/gen4="default.yaml" \
  hardware.gpus=0 batch_size.eval=8
```

## 📁 Project Structure

```
eTraM/
├── README.md                           # This enhanced documentation
├── rvt_eTram/                         # Enhanced RVT implementation
│   ├── utils/evaluation/              # 📊 NEW: Detailed metrics system
│   │   ├── detailed_metrics.py       #     Class-wise evaluation
│   │   └── experiment_logger.py      #     JSON experiment tracking
│   ├── experiments/                   # 📊 NEW: Experiment results storage
│   ├── config/model/maxvit_yolox/     # 🔧 UPDATED: 4-scale FPN configs
│   ├── models/detection/              # Enhanced detection models
│   │   └── yolox_extension/models/    #     4-scale PAFPN implementation
│   ├── modules/detection.py          # 🔧 UPDATED: Detailed metrics integration
│   └── train.py                      # Main training script
├── ultralytics_eTram/                 # YOLO implementation (original)
├── confM/                            # 📊 NEW: Confusion matrices
├── results/                          # 📊 NEW: Experiment results
└── docs/                            # 📊 NEW: Additional documentation
```

## 🔬 Experimental Methodology

### Dataset Configuration

- **Training Data**: eTraM sample dataset (`etram_cls8_sample`)
- **Classes**: 8 traffic participant classes
- **Event Representation**: Stacked histograms (dt=50ms, 10 bins)
- **Resolution**: 640×384 (downsampled by factor 2)

### Training Strategy

1. **Baseline Experiment**: 3-scale FPN (100k steps)
2. **Enhanced Experiment**: 4-scale FPN (100k steps)
3. **Comparison Analysis**: Class-wise performance evaluation
4. **Small Object Focus**: Specialized metrics for classes 2, 3, 4

### Metrics and Evaluation

- **COCO-style metrics**: mAP, AP50, AP75, AP95
- **Class-wise analysis**: Individual performance per traffic class
- **Size-based analysis**: Small vs large object performance
- **Confusion matrices**: Detailed classification analysis

## 🛠️ Technical Implementation

### Key Files Modified

1. **`config/model/maxvit_yolox/default.yaml`**
   ```yaml
   fpn:
     in_stages: [1, 2, 3, 4]  # Enable P1 features
   ```

2. **`models/detection/yolox_extension/models/yolo_pafpn.py`**
   - Extended YOLOPAFPN to support 4-scale processing
   - Added P1 feature pathway for small objects

3. **`utils/evaluation/detailed_metrics.py`**
   - Comprehensive class-wise metrics calculation
   - Small object analysis framework

4. **`modules/detection.py`**
   - Integrated detailed metrics into validation pipeline
   - Automated experiment result logging

### 4-scale FPN Implementation Details

The enhanced FPN processes features at 4 different scales:

- **P1 (stride 4)**: Highest resolution for small objects (motorcycles, bicycles, small pedestrians)
- **P2 (stride 8)**: Medium-small objects (pedestrians, small vehicles)  
- **P3 (stride 16)**: Medium-large objects (cars, trucks)
- **P4 (stride 32)**: Large objects (buses, large trucks)

## 📈 Experiment Tracking

### JSON-based Result Storage

Each experiment generates comprehensive results stored in `experiments/`:

```json
{
  "experiment_metadata": {
    "experiment_id": "4scale_fpn_e001_s010000",
    "git_commit": "8d7f046",
    "timestamp": "2025-07-07_15:30:45"
  },
  "model_modifications": {
    "architecture_changes": [
      "Added P1 features (stride 4) to backbone output",
      "Extended FPN from 3-scale to 4-scale"
    ]
  },
  "evaluation_results": {
    "overall_metrics": { "mAP": 0.XX, "AP50": 0.XX },
    "class_metrics": { ... },
    "small_object_analysis": { ... }
  }
}
```

### Git-based Version Control

- Each major experiment is committed to Git
- Automatic tracking of configuration changes
- Easy comparison between experiment versions

## 🔬 Research Findings & Lessons Learned

### Completed Experiments (July 2025)

#### ✅ 4-scale FPN Experiment (Phase 1)
**Status**: Completed - **Negative Results** 🔍

**What we learned**:
1. **P1 features alone don't improve small object detection** - Counter-intuitive result
2. **Model complexity requires adjusted training strategies** - Same training config failed
3. **High-resolution features may contain excess noise** - Signal-to-noise ratio critical
4. **Negative results are valuable** - Guide future research directions

**Technical Implementation**:
- ✅ Successfully extended FPN from 3-scale to 4-scale
- ✅ Added P1 features (stride 4) for highest resolution detection
- ✅ Modified YOLOPAFPN architecture with backward compatibility
- ❌ Performance decreased across all metrics (-3.09% overall mAP)

### Next Research Directions

#### Phase 2: Corrective Strategies (High Priority)
- [ ] **Size-aware loss functions** - Weight small objects properly
- [ ] **Denoising techniques for P1 features** - Filter high-frequency noise
- [ ] **Progressive training strategy** - Start with 3-scale, gradually add P1
- [ ] **Longer training with adjusted learning rates** - Account for model complexity

#### Phase 3: Advanced Components (Medium Priority)
- [ ] Small object attention mechanisms  
- [ ] Temporal feature enhancement modules
- [ ] Multi-scale training strategy
- [ ] Hard negative mining for small objects

#### Phase 4: Architecture Exploration (Future Work)
- [ ] Deformable convolutions for small objects
- [ ] Neural architecture search adaptations
- [ ] Event-specific attention mechanisms

## 📚 Related Work and References

### Original eTraM Dataset
```bibtex
@InProceedings{Verma_2024_CVPR,
    author    = {Verma, Aayush Atul and Chakravarthi, Bharatesh and Vaghela, Arpitsinh and Wei, Hua and Yang, Yezhou},
    title     = {eTraM: Event-based Traffic Monitoring Dataset},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {22637-22646}
}
```

### RVT Architecture
- Based on Recurrent Vision Transformers for event-based detection
- Enhanced with 4-scale Feature Pyramid Network
- Optimized for small object detection in traffic scenarios

## 📄 License

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

## 🤝 Contributing

1. **Experiment Results**: Share your experimental findings
2. **Code Improvements**: Submit pull requests for enhancements  
3. **Issues**: Report bugs or request features
4. **Documentation**: Help improve documentation and examples

---

**Note**: This is an enhanced research implementation of the original eTraM dataset, focused on improving small object detection for event-based traffic monitoring. All credit for the original dataset goes to the eTraM authors.