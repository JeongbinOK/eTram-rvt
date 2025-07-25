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

## Memory Management

### Experimental Guidelines

**실험 진행 시 준수사항:**
- 항상 `rvt_eTram/experiments/3scale_sizeaware_attention_100k/` 디렉토리 아래에 있는 형식들을 지켜서 실험 결과를 정리해줘
- 실험 결과 문서화 시 표준 구조 유지
- 실험 가설, 수정사항, 성능 지표를 명확히 기록

### 실험 문서화 표준

#### 필수 파일 구조
```
experiments/3scale_sizeaware_attention_100k/
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

### 성능 평가 지표

#### 정량적 지표
- **Overall mAP**: 전체 평균 정밀도
- **Small Objects mAP**: 소형 객체 성능
- **AP50/AP75**: IoU threshold별 성능
- **AR (Average Recall)**: 재현율 지표

### 주요 실험 원칙

1. **재현성 보장**: 모든 설정과 변경사항 명확히 기록
2. **단순성 유지**: 불필요한 복잡성 피하기
3. **데이터 중심 접근**: 아키텍처보다 데이터 품질 개선 우선
4. **표준 절차 준수**: 실험 문서화 가이드라인 엄격히 따르기

## Experimental Insights

### Key Findings

#### 1. 복잡성 역설 (Complexity Paradox)
- 단순한 베이스라인 모델이 종종 복잡한 모델보다 우수한 성능
- 아키텍처 개선보다 데이터 품질에 집중

#### 2. 해상도의 중요성
- 해상도 증가가 아키텍처 개선보다 더 큰 성능 향상 가능
- 1280×720 해상도 실험 강력 추천

#### 3. Small Object Detection 전략
- 이벤트 기반 데이터의 희소성 고려
- 데이터 증강 및 품질 개선에 집중

### Recommended Next Steps

1. 해상도 증가 실험 (640×360 → 1280×720)
2. 데이터 증강 기법 개발
3. 베이스라인 모델의 세밀한 최적화
4. 단순하고 명확한 접근법 유지