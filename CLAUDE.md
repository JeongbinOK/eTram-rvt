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
  training.max_epochs=20 dataset.train.sampling=stream +model.head.num_classes=8
```

**Evaluation:**
```bash
# Evaluate model
python validation.py dataset=gen4 dataset.path=${DATA_DIR} checkpoint=${CKPT_PATH} \
  use_test_set=${USE_TEST} hardware.gpus=${GPU_ID} +experiment/gen4="${MDL_CFG}.yaml" \
  batch_size.eval=8 model.postprocess.confidence_threshold=0.001
```

### Validation 실행 가이드 (Troubleshooting)



#### 4. 검증된 완전한 Validation 명령어
```bash
# 🎯 성공적인 Validation 실행 템플릿
python validation.py \
  dataset=gen4 \
  model=maxvit_yolox/MODEL_CONFIG \
  checkpoint=SIMPLE_FILENAME.ckpt \
  use_test_set=false \
  hardware.gpus=0 \
  dataset.path=/home/oeoiewt/eTraM/rvt_eTram/data/etram_cls8_sample \
  +batch_size.train=6

# 예시: Size-aware Loss 모델 검증
python validation.py \
  dataset=gen4 \
  model=maxvit_yolox/plain_lstm_sizeaware \
  checkpoint=/path/to/simple_filename.ckpt \
  use_test_set=false \
  hardware.gpus=0 \
  dataset.path=/home/oeoiewt/eTraM/rvt_eTram/data/etram_cls8_sample \
  +batch_size.train=6
```

#### 5. Screen Session에서 안전한 실행
```bash
# Screen session 생성 및 실행
screen -dmS validation_session
screen -S validation_session -p 0 -X stuff "source /home/oeoiewt/miniconda3/etc/profile.d/conda.sh && conda activate rvt && cd /home/oeoiewt/eTraM/rvt_eTram\n"
screen -S validation_session -p 0 -X stuff "python validation.py [PARAMETERS]\n"

# 진행 상황 확인
screen -S validation_session -X hardcopy /tmp/validation_status.txt
tail -20 /tmp/validation_status.txt
```

#### 6. Checkpoint 파일명 처리
```bash
# 특수문자(=, -, 등) 포함된 checkpoint 파일명 처리
ORIGINAL_CKPT="/path/epoch=001-step=100000-val_AP=0.31.ckpt"
SIMPLE_CKPT="/path/model_checkpoint.ckpt"
cp "$ORIGINAL_CKPT" "$SIMPLE_CKPT"

# 간단한 파일명으로 validation 실행
python validation.py checkpoint="$SIMPLE_CKPT" [OTHER_PARAMS]
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

## Validation 표준 프로세스

### 성공적인 Validation 실행 방법

**핵심 원칙**: `+experiment/gen4='실험설정.yaml'` 형식을 사용하여 전체 설정을 로드

**표준 명령어 템플릿**:
```bash
python validation.py \
  dataset=gen4 \
  checkpoint=CHECKPOINT_PATH \
  +experiment/gen4='EXPERIMENT_CONFIG.yaml' \
  hardware.gpus=0 \
  batch_size.eval=2 \
  hardware.num_workers.eval=1
```

### 주요 해결 방법들

#### 1. Hydra Configuration 오류 해결
- **문제**: `ConfigAttributeError: Key 'train' is not in struct`
- **해결**: 전체 실험 설정 파일 사용 (`+experiment/gen4='config.yaml'`)
- **예시**: `+experiment/gen4='plain_lstm_3scale_sizeaware_640x360.yaml'`

#### 2. Checkpoint 경로 문제
- **문제**: `FileNotFoundError` - checkpoint 파일을 찾을 수 없음
- **해결**: 실제 checkpoint 위치 확인 (`dummy/[실험ID]/checkpoints/`)
- **명령**: `ls experiments/*/checkpoints/` 또는 `ls dummy/*/checkpoints/`

#### 3. Worker 설정 오류
- **문제**: `AssertionError: Each worker must at least get 'batch_size' number of datapipes`
- **해결**: 낮은 worker와 batch size 설정
- **권장값**: `batch_size.eval=2`, `hardware.num_workers.eval=1`

#### 4. Override 문법 오류
- **문제**: `Could not append to config` 또는 `ConfigKeyError`
- **해결**: 적절한 override 접두사 사용
  - 새 키 추가: `+key=value`
  - 기존 키 강제 덮어쓰기: `++key=value`
  - 일반 설정: `key=value`

### 검증된 성공 사례

#### Plain LSTM + Size-aware Loss
```bash
python validation.py \
  dataset=gen4 \
  checkpoint=dummy/zifokzkb/checkpoints/last_model.ckpt \
  +experiment/gen4='plain_lstm_3scale_sizeaware_640x360.yaml' \
  hardware.gpus=0 \
  batch_size.eval=2 \
  hardware.num_workers.eval=1
```

#### 이전 실험 (3scale_sizeware_100k)
```bash
python validation.py \
  dataset=gen4 \
  dataset.path=/home/oeoiewt/eTraM/rvt_eTram/data/etram_cls8_sample \
  checkpoint=experiments/3scale_sizeaware_100k/checkpoints/final_model.ckpt \
  +experiment/gen4='3scale_sizeaware.yaml' \
  hardware.gpus=0 \
  batch_size.eval=2 \
  hardware.num_workers.eval=1
```

### 표준 실험 워크플로우

#### Phase 1: 실험 계획 및 설정
1. **실험 가설 및 목표 설정**
   - `experiments/[실험명]/experiment_hypothesis.txt` 작성
   - 이론적 근거 및 예상 개선점 명시

2. **Configuration 파일 생성**
   - Model config: `config/model/maxvit_yolox/[모델명].yaml`
   - Experiment config: `config/experiment/gen4/[실험명].yaml`
   - 하이퍼파라미터 최적화 (특히 size-aware loss)

3. **실험 디렉토리 구조 생성**
   ```bash
   mkdir -p experiments/[실험명]/{checkpoints,confusion_matrices,training_logs,validation_results}
   ```

#### Phase 2: 훈련 실행
1. **훈련 명령어 준비**
   ```bash
   python train.py dataset=gen4 \
     dataset.path=/home/oeoiewt/eTraM/rvt_eTram/data/etram_cls8_sample \
     +experiment/gen4='[실험설정].yaml' \
     hardware.gpus=0 \
     batch_size.train=6 batch_size.eval=2 \
     hardware.num_workers.train=4 hardware.num_workers.eval=3 \
     training.max_steps=100000 \
     dataset.train.sampling=stream \
     wandb.project_name=[프로젝트명] \
     wandb.group_name=[실험그룹명]
   ```

2. **Screen 세션에서 훈련 실행**
   ```bash
   screen -dmS [실험명]_100k
   screen -S [실험명]_100k -p 0 -X stuff "cd /home/oeoiewt/eTraM/rvt_eTram\n"
   screen -S [실험명]_100k -p 0 -X stuff "[훈련명령어]\n"
   ```

3. **훈련 모니터링**
   ```bash
   screen -r [실험명]_100k  # 직접 모니터링
   screen -list  # 활성 세션 확인
   ```

#### Phase 3: 훈련 완료 확인
1. **훈련 상태 확인**
   ```bash
   screen -r [실험명]_100k  # 훈련 완료 확인
   ls dummy/*/checkpoints/  # checkpoint 생성 확인
   ```

2. **Checkpoint 복사 및 정리**
   ```bash
   cp dummy/[실험ID]/checkpoints/last_model.ckpt experiments/[실험명]/checkpoints/
   cp confM/*.png experiments/[실험명]/confusion_matrices/
   ```

#### Phase 4: Validation 실행
1. **표준 Validation 명령어**
   ```bash
   python validation.py \
     dataset=gen4 \
     checkpoint=dummy/[실험ID]/checkpoints/last_model.ckpt \
     +experiment/gen4='[실험설정].yaml' \
     hardware.gpus=0 \
     batch_size.eval=2 \
     hardware.num_workers.eval=1
   ```

2. **결과 저장**
   ```bash
   # 백그라운드에서 실행하고 결과 저장
   python validation.py [...] > experiments/[실험명]/validation_results.txt 2>&1 &
   ```

3. **Validation 진행 모니터링**
   ```bash
   tail -f experiments/[실험명]/validation_results.txt  # 실시간 모니터링
   ps aux | grep validation  # 프로세스 상태 확인
   ```

#### Phase 5: 결과 분석 및 문서화
1. **성능 지표 추출**
   - Overall mAP, Small Objects AP, COCO metrics
   - 클래스별 성능 (특히 Classes 2,3,4)
   - Confusion matrix 분석

2. **실험 결과 문서화**
   ```bash
   # 표준 파일 구조 생성
   experiments/[실험명]/
   ├── experiment_hypothesis.txt      # 실험 가설
   ├── modification_details.txt       # 구현 세부사항
   ├── experiment_config.yaml         # 설정 백업
   ├── training_command.txt           # 훈련 명령어
   ├── experiment_results.json        # 성능 결과
   ├── checkpoints/final_model.ckpt   # 최종 모델
   ├── confusion_matrices/            # 혼돈 행렬
   └── validation_results/            # 검증 결과
   ```

3. **성능 비교 분석**
   - 이전 실험 대비 개선점 측정
   - Small object detection 효과 정량화
   - 실패 사례 분석 및 개선 방향 제시

#### Phase 6: 실험 완료 및 다음 단계
1. **Git 커밋**
   ```bash
   git add experiments/[실험명]/
   git commit -m "feat: complete [실험명] experiment with [주요결과]"
   ```

2. **다음 실험 계획**
   - 현재 결과 기반 개선 방향 설정
   - 하이퍼파라미터 조정 계획
   - 아키텍처 변경 검토

### 문제 해결 체크리스트

- [ ] 올바른 experiment config 파일 사용
- [ ] Checkpoint 경로 존재 확인
- [ ] 낮은 batch_size와 num_workers 설정
- [ ] 적절한 Hydra override 문법 사용
- [ ] 환경 변수 및 경로 설정 확인

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

## 실험 표준 워크플로우

### 표준 실험 진행 절차
```
1. 실험 계획 → 2. Configuration 생성 → 3. 실험 폴더 구조 → 4. 훈련 실행 → 5. Validation → 6. 문서화
```

#### Phase 1: 실험 준비
```bash
# 1. 실험 폴더 생성 (표준 구조)
mkdir -p experiments/EXPERIMENT_NAME/{checkpoints,confusion_matrices,training_logs,validation_results}

# 2. Configuration 파일 생성
# - config/model/maxvit_yolox/MODEL_CONFIG.yaml
# - config/experiment/gen4/EXPERIMENT_CONFIG.yaml

# 3. 필수 문서 생성
# - experiment_hypothesis.txt
# - training_command.txt  
# - modification_details.txt
# - code_changes_summary.txt
```

#### Phase 2: 훈련 실행
```bash
# Screen session으로 훈련 시작
screen -dmS EXPERIMENT_NAME
screen -S EXPERIMENT_NAME -p 0 -X stuff "source /home/oeoiewt/miniconda3/etc/profile.d/conda.sh && conda activate rvt && cd /home/oeoiewt/eTraM/rvt_eTram\n"
screen -S EXPERIMENT_NAME -p 0 -X stuff "python train.py [TRAINING_PARAMETERS]\n"

# 진행 상황 모니터링
screen -list | grep EXPERIMENT_NAME
screen -S EXPERIMENT_NAME -X hardcopy /tmp/training_status.txt
```

#### Phase 3: Validation 실행 (핵심 개선사항)
```bash
# 1. Checkpoint 파일명 간단화 (특수문자 제거)
cp "/path/epoch=001-step=100000-val_AP=0.31.ckpt" "/path/final_model.ckpt"

# 2. 검증된 Validation 명령어 (오류 방지)
python validation.py \
  dataset=gen4 \
  model=maxvit_yolox/MODEL_CONFIG \
  checkpoint=/path/final_model.ckpt \
  use_test_set=false \
  hardware.gpus=0 \
  dataset.path=/home/oeoiewt/eTraM/rvt_eTram/data/etram_cls8_sample \
  +batch_size.train=6

# 3. Screen에서 안전한 실행 (선택사항)
screen -dmS validation_EXPERIMENT
screen -S validation_EXPERIMENT -p 0 -X stuff "source /home/oeoiewt/miniconda3/etc/profile.d/conda.sh && conda activate rvt && cd /home/oeoiewt/eTraM/rvt_eTram\n"
screen -S validation_EXPERIMENT -p 0 -X stuff "python validation.py [PARAMETERS]\n"
```

#### Phase 4: 결과 정리 및 문서화
```bash
# 1. Validation 결과 저장
mkdir -p validation_results/EXPERIMENT_NAME
# validation output → validation_results/EXPERIMENT_NAME/validation_output.log

# 2. Confusion Matrix 복사
cp confM/confusion_matrix_*.png experiments/EXPERIMENT_NAME/confusion_matrices/

# 3. 실험 결과 문서 생성
# - experiment_results.json
# - metrics_summary.txt
# - 상세 성능 분석 문서

# 4. Configuration 백업
cp config/model/maxvit_yolox/MODEL_CONFIG.yaml experiments/EXPERIMENT_NAME/model_config.yaml
cp config/experiment/gen4/EXPERIMENT_CONFIG.yaml experiments/EXPERIMENT_NAME/experiment_config.yaml
```

### ⚠️ Validation 실행 시 주의사항
1. **Checkpoint 파일명**: `=` 기호가 포함된 파일명은 Hydra parsing 오류 발생
2. **필수 파라미터**: dataset, model, checkpoint, dataset.path 반드시 포함
3. **Batch Size**: 기존 설정에 없는 경우 `+batch_size.train=6` 형식 사용
4. **Screen Session**: 장시간 실행 시 안전한 환경 제공

## Validation Memory Guidelines

### Object Detection Validation 지침

#### Validation 수행 시 확인 사항

- **Validation 목적**: COCO small/medium/large mAP와 클래스별 크기별 mAP 분석
- **클래스 크기 매핑**:
  - Small Objects:
    - Class 0: Pedestrian
    - Class 2: Bicycle
    - Class 7: Wheelchair
  - Medium Objects:
    - Class 4: Motorbike
  - Large Objects:
    - Class 1: Car
    - Class 3: Bus
    - Class 5: Truck
    - Class 6: Tram

#### Validation 실행 시 추가 체크포인트

- COCO 전체 mAP 확인
- 클래스별 mAP 분석
  - Small object mAP 특별 관찰
  - Large object mAP 비교
  - Medium object mAP 분석

#### 권장 Validation 세부 설정

```bash
python validation.py \
  dataset=gen4 \
  checkpoint=[CHECKPOINT] \
  +experiment/gen4=[CONFIG] \
  evaluation.coco_metrics=True \
  evaluation.class_wise_metrics=True \
  evaluation.object_size_metrics=True
```

### Validation 결과 문서화 템플릿

```markdown
## Validation Results: [실험명]

### Overall Performance
- Total mAP: X.XX%
- COCO mAP: 
  - Small Objects: X.XX%
  - Medium Objects: X.XX%
  - Large Objects: X.XX%

### Class-wise Performance
- Class 0 (Pedestrian): X.XX% 
- Class 1 (Car): X.XX%
...

### Size-aware Analysis
- Small Object Detection: 취약점 및 개선 방향
- Medium Object Detection: 성능 분석
- Large Object Detection: 강점 및 안정성
```

### Experimental Results Tracking

#### New Memory Entry: Experiment Results Summary Template
- **Memory Location**: In the `experiment_results.json` file

```json
{
    "Model structure": "...",
    "Loss function": "...", 
    "Total mAP": "...",
    "COCO small mAP": "...",
    "COCO medium mAP": "...", 
    "COCO large mAP": "...",
    "Small class mAP": "...",
    "Medium class mAP": "...",
    "Large class mAP": "..."
}
```

These guidelines ensure a standardized approach to experimental tracking, validation, and documentation in the eTraM project.

## 🏆 Completed Experiments Summary

### Overview
**Total Experiments**: 14 completed experiments (as of 2025-07-31)  
**Research Focus**: Event-based Traffic Monitoring with 8-class object detection  
**Dataset**: eTraM sample dataset (etram_cls8_sample)  
**Evaluation**: COCO-style mAP metrics with small/medium/large object analysis

### 📊 Performance Rankings

#### Overall mAP Rankings
1. **🥇 Plain LSTM + 4-scale FPN**: **34.6% mAP** (2h 24m training)
   - Best balanced performance across all object sizes
   - P1,P2,P3,P4 features for comprehensive small object coverage
   
2. **🥈 Plain LSTM + Size-aware Loss + 960×540**: **33.9% mAP** (3h 51m training)
   - **Best Small Objects AP: 18.9%** 
   - Optimal for small object detection applications
   
3. **🥉 Optimal Combination (4-scale + Size-aware + 960×540)**: **32.2% mAP** (2h 16m training)
   - Complex combination with non-additive synergy effects
   - Valuable lesson in component interaction limitations
   
4. **Plain LSTM + Size-aware Loss**: **32.9% mAP** (5h 53m training)
   - Strong small object improvement (+55% vs baseline)
   
5. **Plain LSTM Baseline**: **28.2% mAP** (6h training)
   - Successful RVT paper implementation
   - Foundation for all subsequent improvements

#### Small Objects Detection Rankings
1. **🎯 Size-aware + 960×540**: **18.9% AP** (베이스라인 10.2% 대비 +85% 향상)
2. **4-scale FPN**: **16.7% AP** (+64% 향상)
3. **Optimal Combination**: **17.4% AP** (+70% 향상)
4. **Size-aware 640×360**: **15.8% AP** (+55% 향상)

### 🔬 Experiment Categories

#### Architecture Experiments (6개)
- **Plain LSTM Baseline**: RVT 논문 성공적 재현 (28.2% mAP)
- **4-scale FPN**: 최고 전체 성능 달성 (34.6% mAP)
- **Size-aware Loss Variants**: Small object 특화 (32.9% → 33.9% mAP)
- **Lightweight Enhanced**: 복잡성 역설 입증 (20.9% mAP)
- **Resolution Scaling**: 960×540 해상도 효과 검증

#### Class Imbalance Experiments (2개)
- **CB01 (ETRAMClassBalancedLoss)**: 복합 기법 적용 (23.5% mAP, 1h 39m)
- **CB03 (Simple Class-Balanced)**: 1/frequency 가중치 (22.3% mAP, 5h 52m)
- **Key Finding**: CB03이 Small Objects에서 CB01 대비 43% 우수

### 💡 핵심 발견사항

#### 1. 복잡성 역설 (Complexity Paradox)
단순한 아키텍처가 복잡한 구조보다 우수한 경우가 빈번함을 입증:
```
Simple 4-scale FPN:     34.6% mAP  ← 최고 성능
Complex Combination:    32.2% mAP  ← 예상보다 낮음
Enhanced ConvLSTM:      20.9% mAP  ← 최저 성능
```

#### 2. 컴포넌트 시너지의 한계
개별 최적화 기법들의 조합이 선형적으로 합쳐지지 않음:
- **예상 시너지**: 40.3% mAP (이론적 계산)
- **실제 결과**: 32.2% mAP (시너지 효율 33%)

#### 3. Small Object Detection 전략
- **Size-aware Loss**: 가장 효과적인 단일 기법 (+55% 향상)
- **해상도 증가**: 추가적 이익 제공 (+19.6% 추가 향상)
- **4-scale FPN**: 균형잡힌 성능으로 모든 객체 크기에 도움

#### 4. 훈련 효율성 인사이트
- **가장 빠른 훈련**: CB01 (1h 39m, 23.5% mAP)
- **최고 성능 대비 시간**: 4-scale FPN (2h 24m, 34.6% mAP)
- **가장 긴 훈련**: 베이스라인 (6h, 28.2% mAP)

### 🎯 Production 추천사항

#### Use Case별 최적 모델
1. **균형잡힌 전체 성능**: Plain LSTM + 4-scale FPN (34.6% mAP)
2. **Small Objects 특화**: Plain LSTM + Size-aware + 960×540 (18.9% Small AP)
3. **빠른 훈련 필요**: CB01 Class-Balanced (1h 39m, 23.5% mAP)
4. **클래스 불균형 해결**: CB03 Simple Class-Balanced (22.3% mAP)

#### 개발 지침
- **단순성 우선**: 복잡한 조합보다 검증된 단일 기법 사용
- **데이터 중심 접근**: 아키텍처 복잡화보다 데이터 품질 개선
- **점진적 최적화**: 한 번에 하나씩 개선사항 적용 및 검증

### 📈 향후 연구 방향

#### 즉시 추진 가능한 방향
1. **4-scale FPN 극한 최적화**: 현재 최고 성능 모델의 하이퍼파라미터 튜닝
2. **1280×720 해상도 실험**: 더 높은 해상도에서의 성능 한계 탐색
3. **데이터 증강 기법**: 아키텍처 개선보다 데이터 품질 향상

#### 장기 연구 방향
1. **Sequential Optimization**: 단계별 최적화로 시너지 효과 극대화
2. **Adaptive Component Weighting**: 동적 가중치 조정 시스템
3. **Event-based 특화 아키텍처**: 기존 CNN 기반을 넘어선 새로운 접근

### 📚 실험 문서화 표준
모든 실험은 다음 구조로 완전히 문서화되어 있습니다:
```
experiments/[실험명]/
├── experiment_hypothesis.txt      # 가설 및 이론적 근거
├── experiment_results.json        # 정량적 성능 결과
├── comprehensive_analysis.md       # 상세 분석 보고서
├── training_command.txt           # 재현 가능한 명령어
├── checkpoints/final_model.ckpt   # 훈련된 모델
├── confusion_matrices/            # 성능 시각화
└── validation_results/            # 상세 검증 결과
```

### 🔬 실험 재현성
모든 실험은 다음을 통해 완전히 재현 가능합니다:
- **Configuration Files**: Hydra 기반 완전한 설정 관리
- **Training Commands**: Screen session과 함께 검증된 명령어
- **Environment**: `conda activate rvt` 환경에서 실행
- **Data Path**: `/home/oeoiewt/eTraM/rvt_eTram/data/etram_cls8_sample`

이 comprehensive experiment record는 Event-based Object Detection 분야에서 체계적인 연구 방법론과 재현 가능한 결과를 제공합니다.