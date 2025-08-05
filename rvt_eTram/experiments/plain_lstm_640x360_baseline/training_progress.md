# Plain LSTM 640x360 Baseline Experiment - Training Progress

## 실험 개요
- **실험명**: plain_lstm_640x360_baseline
- **목표**: RVT 논문의 Plain LSTM 구현으로 +1.1% mAP 개선 달성
- **시작 시간**: 2025-07-24 16:23 KST
- **예상 소요시간**: ~5-6시간 (100K steps)

## 아키텍처 변경사항
- **기존**: DWSConvLSTM2d (3x3 depthwise-separable convolution)
- **신규**: PlainLSTM2d (1x1 standard convolution)
- **이론적 근거**: RVT 논문에서 Plain LSTM이 ConvLSTM 대비 1.1% mAP 향상 및 50% 파라미터 감소

## 훈련 설정
```yaml
model: maxvit_yolox/plain_lstm
dataset: gen4 (etram_cls8_sample)
training:
  max_steps: 100000
  batch_size: 6 (train), 2 (eval)
  sampling: stream
hardware:
  gpu: 0
  workers: 4 (train), 3 (eval)
```

## 실시간 훈련 진행 상황

### 초기 단계 (0-1분)
- **Step 0-17**: Loss 56.1 → 훈련 시작
- **속도**: 1.45it/s (초기 단계)
- **상태**: Sanity check 완료, 정상 시작

### 안정화 단계 (1-2분)
- **Step 17-330**: Loss 56.1 → 10.1 (크게 개선)
- **속도**: 4.27it/s (안정적 속도)
- **상태**: ✅ 정상적인 수렴 패턴

## 예상 결과
- **Overall mAP**: 35.1% (34.02% baseline + 1.1% improvement)
- **Small Objects mAP**: 18.5% (17.28% baseline + proportional improvement)
- **Parameter Efficiency**: ~50% reduction vs ConvLSTM

## 모니터링 포인트
1. **Loss 수렴**: 정상적으로 감소 중 ✅
2. **훈련 속도**: 4.27it/s로 양호 ✅
3. **Memory Usage**: OOM 없이 안정적 ✅
4. **WandB Logging**: etram_enhanced 프로젝트에 기록 중

## 다음 단계 준비
- Phase 2.1: Progressive Training 구현 준비
- Phase 2.2: Memory Optimization 계획
- Phase 2.3: 1280x720 고해상도 실험 준비

**Status**: 🟢 훈련 정상 진행 중 (ETA: ~4-5시간 남음)