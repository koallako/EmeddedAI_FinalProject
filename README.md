# Age Prediction Project with Jetson Nano

이 프로젝트는 Jetson Nano를 사용하여 실시간으로 사람의 나잇대를 예측하는 경량화된 AI 모델을 구현합니다.

## 프로젝트 개요

- CSI 카메라를 통해 실시간으로 사람 얼굴을 인식
- UTKFace 데이터셋 기반 나이 예측 모델 구현
- Knowledge Distillation과 Pruning을 통한 모델 경량화
- Jetson Nano에 최적화된 추론 구현

## 핵심 기능

1. 실시간 얼굴 인식 및 나이 예측
2. 모델 경량화 (Knowledge Distillation & Pruning)
3. 리소스 제약 환경에서의 효율적인 추론
4. 실시간 성능 모니터링 (FPS, 처리 시간 등)

## 프로젝트 구조

```
.
├── agemodel.py           # 모델 아키텍처 및 pruning 구현
├── agetrain.ipynb        # 모델 학습 및 경량화 프로세스
├── jetson_infer.ipynb    # Jetson Nano 실시간 추론
└── README.md             # 프로젝트 설명서
```

## 설치 방법

1. 필요 패키지 설치:
```bash
pip install torch torchvision opencv-python numpy pandas tqdm matplotlib
```

2. Jetson Nano 설정:
   - JetPack 설치
   - CSI 카메라 연결 및 설정

## 사용된 기술

### 1. 모델 경량화 기법
- **Knowledge Distillation**
  - Teacher 모델: 32-64-128 채널 구조
  - Student 모델: 16-32-64 채널 구조 (파라미터 51% 감소)
  - Temperature=3.0, alpha=0.5 설정

- **Pruning**
  - L1-norm 기반 Unstructured pruning
  - Convolution 레이어 대상
  - 30%, 50%, 70% 비율로 실험

### 2. 최적화 결과
- Teacher 모델 대비 Student 모델:
  - 파라미터 수: 51% 감소
  - FLOPs: 72% 감소
  - 성능 저하: MAE 기준 약 5%

## 데이터셋 준비

1. UTKFace 데이터셋 다운로드:
   - [UTKFace 데이터셋](https://susanqq.github.io/UTKFace/) 다운로드
   - 이미지 형식: [age]_[gender]_[race]_[date&time].jpg
   - 연령대별 데이터 분포가 균일하도록 **전처리** 필요

2. 데이터 구조:
```
agedata/
└── 2400_dataset/
    ├── 0_0_0_20170109150557335.jpg
    ├── 1_0_2_20170109150557358.jpg
    └── ...
```

3. 데이터 전처리:
   - 각 연령대별 300개의 데이터로 균일화
   - 이미지 크기: 64x64 RGB
   - 총 3,000개의 학습 데이터

## 실행 방법

1. 모델 학습:
```python
# 데이터셋 경로 설정
DATA_PATH = '/path/to/your/agedata/3000_dataset'

# 학습 실행
jupyter notebook agetrain.ipynb
```

2. Jetson Nano에서 실시간 추론:
```python
jupyter notebook jetson_infer.ipynb
```

## 성능 지표

- MAE (Mean Absolute Error)
- 10년/15년 이내 정확도
- FPS (Frames Per Second)
- 추론 시간
- 메모리 사용량

