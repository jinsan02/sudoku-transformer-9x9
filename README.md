# 🧩 9x9 Sudoku Solver with Transformer (Expert Baseline)

Transformer 아키텍처를 활용하여 **Expert(고난이도)** 수준의 9x9 스도쿠를 해결하는 AI 프로젝트입니다.
단순 패턴 인식을 넘어, **MRV(Minimum Remaining Values)** 기반의 데이터 생성과 **구조적 임베딩(Structural Embeddings)**을 통해 스도쿠의 복잡한 제약 조건을 학습했습니다.

현재 **8-Layer Baseline 모델**로 Expert 난이도에서 유의미한 추론 능력을 확보했으며(약 63%), 추후 모델 확장을 위한 유연한 코드 구조로 설계되었습니다.

---

## 📌 주요 특징 (Key Features)

### 1. 구조적 임베딩 & Attention (Structural Awareness)
* 기존의 단순 1D 위치 인코딩 대신, 9x9 격자의 **행(Row), 열(Col), 3x3 박스(Box)** 정보를 결합한 독자적인 임베딩을 사용합니다.
* 이를 통해 모델이 "같은 행에는 숫자가 중복될 수 없다"는 스도쿠의 불변 규칙을 명확하게 이해합니다.

### 2. MRV 기반 고난이도 데이터 생성 (Advanced Data Generation)
* 무작위 빈칸 뚫기가 아닌, **MRV(Minimum Remaining Values)** 알고리즘과 **백트래킹(Backtracking)**을 결합하여 인간 전문가 수준의 논리가 필요한 데이터를 생성합니다.
* **'유일한 해(Unique Solution)'**가 보장된 문제만 엄선하여 학습 데이터의 품질을 극대화했습니다.

### 3. 커리큘럼 학습 & 검증 (Curriculum Learning)
* `Medium` → `Expert` 순으로 난이도별 데이터 분포를 조절하여 학습 효율을 높였습니다.
* 추론 시 정답지와 단순히 비교하는 것을 넘어, **AI가 채운 답안이 스도쿠 규칙을 논리적으로 만족하는지(Validity Check)** 검사하는 로직을 포함합니다.

### 4. 중앙 집중식 설정 관리 (Centralized Config)
* `src/config.py` 파일 하나로 모델 크기(Layer, Hidden dim), 학습 파라미터, 데이터 경로 등을 통합 관리합니다.
* 모델의 깊이(Depth)나 너비(Width)를 `config` 수정만으로 즉시 변경할 수 있어 실험 용이성이 뛰어납니다.

---

## 📂 디렉토리 구조 (Structure)

```text
sudoku-transformer-9x9/
├── data/                        # 데이터 저장소 (Git 제외)
│   └── processed/               # 생성된 학습용 데이터 (.pt)
├── saved_models/                # 학습된 모델 가중치 (.pth, Git 제외)
├── src/
│   ├── data/
│   │   ├── generator.py         # MRV 기반 스도쿠 생성 및 검증 로직
│   │   └── dataset.py           # PyTorch Dataset 클래스
│   ├── model/
│   │   └── transformer.py       # Config 기반 Transformer 아키텍처
│   ├── config.py                # [핵심] 하이퍼파라미터 통합 설정 파일
│   └── utils.py                 # 시드 고정, 유효성 검사 등 유틸리티
├── generate_data.py             # [1단계] 데이터 생성 스크립트
├── train.py                     # [2단계] 모델 학습 스크립트
├── inference.py                 # [3단계] 추론 및 성능 테스트 스크립트
├── requirements.txt             # 의존성 목록
└── README.md                    # 프로젝트 문서

```

---

## 🚀 설치 및 실행 순서 (Installation & Usage)

이 프로젝트는 Python 3.10 및 PyTorch(CUDA 권장) 환경에서 테스트되었습니다.

### 1. 환경 설정 (Installation)

```bash
# 레포지토리 클론
git clone [https://github.com/your-username/sudoku-transformer-9x9.git](https://github.com/your-username/sudoku-transformer-9x9.git)
cd sudoku-transformer-9x9

# 필수 라이브러리 설치
pip install -r requirements.txt

```

### 2. 데이터 생성 (Data Generation)

MRV 알고리즘을 사용하여 고품질 데이터를 생성합니다. (기본 설정: 30만 개)

```bash
python generate_data.py

```

* **결과:** `data/processed/` 폴더에 `train.pt`, `val.pt`가 생성됩니다.
* **특징:** Expert 난이도 비율이 높게 설정되어 있습니다.

### 3. 모델 학습 (Training)

```bash
python train.py

```

* **결과:** `saved_models/best_model.pth`에 최적의 가중치가 저장됩니다.
* **로그:** 학습 진행률과 검증 정확도(Val Acc)가 실시간으로 표시됩니다.

### 4. 테스트 및 추론 (Inference)

AI의 문제 풀이 능력을 검증합니다. (기본값: Expert 난이도 100문제 테스트)

```bash
python inference.py

```

* 단순 정답 비교가 아닌, **스도쿠 규칙 유효성(Validity)**을 기준으로 채점합니다.
* 실제 스도쿠가 풀리는 과정을 시각적으로 확인할 수 있습니다.

---

## 🧠 모델 아키텍처 정보 (Baseline Architecture)

| 항목 | 설정값 (Baseline) | 설명 |
| --- | --- | --- |
| **Grid Size** | 9x9 | Standard Sudoku |
| **Embedding** | 512 dim | Token + (Row + Col + Box) Embeddings |
| **Layers** | 8 | Transformer Encoder Blocks |
| **Heads** | 8 | Multi-head Attention |
| **Algorithm** | MRV | Minimum Remaining Values for Data Gen |
| **Performance** | ~63% (Expert) | Zero-shot Reasoning (No backtracking) |

> **Note:** 현재 버전(v1.0)은 8-layer Baseline 모델입니다. Expert 난이도의 완전한 해결(99%+)을 위해서는 모델의 깊이(Layers)와 파라미터(Hidden Dim)를 확장하는 Scale-Up이 권장됩니다.

---

## 🏆 Performance History (Benchmarks)

이 프로젝트는 다양한 모델 크기와 난이도 설정에서 테스트되었습니다.
이전 실험에서 **Medium 난이도(빈칸 24~48개)** 기준 **99% 이상의 정확도**를 달성하여 아키텍처의 우수성을 입증했습니다. 현재는 가장 어려운 **Expert 난이도** 정복을 목표로 최적화를 진행 중입니다.

| Model Version | Parameters | Difficulty (Holes) | Accuracy | Note |
| --- | --- | --- | --- | --- |
| **Current Baseline** | ~14M (8-Layer) | **Expert (40-64)** | ~63.0% | 🚧 **Work in Progress** (Hardest Task) |
| **Legacy Large** | 38M | Medium (24-48) | **99.41%** | `best_model_38M_H24-48` |
| **Legacy Base** | 25M | Medium (24-48) | **99.12%** | `best_model_25M_H24-48` |

> **Insight:** 빈칸이 적절한(Medium) 수준에서는 Transformer가 스도쿠 패턴을 완벽하게 학습(99%+)함을 확인했습니다. 현재 Expert 단계에서의 성능 저하는 추론 깊이(Reasoning Depth)의 부족 때문으로 분석되며, 모델 Scale-Up을 통해 해결할 예정입니다.

---

## 📝 License

This project is licensed under the MIT License.

```

```
