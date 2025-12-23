# src/config.py
import torch

class Config:
    # 1. 스도쿠 규격
    GRID_SIZE = 9
    BOX_H = 3
    BOX_W = 3
    
    SEQ_LEN = GRID_SIZE * GRID_SIZE
    NUM_CLASSES = GRID_SIZE + 1
    
    # 2. 데이터셋 (기본 범위 - 참고용)
    MIN_HOLES = 24
    MAX_HOLES = 60
    
    TRAIN_SIZE = 300000 
    VAL_SIZE = 10000    # 검증용 1만 개 (충분함)

    # ==========================================
    # 🆕 3. 커리큘럼 데이터 생성 설정 (Curriculum)
    # ==========================================
    # 여기서 비율과 난이도를 관리합니다. (수정이 필요하면 여기만 고치면 됩니다)
    CURRICULUM = {
        'train': [
            {'label': 'Medium', 'count': 100000, 'min': 24, 'max': 39}, # 기초 10만
            {'label': 'Expert', 'count': 200000, 'min': 40, 'max': 60}, # 심화 20만
        ],
        'val': [
            {'label': 'Medium', 'count': 3000,  'min': 24, 'max': 39},
            {'label': 'Expert', 'count': 7000,  'min': 40, 'max': 60},
        ]
    }

    # ==========================================
    # 🆕 4. 인퍼런스(테스트) 전용 설정
    # ==========================================
    # 실전 모의고사는 "가장 어려운 난이도"로 고정합니다.
    TEST_MIN_HOLES = 40
    TEST_MAX_HOLES = 60
    TEST_SIZE = 100

    # 5. 모델 설정 (8층 가성비 모델)
    D_MODEL = 512       
    NHEAD = 8           
    NUM_LAYERS = 8      
    
    DROPOUT = 0.1
    ACTIVATION = "gelu" 

    # 6. 학습 설정
    BATCH_SIZE = 256    
    LR = 0.0005
    LR_MIN = 0.00005
    EPOCHS = 30         
    
    # 7. 경로 설정
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_DIR = "data/processed"
    MODEL_SAVE_DIR = "saved_models"
    MODEL_PATH = f"{MODEL_SAVE_DIR}/best_model.pth"
    CHECKPOINT_PATH = f"{MODEL_SAVE_DIR}/last_checkpoint.pth"