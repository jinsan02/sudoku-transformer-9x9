# src/utils.py
import torch
import numpy as np
import random
import os

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_accuracy(outputs, targets):
    """
    학습(Train) 중에는 빠른 속도를 위해 단순 비교 방식을 유지합니다.
    """
    predictions = torch.argmax(outputs, dim=-1)
    if targets.dim() == 3: targets = targets.view(targets.size(0), -1)
    correct = (predictions == targets).sum().item()
    total = targets.numel()
    return correct / total

def check_sudoku_validity(grids):
    """
    [NEW] AI가 푼 답안(grids)이 스도쿠 논리 규칙을 완벽하게 지켰는지 검사합니다.
    정답지와 달라도 규칙을 지켰으면 '정답'으로 인정합니다.
    
    Args:
        grids: (Batch, 9, 9) 형태의 텐서 또는 넘파이 배열 (1~9 숫자로 채워짐)
    Returns:
        int: 규칙을 통과한 정답 개수
    """
    if isinstance(grids, torch.Tensor):
        grids = grids.cpu().detach().numpy()
    
    batch_size = grids.shape[0]
    valid_count = 0
    
    for i in range(batch_size):
        grid = grids[i]
        
        # 0. 기본 검사 (1~9 숫자만 있어야 함, 빈칸 0이 없어야 함)
        if not np.all((grid >= 1) & (grid <= 9)):
            continue 

        is_valid = True
        
        # 1. 행(Row) 검사: 중복 없어야 함
        for r in range(9):
            if len(np.unique(grid[r, :])) != 9:
                is_valid = False; break
        if not is_valid: continue

        # 2. 열(Col) 검사: 중복 없어야 함
        for c in range(9):
            if len(np.unique(grid[:, c])) != 9:
                is_valid = False; break
        if not is_valid: continue

        # 3. 박스(Box) 검사: 중복 없어야 함
        for r in range(0, 9, 3):
            for c in range(0, 9, 3):
                box = grid[r:r+3, c:c+3].flatten()
                if len(np.unique(box)) != 9:
                    is_valid = False; break
        if not is_valid: continue

        # 모든 테스트 통과!
        valid_count += 1
        
    return valid_count