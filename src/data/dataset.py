import torch
from torch.utils.data import Dataset
import os

class SudokuDataset(Dataset):
    """
    저장된 .pt 파일을 불러와서 학습에 사용하는 클래스
    """
    def __init__(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {path}")
        
        # 호환성을 위해 안전하게 로드
        try:
            # 최신 PyTorch 권장 방식
            data = torch.load(path, weights_only=False) 
        except:
            # 구버전 호환
            data = torch.load(path)
            
        self.problems = data['problems']
        self.solutions = data['solutions']
        
    def __len__(self):
        return len(self.problems)
    
    def __getitem__(self, idx):
        return self.problems[idx], self.solutions[idx]

def save_dataset(problems, solutions, path):
    """
    생성된 데이터를 .pt 파일로 저장하는 함수 (generate_data.py에서 사용)
    """
    # 저장할 폴더가 없으면 자동으로 생성
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        
    print(f"💾 데이터를 저장합니다: {path}")
    torch.save({
        'problems': problems,
        'solutions': solutions
    }, path)
    print("✅ 저장 완료!")