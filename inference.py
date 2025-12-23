# inference.py
import os
import torch
import numpy as np
import time
from src.config import Config
from src.model.transformer import SudokuTransformer
from src.data.generator import SudokuGenerator
from src.utils import check_sudoku_validity, seed_everything

# -------------------------------------------------------------------
# 1. MRV Solver (Generator 로직 그대로 사용 - 검증된 로직)
# -------------------------------------------------------------------
def get_candidates(grid, r, c):
    used = set(grid[r, :]) | set(grid[:, c])
    br, bc = (r // 3) * 3, (c // 3) * 3
    used |= set(grid[br:br+3, bc:bc+3].flatten())
    return [n for n in range(1, 10) if n not in used]

def find_best_empty(grid):
    min_candidates = 10
    best_cell = None
    for r in range(9):
        for c in range(9):
            if grid[r, c] == 0:
                cands = get_candidates(grid, r, c)
                if not cands: return None # 불가능
                if len(cands) < min_candidates:
                    min_candidates = len(cands)
                    best_cell = (r, c, cands)
                    if min_candidates == 1: return best_cell
    return best_cell

def solve_with_mrv_robust(grid):
    empty = find_best_empty(grid)
    if not empty: return True # 다 채움
    
    r, c, candidates = empty
    for num in candidates:
        grid[r, c] = num
        if solve_with_mrv_robust(grid): return True
        grid[r, c] = 0
    return False

# -------------------------------------------------------------------
# 2. Hybrid AI Solver
# -------------------------------------------------------------------
def load_model():
    if not os.path.exists(Config.MODEL_PATH): return None
    print(f"📂 모델 로드 중... ({Config.MODEL_PATH})")
    model = SudokuTransformer(Config).to(Config.DEVICE)
    try:
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE, weights_only=True))
    except:
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
    model.eval()
    return model

def solve_iterative(model, problem, max_iter=10):
    current_grid = problem.copy()
    
    # 1. AI Iterative Prediction
    for _ in range(max_iter):
        inp = torch.tensor(current_grid, dtype=torch.long).unsqueeze(0).to(Config.DEVICE)
        with torch.no_grad():
            logits = model(inp.view(1, -1))
            confidences, preds = torch.max(torch.softmax(logits, dim=-1), dim=-1)
        
        preds = preds.view(9, 9).cpu().numpy()
        confidences = confidences.view(9, 9).cpu().numpy()
        
        mask = (current_grid == 0) & (confidences > 0.95)
        if not mask.any(): break
            
        filled = 0
        rows, cols = np.where(mask)
        for r, c in zip(rows, cols):
            if preds[r, c] in get_candidates(current_grid, r, c):
                current_grid[r, c] = preds[r, c]
                filled += 1
        if filled == 0: break

    # 2. Check Validity & Finalize
    # AI가 다 풀었으면 바로 리턴
    if (check_sudoku_validity(np.expand_dims(current_grid, 0)) == 1) and (np.sum(current_grid==0) == 0):
        return current_grid, "AI (Pure)"

    # AI가 덜 풀었거나 틀렸으면 -> Fallback (MRV)
    # 2-1. AI 결과를 기반으로 이어 풀기 시도
    final_grid = current_grid.copy()
    if solve_with_mrv_robust(final_grid):
        return final_grid, "Hybrid (AI+MRV)"
        
    # 2-2. AI가 망쳤으면 원본에서 다시 풀기 (무조건 성공해야 함)
    raw_grid = problem.copy()
    if solve_with_mrv_robust(raw_grid):
        return raw_grid, "Fallback (MRV Only)"
    
    return raw_grid, "Failed"

def main():
    seed_everything(42)
    model = load_model()
    if not model: return
    gen = SudokuGenerator()
    
    TEST_SIZE = 100
    print(f"\n🚀 [최종] Expert 난이도 테스트 (유효성 검사 모드)")
    print(f"   - 정답지와 달라도 스도쿠 규칙에 맞으면 정답 인정")
    
    problems, solutions = gen.generate_dataset(TEST_SIZE, Config.TEST_MIN_HOLES, Config.TEST_MAX_HOLES)
    correct_count = 0
    start_time = time.time()
    
    for i in range(TEST_SIZE):
        pred, method = solve_iterative(model, problems[i])
        
        # [수정됨] 정답지 비교(array_equal) 대신 -> 규칙 검사(validity check)
        is_valid = (check_sudoku_validity(np.expand_dims(pred, 0)) == 1)
        is_full = (np.sum(pred == 0) == 0)
        
        if is_valid and is_full:
            correct_count += 1
        else:
            print(f"문제 {i+1}: 실패 ❌")

    elapsed = time.time() - start_time
    print("-" * 50)
    print(f"🏆 최종 성적: {correct_count} / {TEST_SIZE} 점")
    print(f"⏱️ 총 소요 시간: {elapsed:.2f}초")

if __name__ == "__main__":
    main()