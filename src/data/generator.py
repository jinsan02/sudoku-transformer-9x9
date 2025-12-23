# src/data/generator.py
import numpy as np
import random
from src.config import Config

class SudokuGenerator:
    def __init__(self):
        self.rows = Config.GRID_SIZE
        self.cols = Config.GRID_SIZE

    def generate_dataset(self, num_samples, min_holes, max_holes):
        problems = []
        solutions = []
        
        print(f"⚡ [최적화 모드] 고난이도 데이터 {num_samples}개 생성 시작 (MRV 적용)...")
        
        count = 0
        while count < num_samples:
            solution = self._generate_full_board()
            
            # 구멍 뚫기
            target_holes = random.randint(min_holes, max_holes)
            problem = self._remove_numbers_unique(solution, target_holes)
            
            problems.append(problem)
            solutions.append(solution)
            
            count += 1
            if count % 1000 == 0:
                print(f"   🚀 {count}/{num_samples} 완료")
                
        return np.array(problems), np.array(solutions)

    def _generate_full_board(self):
        grid = np.zeros((9, 9), dtype=int)
        self._solve_mrv(grid) # MRV로 빠르게 채우기
        return grid

    def _remove_numbers_unique(self, grid, target_holes):
        problem = grid.copy()
        coords = [(r, c) for r in range(9) for c in range(9)]
        random.shuffle(coords)
        
        holes_made = 0
        for r, c in coords:
            if holes_made >= target_holes:
                break
            
            original_val = problem[r, c]
            problem[r, c] = 0
            
            # [핵심] 해가 2개 이상인지 검사 (MRV 적용으로 초고속)
            # limit=2: 해가 2개 발견되면 즉시 중단
            if self._count_solutions_mrv(problem, limit=2) != 1:
                problem[r, c] = original_val # 복구
            else:
                holes_made += 1
        return problem

    # =========================================================
    # 🧠 핵심 알고리즘: MRV (Minimum Remaining Values)
    # 빈칸 중 '가능한 숫자가 가장 적은 칸'을 먼저 찾습니다.
    # =========================================================

    def _solve_mrv(self, grid):
        """해를 1개만 찾으면 True 반환 (보드 생성용)"""
        empty_pos = self._find_best_empty(grid)
        if not empty_pos:
            return True # 다 채움
        
        r, c, candidates = empty_pos
        random.shuffle(candidates) # 무작위성 부여
        
        for num in candidates:
            grid[r, c] = num
            if self._solve_mrv(grid):
                return True
            grid[r, c] = 0
        return False

    def _count_solutions_mrv(self, grid, limit=2):
        """해의 개수를 셉니다 (limit 도달 시 중단)"""
        empty_pos = self._find_best_empty(grid)
        if not empty_pos:
            return 1 # 해 1개 발견
        
        r, c, candidates = empty_pos
        count = 0
        
        for num in candidates:
            grid[r, c] = num
            count += self._count_solutions_mrv(grid, limit)
            grid[r, c] = 0
            
            if count >= limit: # 더 셀 필요 없음
                return count
        return count

    def _find_best_empty(self, grid):
        """
        [MRV] 모든 빈칸을 검사해서, 들어갈 수 있는 숫자가 가장 적은 칸을 반환
        Returns: (r, c, [가능한 숫자 리스트])
        """
        min_candidates = 10 # 9보다 큰 수로 초기화
        best_cell = None
        
        for r in range(9):
            for c in range(9):
                if grid[r, c] == 0:
                    candidates = self._get_candidates(grid, r, c)
                    num_candidates = len(candidates)
                    
                    if num_candidates == 0:
                        # 불가능한 칸이 있으면 즉시 실패 처리 (가지치기)
                        return None 
                    
                    if num_candidates < min_candidates:
                        min_candidates = num_candidates
                        best_cell = (r, c, candidates)
                        if min_candidates == 1:
                            return best_cell # 1개면 더 볼 것도 없이 이걸로 결정
                            
        return best_cell

    def _get_candidates(self, grid, r, c):
        """해당 칸(r,c)에 들어갈 수 있는 유효한 숫자들을 구함"""
        used = set(grid[r, :]) | set(grid[:, c])
        
        br, bc = (r // 3) * 3, (c // 3) * 3
        used |= set(grid[br:br+3, bc:bc+3].flatten())
        
        return [n for n in range(1, 10) if n not in used]