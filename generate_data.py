# generate_data.py
import multiprocessing
import numpy as np
import os
import time
from tqdm import tqdm
from src.config import Config
from src.data.generator import SudokuGenerator
from src.data.dataset import save_dataset

def generate_chunk(args):
    """
    개별 프로세스 작업 함수
    args: (count, min_holes, max_holes)
    """
    count, min_h, max_h = args
    gen = SudokuGenerator()
    problems, solutions = gen.generate_dataset(count, min_h, max_h)
    return problems, solutions

def run_mixed_generation(target_config, output_filename, mode_name):
    """
    Config에 정의된 커리큘럼대로 데이터를 생성하고 섞습니다.
    """
    print(f"\n🚀 [{mode_name}] 데이터 생성을 시작합니다 (Curriculum Mode)")
    
    start_time = time.time()
    num_workers = max(1, multiprocessing.cpu_count() - 2) # 여유 코어 2개
    
    all_problems = []
    all_solutions = []
    
    # Config에 있는 단계별(Medium/Expert) 생성
    for phase in target_config:
        total_count = phase['count']
        min_h, max_h = phase['min'], phase['max']
        label = phase['label']
        
        print(f"   👉 Phase: {label} (빈칸 {min_h}~{max_h}) -> {total_count}개 생성 중...")
        
        # 작업 분배
        chunk_size = total_count // num_workers
        remainder = total_count % num_workers
        tasks = []
        for i in range(num_workers):
            c = chunk_size + (1 if i < remainder else 0)
            if c > 0:
                tasks.append((c, min_h, max_h))
        
        # 병렬 처리
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(generate_chunk, tasks), total=len(tasks), desc=f"      Creating {label}"))
            
        # 결과 모으기
        phase_probs = np.vstack([r[0] for r in results])
        phase_sols = np.vstack([r[1] for r in results])
        
        all_problems.append(phase_probs)
        all_solutions.append(phase_sols)

    # 1. 전체 병합
    final_problems = np.vstack(all_problems)
    final_solutions = np.vstack(all_solutions)
    
    print(f"   🎲 데이터 섞는 중 (Shuffling)...")
    # 2. 셔플 (기초와 심화를 골고루 섞음)
    indices = np.arange(len(final_problems))
    np.random.shuffle(indices)
    
    final_problems = final_problems[indices]
    final_solutions = final_solutions[indices]

    # 3. 저장
    save_path = os.path.join(Config.DATA_DIR, output_filename)
    save_dataset(final_problems, final_solutions, save_path)
    
    elapsed = time.time() - start_time
    print(f"✨ [{mode_name}] 완료! 총 {len(final_problems)}개 ({elapsed:.1f}초) -> {save_path}")

def main():
    multiprocessing.freeze_support()
    os.makedirs(Config.DATA_DIR, exist_ok=True)

    print("="*60)
    print(f"🧩 스도쿠 커리큘럼 데이터 생성기 (Config 기반)")
    print("="*60)

    # 1. 학습용 (Train) - Config.CURRICULUM 사용
    run_mixed_generation(Config.CURRICULUM['train'], "train.pt", "Train Set")
    
    # 2. 검증용 (Val) - Config.CURRICULUM 사용
    run_mixed_generation(Config.CURRICULUM['val'], "val.pt", "Validation Set")
    
    print("\n🎉 모든 데이터 준비 완료! 이제 학습하면 지능이 더 좋아집니다.")

if __name__ == "__main__":
    main()