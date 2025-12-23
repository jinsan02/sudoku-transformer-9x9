# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config
from src.data.dataset import SudokuDataset
from src.model.transformer import SudokuTransformer
from src.utils import calculate_accuracy, seed_everything

def main():
    # 1. 초기 설정
    seed_everything(42)
    print(f"🔧 학습 장치: {Config.DEVICE} (Curriculum & Strict Mode)")
    print(f"   - 모델 스펙: d_model={Config.D_MODEL}, layers={Config.NUM_LAYERS}, head={Config.NHEAD}")
    print(f"   - 데이터셋: {Config.TRAIN_SIZE}개 (학습), {Config.VAL_SIZE}개 (검증)")

    if not os.path.exists(Config.MODEL_SAVE_DIR):
        os.makedirs(Config.MODEL_SAVE_DIR)
    
    # 2. 데이터 로더 (Config 경로 사용)
    train_loader = DataLoader(
        SudokuDataset(f"{Config.DATA_DIR}/train.pt"), 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True,
        num_workers=4,        
        pin_memory=True,      
        persistent_workers=True 
    )
    
    val_loader = DataLoader(
        SudokuDataset(f"{Config.DATA_DIR}/val.pt"), 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    # 3. 모델 및 최적화 도구 설정
    model = SudokuTransformer(Config).to(Config.DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR)
    criterion = nn.CrossEntropyLoss() 
    
    # [핵심 수정] 스케줄러 설정 (LR_MIN 적용)
    warmup_epochs = 3
    
    # Phase 1: 웜업 (0 -> 0.0005)
    scheduler1 = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    
    # Phase 2: 코사인 감소 (0.0005 -> Config.LR_MIN)
    # eta_min을 설정하여 학습률이 0으로 죽지 않고 끝까지 유지되게 함
    scheduler2 = CosineAnnealingLR(
        optimizer, 
        T_max=Config.EPOCHS - warmup_epochs, 
        eta_min=Config.LR_MIN 
    )
    
    scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_epochs])
    
    # 4. 체크포인트 로드 (이어하기)
    start_epoch = 0
    best_acc = 0.0
    
    if os.path.exists(Config.CHECKPOINT_PATH):
        print(f"🔄 체크포인트 발견! 학습을 재개합니다: {Config.CHECKPOINT_PATH}")
        try:
            ckpt = torch.load(Config.CHECKPOINT_PATH, map_location=Config.DEVICE, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_acc = ckpt.get('best_acc', 0.0)
            print(f"   ▶ Epoch {start_epoch+1}부터 시작 (현재 최고 기록: {best_acc*100:.2f}%)")
        except Exception as e:
            print(f"⚠️ 체크포인트 로드 실패 ({e}). 처음부터 다시 시작합니다.")
    else:
        print("✨ 새로운 학습을 시작합니다.")

    # 5. 학습 루프
    for epoch in range(start_epoch, Config.EPOCHS):
        model.train()
        train_loss = 0
        train_acc = 0
        
        # 진행률 표시줄 (TQDM)
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        
        for p, s in loop:
            p, s = p.to(Config.DEVICE, non_blocking=True), s.to(Config.DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            out = model(p)
            
            # (Batch*Seq, Classes) 형태로 변환 후 Loss 계산
            loss = criterion(out.view(-1, Config.NUM_CLASSES), s.view(-1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += calculate_accuracy(out, s)
            
            loop.set_postfix(loss=f"{loss.item():.4f}")

        # 에폭 끝날 때마다 스케줄러 갱신
        scheduler.step()
        
        # 6. 검증 (Validation)
        model.eval()
        val_acc = 0
        with torch.no_grad():
            for p, s in val_loader:
                p, s = p.to(Config.DEVICE), s.to(Config.DEVICE)
                val_acc += calculate_accuracy(model(p), s)
        
        avg_val_acc = val_acc / len(val_loader)
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"   Done! Val Acc: {avg_val_acc*100:.2f}% | LR: {current_lr:.6f}")
        
        # 7. 저장 (체크포인트 & 베스트 모델)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc
        }, Config.CHECKPOINT_PATH)

        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"   🏆 최고 기록 경신! 모델 저장됨: {Config.MODEL_PATH}")

if __name__ == "__main__":
    main()