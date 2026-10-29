from train_stage1 import train_stage1
from train_stage2 import train_stage2

if __name__ == '__main__':
    print("=== Starting Stage1 Training (Image Multi-Task) ===")
    train_stage1()
    print("=== Starting Stage2 Training (Dual-Mamba Multimodal) ===")
    train_stage2()
    print("=== Full Pipeline Finished ===")