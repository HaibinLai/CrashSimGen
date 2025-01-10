# import optuna
from DriveSceneGen.scripts.train import training_mine
# import re

# log_path = '/data/haibin/ML_DM/logs/training_logs.txt'


# 定义 Optuna 优化
if __name__ == "__main__":
    # 从 Optuna 提供的 trial 对象中采样参数
    NUM_EPOCHS = 10
    TRAIN_BATCH_SIZE = 32
    Learning_rate = 0.0001
    INFER_STEPS = 1000
    
    # 调用 training_mine 函数
    training_mine(NUM_EPOCHS=NUM_EPOCHS, INFER_STEPS=INFER_STEPS, TRAIN_BATCH_SIZE=TRAIN_BATCH_SIZE, LEARNING_RATE=Learning_rate)  # 调用训练函数
