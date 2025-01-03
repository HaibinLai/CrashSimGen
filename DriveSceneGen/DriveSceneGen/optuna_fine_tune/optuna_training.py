import optuna
from DriveSceneGen.scripts.train import training_mine
import re

log_path = '/data/haibin/ML_DM/logs/training_logs.txt'

# 定义目标函数
def objective(trial):
    # 从 Optuna 提供的 trial 对象中采样参数
    NUM_EPOCHS = trial.suggest_int("NUM_EPOCHS", 3, 20) 
    TRAIN_BATCH_SIZE = trial.suggest_int("TRAIN_BATCH_SIZE", 32, 128)
    INFER_STEPS = 1000
    
    # 调用 training_mine 函数
    training_mine(NUM_EPOCHS=NUM_EPOCHS, INFER_STEPS=INFER_STEPS, TRAIN_BATCH_SIZE=TRAIN_BATCH_SIZE)  # 调用训练函数

    # 定义正则表达式，匹配 step、final_loss 和 total_epochs
    pattern = r"step (\d+): Final Loss: ([\d.]+)  Total Epochs: (\d+)"

    # 读取并解析日志文件
    with open(log_path, 'r') as file:
        logs = file.readlines()

    # 提取日志中的参数
    for line in logs:
        match = re.search(pattern, line)
        if match:
            global_step = int(match.group(1))  # 提取 step
            final_loss = float(match.group(2))  # 提取 final_loss
            total_epochs = int(match.group(3))  # 提取 total_epochs
            
            # 打印提取的参数
            print(f"Global Step: {global_step}, Final Loss: {final_loss}, Total Epochs: {total_epochs}")

    
    # 返回损失或目标指标
    return final_loss

# 定义 Optuna 优化
if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize",  # 假设目标是最小化损失
        storage="sqlite:///db.sqlite3",  # 存储优化的结果
        study_name="training-DM004"  # 研究名称
    )
    study.optimize(objective, n_trials=50)  # 优化 50 次

    # 打印出最好的结果和参数
    print(f"Best value: {study.best_value} (params: {study.best_params})")
