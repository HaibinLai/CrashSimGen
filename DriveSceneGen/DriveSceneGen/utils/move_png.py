import os
import shutil

# 文件所在的目录（根据实际路径修改）
source_directory = "/data/haibin/ML_DM/rasterized_training_20s/1_1_new/"
destination_directory = "/data/haibin/ML_DM/rasterized_training_20s/1_2_new/"

# 确保目标文件夹存在
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

# 获取源文件夹中的所有文件
files = os.listdir(source_directory)

# 筛选出以 "1" 开头的文件
# for file in files:
#     print(file.lower())
files_to_move = [f for f in files if f.lower().startswith("1")]

print(f"Found {len(files_to_move)} files to move.")

i = 0

# 移动文件
for file in files:
    i += 1
    if i % 5 == 0:
        source_path = os.path.join(source_directory, file)
        destination_path = os.path.join(destination_directory, file)
    
        # 移动文件到目标文件夹
        shutil.move(source_path, destination_path)
        print(f"Moved {file} to {destination_directory}")
