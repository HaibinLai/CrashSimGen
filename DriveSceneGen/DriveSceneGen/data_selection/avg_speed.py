import re
import math

def calculate_average_speed(file_path):
    # 初始化变量
    total_speed = 0
    count = 0

    # 定义正则表达式，用于提取每个states块中的velocity_x和velocity_y
    pattern = r'velocity_x:\s*(-?\d+\.\d+)\s+velocity_y:\s*(-?\d+\.\d+)'

    with open(file_path, 'r') as file:
        file_content = file.read()

        # 使用正则表达式找到所有匹配的velocity_x和velocity_y
        matches = re.findall(pattern, file_content)

        # 遍历所有匹配项，计算每个样本的速度
        for match in matches:
            velocity_x = float(match[0])
            velocity_y = float(match[1])
            
            # 计算速度大小（欧几里得速度）
            speed = math.sqrt(velocity_x ** 2 + velocity_y ** 2)
            
            # 累加速度和样本数量
            total_speed += speed
            count += 1

    # 计算并返回平均速度
    if count > 0:
        average_speed = total_speed / count
        return average_speed
    else:
        return 0  # 如果没有样本，返回0

# 调用函数并打印结果
for i in range(1000):
    file_path = 'one_scenario_{i}.txt'  # Average speed: 3.6949330306565877
    average_speed = calculate_average_speed(file_path)
    print(f'Average speed: of record {i} : {average_speed}')
