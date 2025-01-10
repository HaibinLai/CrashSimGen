import numpy as np

from scenariogeneration import xodr, prettyprint, ScenarioGenerator

import os
import pickle

pkl_path = "/data/haibin/ML_DM/test/vectorized/1.pkl"
# 打开并读取 pickle 文件
with open(pkl_path, 'rb') as file:
    # 加载 pickle 文件中的对象
    data = pickle.load(file)

# 打印数据，查看其内容
print(data['lane'][0])

# 假设这是你给出的道路坐标数据
# data = np.array([
#     [37.96875, 40.0, 0.0, 0.89442719, -0.4472136, 0.0],
#     [38.10850425, 39.93012288, 0.0, 0.89442719, -0.4472136, 0.0],
#     [38.2482585, 39.86024575, 0.0, 0.89442719, -0.4472136, 0.0],
#     [38.38801275, 39.79036863, 0.0, 0.89442719, -0.4472136, 0.0],
#     [38.52776699, 39.7204915, 0.0, 0.89442719, -0.4472136, 0.0],
#     [38.66752124, 39.65061438, 0.0, 0.89442719, -0.4472136, 0.0],
#     [38.80727549, 39.58073725, 0.0, 0.89442719, -0.4472136, 0.0],
#     [38.94702974, 39.51086013, 0.0, 0.89442719, -0.4472136, 0.0],
#     [39.08678399, 39.44098301, 0.0, 0.89442719, -0.4472136, 0.0],
#     [39.22653824, 39.37110588, 0.0, 0.89442719, -0.4472136, 0.0],
#     [39.36629249, 39.30122876, 0.0, 0.89442719, -0.4472136, 0.0],
#     [39.50604673, 39.23135163, 0.0, 0.89442719, -0.4472136, 0.0],
#     [39.64580098, 39.16147451, 0.0, 0.89442719, -0.4472136, 0.0],
#     [39.78555523, 39.09159738, 0.0, 0.89442719, -0.4472136, 0.0]
# ])

# 计算道路的长度：计算相邻点之间的距离并求和
def calculate_road_length(data):
    length = 0.0
    for i in range(1, len(data)):
        x1, y1 = data[i-1, 0], data[i-1, 1]
        x2, y2 = data[i, 0], data[i, 1]
        length += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return length

road_length = calculate_road_length(data)
print(f"Road Length: {road_length:.2f} meters")

# 计算起始曲率：我们可以使用相邻三个点来估计曲率
def calculate_initial_curvature(data):
    # 取前两个点来估算初始曲率
    x1, y1, _, _, _, _ = data[0]
    x2, y2, _, _, _, _ = data[1]
    x3, y3, _, _, _, _ = data[2]

    # 计算两个相邻的切线方向向量
    dx1, dy1 = x2 - x1, y2 - y1
    dx2, dy2 = x3 - x2, y3 - y2

    # 估算曲率：通过曲率公式 k = 2 * (x2 - x1) / (|r1| * |r2|)
    # 这里假设 x1, x2, x3 为弯道上的连续三个点
    numerator = dx1 * dy2 - dy1 * dx2
    denominator = (dx1**2 + dy1**2) * (dx2**2 + dy2**2)
    curvature = 2 * numerator / denominator if denominator != 0 else 0
    return curvature

initial_curvature = calculate_initial_curvature(data)
print(f"Initial Curvature: {initial_curvature:.5f}")

# 假设曲率变化率为线性变化
curvature_change = (data[-1, 3] - data[0, 3]) / road_length  # 根据方向向量的变化来估算曲率变化

print(f"Curvature Change: {curvature_change:.5f}")


class Scenario(ScenarioGenerator):
    def __init__(self):
        super().__init__()

    def road(self, **kwargs):
        # 创建 OpenDRIVE 道路：使用 Spira 和 Arc 描述
        roads = []

        if curvature_change > 0:
            roads.append(
                xodr.create_road(
                    [
                        xodr.Spiral(initial_curvature, curvature_change, road_length),  # 起始曲率，曲率变化，长度
                        xodr.Arc(curvature_change, road_length),  # 曲率变化率，弧段长度
                    ],
                    id=5,
                    left_lanes=2,
                    right_lanes=3,
                )
            )
        else:
            roads.append(
                xodr.create_road(xodr.Line(road_length), id=2, left_lanes=1, right_lanes=3)
            )

    
                # create the opendrive
        odr = xodr.OpenDrive("myroad4")
        for r in roads:
                    odr.add_road(r)
        odr.adjust_roads_and_lanes()
                # odr.add_junction(junction)
        return odr


if __name__ == "__main__":
    sce = Scenario()
    # Print the resulting xml
    prettyprint(sce.road().get_element())

    # write the OpenDRIVE file as xosc using current script name
    sce.generate(".")

    print("Road created successfully!")
