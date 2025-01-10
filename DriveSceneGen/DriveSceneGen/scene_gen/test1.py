import numpy as np

from scenariogeneration import xodr, prettyprint, ScenarioGenerator

# 给定起始点和最终点
x1, y1 = 37.96875, 40.0  # 初始点坐标
x2, y2 = 39.78555523, 39.09159738  # 最终点坐标

# 计算道路的长度（两点之间的直线距离）
distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
print(f"Road Length: {distance:.2f} meters")



# 计算道路的长度：计算相邻点之间的距离并求和
def calculate_road_length(data):
    length = 0.0
    for i in range(1, len(data)):
        x1, y1 = data[i-1, 0], data[i-1, 1]
        x2, y2 = data[i, 0], data[i, 1]
        length += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return length

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

# 假设初始曲率为 -0.00001，曲率变化率为 -0.003
initial_curvature = -0.00001
curvature_change = -0.003

# 假设道路的弯道段长度为 60 米
arc_length = distance+1

# 创建道路的几何路径，包含螺旋（Spiral）和弧形（Arc）
# roads = []
# roads.append(
#     xodr.create_road(
#         [
#             xodr.Spiral(initial_curvature, curvature_change, distance),  # 起始曲率，曲率变化，长度
#             xodr.Arc(curvature_change, arc_length),  # 曲率变化率，弧段长度
#         ],
#         id=5,
#         left_lanes=2,
#         right_lanes=3,
#     )
# )

print("Road created successfully with spiral and arc!")

data = []
initial_curvature = 0
road_length = 0

# 计算道路的长度：计算相邻点之间的距离并求和
def calculate_road_length(data):
    length = 0.0
    for i in range(1, len(data)):
        x1, y1 = data[i-1, 0], data[i-1, 1]
        x2, y2 = data[i, 0], data[i, 1]
        length += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    road_length = calculate_road_length(data)
    print(f"Road Length: {road_length:.2f} meters")
    return length



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
    initial_curvature = calculate_initial_curvature(data)
    print(f"Initial Curvature: {initial_curvature:.5f}")

    # 假设曲率变化率为线性变化
    curvature_change = (data[-1, 3] - data[0, 3]) / road_length  # 根据方向向量的变化来估算曲率变化
    print(f"Curvature Change: {curvature_change:.5f}")
    return curvature, curvature_change







class Scenario(ScenarioGenerator):
    def __init__(self):
        super().__init__()

    def road(self, **kwargs):
        # create some roads
        roads = []

        # create the road
        roads.append(
    xodr.create_road(
        [
            xodr.Spiral(initial_curvature, curvature_change, distance),  # 起始曲率，曲率变化，长度
            xodr.Arc(curvature_change, arc_length),  # 曲率变化率，弧段长度
        ],
        id=5,
        left_lanes=2,
        right_lanes=3,
    )
        )


        # junction = xodr.create_junction(roads[3:], 1, roads[0:3])

        # create the opendrive
        odr = xodr.OpenDrive("myroad3")
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

