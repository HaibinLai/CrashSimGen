import glob
import logging
import multiprocessing
import os
import argparse
import yaml

import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import numpy as np

# from scenariogeneration import xodr, prettyprint, ScenarioGenerator

from scenariogeneration import xodr, prettyprint, ScenarioGenerator
import os

from DriveSceneGen.utils.io import get_logger
from DriveSceneGen.utils.render import render_vectorized_scenario_on_axes
from DriveSceneGen.vectorization.direct.extract_vehicles import extract_agents
from DriveSceneGen.vectorization.graph import image_to_polylines, image_to_vectors_graph

logger = get_logger('vectorization', logging.WARNING)


def vectorize(img_color: Image, method: str = "GRAPH_FIT", map_range: float = 80.0, plot: bool = True, pic_save_path: str = None) -> tuple:
    """
    Returns
    ---
    lanes: `list` [lane1, lane2, ...]
        lane: `list` [point1, point2, ...] (follow the sequence of the traffic flow)
            point: `list` [x, y, z, dx, dy, dz]
    agents: `list` [agent1, agent2, ...]
        agent: `list` [center_x, center_y, center_z, length, width, height, angle, velocity_x, velocity_y]
    """
    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(img_color)

    try:
        # Extract centerlines
        if method == "GRAPH":
            lanes, graph = image_to_vectors_graph.extract_polylines_from_img(img_color, map_range=map_range, plot=plot, save_path=pic_save_path)
        
        elif method == "GRAPH_FIT":
            lanes, graph = image_to_polylines.extract_polylines_from_img(img_color, map_range=map_range, plot=plot, save_path=pic_save_path)
        
        elif method == "SEARCH":
            # TODO: Implement this method
            pass

        elif method == "DETR":
            # TODO: Implement this method
            pass

        else:
            print("Unknown method, Vectorization failed")
            return None, None, None, None
    
    except ValueError:
        logger.warning(f'Could not extract polylines from img')
        return None, None, None
    
    # Extract agents' properties
    agents = extract_agents(img_tensor, lanes)

    fig, axes = plt.subplots(1, 3)
    dpi = 100
    size_inches = 800 / dpi
    fig.set_size_inches([3*size_inches, size_inches])
    fig.set_dpi(dpi)
    fig.set_tight_layout(True)
    fig.set_facecolor("azure")  # 'azure', '#FFF5E0', 'lightcyan', 'xkcd:grey'
    axes = axes.ravel()
    axes[0].imshow(img_color)
    axes[0].set_aspect("equal")
    axes[0].margins(0)
    axes[0].grid(False)
    axes[0].axis("off")
    axes[1] = render_vectorized_scenario_on_axes(
        axes[1], lanes, [], map_range=map_range
    )
    axes[2] = render_vectorized_scenario_on_axes(
        axes[2], [], agents, map_range=map_range
    )
    
    return lanes, graph, agents, fig

global roads_out
roads_out = []
data = []
initial_curvature = 0
road_length = 0
curvature_change = 0

def calculate_road_length(data):
    length = 0.0
    for i in range(1, len(data)):
        x1, y1 = data[i-1, 0], data[i-1, 1]
        x2, y2 = data[i, 0], data[i, 1]
        length += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    print(f"Road Length: {length:.2f} meters")
    return length

def calculate_initial_curvature(data, road_length):
    # 取前三个点来估算初始曲率
    x1, y1, _, _, _, _ = data[0]
    x2, y2, _, _, _, _ = data[1]
    x3, y3, _, _, _, _ = data[2]

    # 计算两个相邻的切线方向向量
    dx1, dy1 = x2 - x1, y2 - y1
    dx2, dy2 = x3 - x2, y3 - y2

    # Curvature Estimation: k = 2 * (dx1 * dy2 - dy1 * dx2) / (|r1| * |r2|)
    numerator = dx1 * dy2 - dy1 * dx2
    denominator = (dx1**2 + dy1**2) * (dx2**2 + dy2**2)
    curvature = 2 * numerator / np.sqrt(denominator) if denominator != 0 else 0

    print(f"Initial Curvature: {curvature:.5f}")

    # 假设曲率变化率为线性变化
    curvature_change = (data[-1, 3] - data[0, 3]) / road_length  # 假设 data[-1, 3] 是最后一个点的方向
    print(f"Curvature Change: {curvature_change:.5f}")

    return curvature, curvature_change

signal = xodr.Signal(
                s=100,
                t=-12,
                zOffset=2.3,
                orientation=xodr.Orientation.positive,
                country="se",
                Type="c",
                subtype="31",
                value=10,
                name="right_100_sign",
                id=1,
                unit="km/h",
            )


class Scenario(ScenarioGenerator):
    def __init__(self):
        super().__init__()

    def road(self, **kwargs):
        # 创建 OpenDRIVE 道路：使用 Spira 和 Arc 描述
        # roads = []

        try:
            global roads_out

            id_r = 0
            for road in roads_out:

                road_length = calculate_road_length(road)
                print(f"Road Length: {road_length:.2f} meters")
                initial_curvature, curvature_change = calculate_initial_curvature(road,road_length)
                # print(f"Initial Curvature: {initial_curvature:.5f}")
                # curvature_change = (road[-1, 3] - road[0, 3]) / road_length  # 根据方向向量的变化来估算曲率变化


                if curvature_change > 0:
                    roads.append(
                        xodr.create_road(
                            [
                                xodr.Spiral(initial_curvature, curvature_change, road_length),  # 起始曲率，曲率变化，长度
                                xodr.Arc(curvature_change, road_length),  # 曲率变化率，弧段长度
                            ],
                            id=id_r,
                            center_road_mark=xodr.std_roadmark_broken(),
                            left_lanes=2,
                            right_lanes=3,
                        )
                    )
                else:
                    roads.append(
                        xodr.create_road(xodr.Line(road_length), id=id_r, left_lanes=2, right_lanes=3)
                    )

                id_r += 1

            # for i in range(len(roads) - 1):
            #      roads[i].add_successor(xodr.ElementType.junction, i+1)
            # # roads[1].add_successor(xodr.ElementType.junction, 1)


            # # junction = xodr.create_junction(roads[id_r:], 1, roads[0:id_r])

            
            #         # create the opendrive
            # odr = xodr.OpenDrive("myroad4")
            # for r in roads:
            #     odr.add_road(r)
            # odr.adjust_roads_and_lanes()
            #         # odr.add_junction(junction)

            roads = []
            numintersections = 3
            nlanes = 2

            # setup junction creator
            # junction_creator = xodr.CommonJunctionCreator(100, "my junction")

            # create some roads
            for road in  roads_out:
                road_length = calculate_road_length(road)
                print(f"Road Length: {road_length:.2f} meters")
                initial_curvature, curvature_change = calculate_initial_curvature(road,road_length)

                if curvature_change > 0:
                    roads.append(
                        xodr.create_road(
                            [
                                xodr.Spiral(initial_curvature, curvature_change, road_length),  # 起始曲率，曲率变化，长度
                                xodr.Arc(curvature_change, road_length),  # 曲率变化率，弧段长度
                            ],
                            id=id_r,
                            center_road_mark=xodr.std_roadmark_broken(),
                            left_lanes=2,
                            right_lanes=3,
                        )
                    )
                else:
                    roads.append(
                        xodr.create_road(xodr.Line(road_length), id=id_r, left_lanes=2, right_lanes=3)
                    )

                # roads.append(
                #     xodr.create_road(
                #         [xodr.Line(100)],
                #         id_r,
                #         center_road_mark=xodr.std_roadmark_broken(),
                #         left_lanes=nlanes,
                #         right_lanes=nlanes,
                #     )
                # )

                # # add road to junciton
                # junction_creator.add_incoming_road_circular_geometry(
                #     roads[id_r], 20, id_r * 2 * np.pi / numintersections, "successor"
                # )

                # add connection to all previous roads
                # for j in range(id_r):
                #     junction_creator.add_connection(j, id_r)
                id_r += 1

            roads[0].add_predecessor(xodr.ElementType.road, 1, xodr.ContactPoint.end)
            roads[0].add_successor(xodr.ElementType.road, len(roads) -1 , xodr.ContactPoint.start)

            for i in range(1, len(roads) - 1):
                roads[i].add_predecessor(xodr.ElementType.road, i+1, xodr.ContactPoint.end)
                roads[i].add_successor(xodr.ElementType.road, i-1, xodr.ContactPoint.start)
                # road4.add_predecessor(xodr.ElementType.road, 2, )
                # road4.add_successor(xodr.ElementType.road, 1, xodr.ContactPoint.start)

            odr = xodr.OpenDrive("myroad5")

            for r in roads:
                odr.add_road(r)
            # odr.add_junction_creator(junction_creator)

            odr.adjust_roads_and_lanes()
            return odr

        except:
            # create some roads
            roads = []
            roads.append(
                xodr.create_road(xodr.Line(300), id=0, left_lanes=1, right_lanes=2)
            )
            roads.append(
                xodr.create_road(xodr.Line(100), id=1, left_lanes=0, right_lanes=1)
            )
            roads.append(
                xodr.create_road(xodr.Line(100), id=2, left_lanes=1, right_lanes=3)
            )
            roads.append(
                xodr.create_road(
                    xodr.Spiral(0.001, 0.02, 30),
                    id=3,
                    left_lanes=1,
                    right_lanes=2,
                    road_type=1,
                )
            )
            roads.append(
                xodr.create_road(
                    xodr.Spiral(-0.001, -0.02, 30),
                    id=4,
                    left_lanes=0,
                    right_lanes=1,
                    road_type=1,
                )
            )

            # add some connections to non junction roads
            roads[0].add_successor(xodr.ElementType.junction, 1)
            roads[1].add_successor(xodr.ElementType.junction, 1)
            roads[2].add_predecessor(xodr.ElementType.junction, 1)

            # add connections to the first connecting road
            roads[3].add_predecessor(xodr.ElementType.road, 0, xodr.ContactPoint.end)
            roads[3].add_successor(xodr.ElementType.road, 2, xodr.ContactPoint.start)

            # add connections to the second connecting road with an offset
            roads[4].add_predecessor(xodr.ElementType.road, 1, xodr.ContactPoint.end)
            roads[4].add_successor(
                xodr.ElementType.road, 2, xodr.ContactPoint.start, lane_offset=-2
            )

            junction = xodr.create_junction(roads[3:], 1, roads[0:3])

            # create the opendrive
            odr = xodr.OpenDrive("myroad2")
            for r in roads:
                odr.add_road(r)
            odr.adjust_roads_and_lanes()
            odr.add_junction(junction)
            return odr
        
    
class Scenario3(ScenarioGenerator):
    def __init__(self):
        super().__init__()

    def road(self, **kwargs):
        ## create 3 roads
        road1 = xodr.create_road(
            [xodr.Line(30), xodr.Spiral(-0.00001, -0.035, 200)], 1, 2, 2
        )
        road2 = xodr.create_road(xodr.Line(100), 2, 2, 2)
        road3 = xodr.create_road(xodr.Line(100), 3, 2, 2)

        ## make a common junction

        jc = xodr.CommonJunctionCreator(100, "my junc")

        jc.add_incoming_road_cartesian_geometry(road1, 0, 0, 0, "successor")
        jc.add_incoming_road_cartesian_geometry(road2, 30, 0, -3.14, "predecessor")
        jc.add_incoming_road_cartesian_geometry(road3, 15, 15, -3.14 / 2, "successor")

        jc.add_connection(1, 2)
        jc.add_connection(3, 2)
        jc.add_connection(3, 1)

        ## make a forth road to connect two of the roads connected to the junction
        # since road1 has a Spiral as one geometry, it is very hard to determine the geometry of road4
        # instead use AdjustablePlanview as geometry!
        road4 = xodr.create_road(xodr.AdjustablePlanview(100), 4, 2, 2)

        # add predecessors and successors to the roads
        road4.add_predecessor(xodr.ElementType.road, 2, xodr.ContactPoint.end)
        road4.add_successor(xodr.ElementType.road, 1, xodr.ContactPoint.start)

        road2.add_successor(xodr.ElementType.road, 4, xodr.ContactPoint.start)
        road1.add_predecessor(xodr.ElementType.road, 4, xodr.ContactPoint.end)

        ## add roads to OpenDrive
        odr = xodr.OpenDrive("my road")
        odr.add_road(road1)
        odr.add_road(road2)
        odr.add_road(road3)
        odr.add_road(road4)
        odr.add_junction_creator(jc)

        ## Adjust initial positions of the roads looking at succ-pred logic
        odr.adjust_roads_and_lanes()

        return odr

class Scenario2(ScenarioGenerator):
    def __init__(self):
        super().__init__()

    def road(self, **kwargs):
        roads = []
        incoming_roads = 4
        nlanes = 1
        # setup junction creator
        junction_creator = xodr.CommonJunctionCreator(100, "my junction")

        # create roads and connections
        for i in range(incoming_roads):
            roads.append(
                xodr.create_road(
                    [xodr.Line(100)],
                    i,
                    center_road_mark=xodr.std_roadmark_broken(),
                    left_lanes=nlanes,
                    right_lanes=nlanes,
                )
            )
            roads[-1].add_signal(
                xodr.Signal(s=98.0, t=-4, country="USA", Type="R1", subtype="1")
            )

            # add road to junciton
            junction_creator.add_incoming_road_circular_geometry(
                roads[i], 20, i * 2 * np.pi / incoming_roads, "successor"
            )

            # add connection to all previous roads
            for j in range(i):
                junction_creator.add_connection(j, i)

        odr = xodr.OpenDrive("myroad")

        for r in roads:
            odr.add_road(r)
        odr.add_junction_creator(junction_creator)

        odr.adjust_roads_and_lanes()

        return odr

class Scenario3(ScenarioGenerator):
    def __init__(self):
        super().__init__()

    def road(self, **kwargs):
        # create some roads
        roads = []
        roads.append(
            xodr.create_road(xodr.Line(100), id=0, left_lanes=1, right_lanes=2)
        )
        roads.append(
            xodr.create_road(xodr.Line(100), id=1, left_lanes=0, right_lanes=1)
        )
        roads.append(
            xodr.create_road(xodr.Line(100), id=2, left_lanes=1, right_lanes=3)
        )
        roads.append(
            xodr.create_road(
                xodr.Spiral(0.001, 0.02, 30),
                id=3,
                left_lanes=1,
                right_lanes=2,
                road_type=1,
            )
        )
        roads.append(
            xodr.create_road(
                xodr.Spiral(-0.001, -0.02, 30),
                id=4,
                left_lanes=0,
                right_lanes=1,
                road_type=1,
            )
        )

        # add some connections to non junction roads
        roads[0].add_successor(xodr.ElementType.junction, 1)
        roads[1].add_successor(xodr.ElementType.junction, 1)
        roads[2].add_predecessor(xodr.ElementType.junction, 1)

        # add connections to the first connecting road
        roads[3].add_predecessor(xodr.ElementType.road, 0, xodr.ContactPoint.end)
        roads[3].add_successor(xodr.ElementType.road, 2, xodr.ContactPoint.start)

        # add connections to the second connecting road with an offset
        roads[4].add_predecessor(xodr.ElementType.road, 1, xodr.ContactPoint.end)
        roads[4].add_successor(
            xodr.ElementType.road, 2, xodr.ContactPoint.start, lane_offset=-2
        )

        junction = xodr.create_junction(roads[3:], 1, roads[0:3])

        # create the opendrive
        odr = xodr.OpenDrive("myroad2")
        for r in roads:
            odr.add_road(r)
        odr.adjust_roads_and_lanes()
        odr.add_junction(junction)
        return odr



class Scenario4(ScenarioGenerator):
    def __init__(self):
        super().__init__()

    def road(self, **kwargs):
        # create a road with some strange lane sections
        road1 = xodr.create_road(
            xodr.Line(195),
            1,
            [
                xodr.LaneDef(
                    13,
                    80,
                    3,
                    4,
                    3,
                    lane_start_widths=[3, 3, 3],
                    lane_end_widths=[3, 3, 3, 3],
                ),
                xodr.LaneDef(
                    120,
                    140,
                    4,
                    3,
                    2,
                    lane_start_widths=[3, 3, 3, 3],
                    lane_end_widths=[3, 3, 3],
                ),
            ],
            [
                xodr.LaneDef(
                    13,
                    80,
                    3,
                    4,
                    -1,
                    lane_start_widths=[3, 3, 3],
                    lane_end_widths=[3, 3, 3, 3],
                ),
                xodr.LaneDef(
                    120,
                    140,
                    4,
                    3,
                    -4,
                    lane_start_widths=[3, 3, 3, 3],
                    lane_end_widths=[3, 3, 3],
                ),
            ],
            center_road_mark=xodr.std_roadmark_broken_broken(),
        )

        # create more roads
        road2 = xodr.create_road(
            xodr.Line(103),
            2,
            xodr.LaneDef(40, 80, 3, 3, lane_start_widths=[3, 3, 3]),
            3,
            center_road_mark=xodr.std_roadmark_broken_broken(),
        )

        road3 = xodr.create_road(xodr.Spiral(0.00001, 0.01, 100), 3, 3, 0)
        road4 = xodr.create_road(xodr.Spiral(-0.0001, -0.01, 100), 4, 0, 3)

        road5 = xodr.create_road(
            xodr.Line(100),
            5,
            xodr.LaneDef(40, 80, 3, 3, lane_start_widths=[3, 3, 3]),
            3,
            center_road_mark=xodr.std_roadmark_broken_broken(),
        )
        road6 = xodr.create_road(
            xodr.Line(100),
            6,
            xodr.LaneDef(40, 80, 3, 3, lane_start_widths=[3, 3, 3]),
            3,
            center_road_mark=xodr.std_roadmark_broken_broken(),
        )

        # connect all roads
        road1.add_successor(xodr.ElementType.road, 2, xodr.ContactPoint.start)
        road1.add_predecessor(xodr.ElementType.junction, 100)

        road2.add_predecessor(xodr.ElementType.road, 1, xodr.ContactPoint.end)

        road3.add_predecessor(xodr.ElementType.junction, 100)
        road4.add_predecessor(xodr.ElementType.junction, 100)

        # create a direct junction
        junc = xodr.DirectJunctionCreator(100, "direct_junction")

        junc.add_connection(road1, road3)
        junc.add_connection(road1, road4)

        # create a common junction
        junc2 = xodr.CommonJunctionCreator(200, "common junction")
        junc2.add_incoming_road_cartesian_geometry(
            road2, 0, 0, -3.14 * 3 / 2, "successor"
        )
        junc2.add_incoming_road_cartesian_geometry(road5, 30, 20, -3.14, "successor")
        junc2.add_incoming_road_cartesian_geometry(road6, -30, 20, 0, "predecessor")

        junc2.add_connection(2, 5)
        junc2.add_connection(2, 6)

        # add everything to the OpenDRIVE object
        odr = xodr.OpenDrive("my adjusted road")
        odr.add_road(road1)
        odr.add_road(road2)
        odr.add_road(road3)
        odr.add_road(road4)
        odr.add_road(road5)
        odr.add_road(road6)
        odr.add_junction_creator(junc)
        odr.add_junction_creator(junc2)

        # adjust roads and lanes
        odr.adjust_roads_and_lanes()

        # adjust roadmarks (comment out to see difference)
        odr.adjust_roadmarks()
        return odr


class Scenario5(ScenarioGenerator):
    def __init__(self):
        super().__init__()

    def road(self, **kwargs):
        # create a road with some strange lane sections
        road1 = xodr.create_road(xodr.Line(200), 1, 3, 3, lane_width=4)

        # create 3 poles
        road1.add_object(
            xodr.Object(
                s=100, t=-12, id=1, length=0.05, width=0.05, radius=0.5, height=2.3
            )
        )
        road1.add_object(
            xodr.Object(
                s=100, t=0, id=1, length=0.05, width=0.05, radius=0.5, height=2.3
            )
        )
        road1.add_object(
            xodr.Object(
                s=100, t=12, id=1, length=0.05, width=0.05, radius=0.5, height=2.3
            )
        )

        # create some signs (signals)
        road1.add_signal(
            xodr.Signal(
                s=100,
                t=-12,
                zOffset=2.3,
                orientation=xodr.Orientation.positive,
                country="se",
                Type="c",
                subtype="31",
                value=10,
                name="right_100_sign",
                id=1,
                unit="km/h",
            )
        )
        road1.add_signal(
            xodr.Signal(
                s=100,
                t=0,
                zOffset=2.3,
                orientation=xodr.Orientation.positive,
                country="se",
                Type="c",
                subtype="31",
                value=10,
                name="right_100_sign",
                id=1,
                unit="km/h",
            )
        )
        road1.add_signal(
            xodr.Signal(
                s=100,
                t=0,
                zOffset=2.3,
                orientation=xodr.Orientation.negative,
                country="se",
                Type="c",
                subtype="31",
                value=9,
                name="right_100_sign",
                id=1,
                unit="km/h",
            )
        )
        road1.add_signal(
            xodr.Signal(
                s=100,
                t=12,
                zOffset=2.3,
                orientation=xodr.Orientation.negative,
                country="se",
                Type="c",
                subtype="31",
                value=9,
                name="right_100_sign",
                id=1,
                unit="km/h",
            )
        )

        # add everything to the OpenDRIVE object
        odr = xodr.OpenDrive("my adjusted road")
        odr.add_road(road1)

        # adjust roads and lanes
        odr.adjust_roads_and_lanes()

        # adjust roadmarks (comment out to see difference)
        odr.adjust_roadmarks()
        return odr

class Scenario6(ScenarioGenerator):
    def __init__(self):
        super().__init__()

    def road(self, **kwargs):
        # create a road
        road = xodr.create_road([xodr.Line(1000)], 0, 2, 2)

        ## Create the OpenDrive class (Master class)
        odr = xodr.OpenDrive("myroad")

        ## Finally add roads to Opendrive
        odr.add_road(road)

        # Adjust initial positions of the roads looking at succ-pred logic
        odr.adjust_roads_and_lanes()

        # After adjustment, repeating objects on side of the road can be added automatically
        guardrail = xodr.Object(
            0,
            0,
            height=0.3,
            zOffset=0.4,
            Type=xodr.ObjectType.barrier,
            name="guardRail",
        )
        road.add_object_roadside(guardrail, 0, 0, tOffset=0.8)

        delineator = xodr.Object(
            0, 0, height=1, zOffset=0, Type=xodr.ObjectType.pole, name="delineator"
        )
        road.add_object_roadside(delineator, 50, sOffset=25, tOffset=0.85)

        ## Add some other objects at specific positions
        # single emergency callbox
        emergencyCallbox = xodr.Object(
            30, -6, Type=xodr.ObjectType.pole, name="emergencyCallBox"
        )
        road.add_object(emergencyCallbox)

        # repeating jersey barrier
        jerseyBarrier = xodr.Object(
            0,
            0,
            height=0.75,
            zOffset=0,
            Type=xodr.ObjectType.barrier,
            name="jerseyBarrier",
        )
        jerseyBarrier.repeat(repeatLength=25, repeatDistance=0, sStart=240)
        road.add_object(jerseyBarrier)

        return odr


class Scenario7(ScenarioGenerator):
    def __init__(self):
        super().__init__()

    def road(self, **kwargs):
        # create a simple planview
        planview = xodr.PlanView()
        planview.add_geometry(xodr.Line(300))

        ## create the customized centerlanes with different lanemarkings
        centerlanes = []
        # standard solid solid

        solid_solid = xodr.Lane()
        solid_solid.add_roadmark(xodr.std_roadmark_solid_solid())
        centerlanes.append(solid_solid)

        # standard solid broken
        solid_broken = xodr.Lane()
        solid_broken.add_roadmark(xodr.std_roadmark_solid_broken())
        centerlanes.append(solid_broken)

        # customized broken broken
        broken_broken_roadmark = xodr.RoadMark(xodr.RoadMarkType.broken_broken)
        broken_broken_roadmark.add_specific_road_line(xodr.RoadLine(0.2, 9, 3, 0.2))
        broken_broken_roadmark.add_specific_road_line(xodr.RoadLine(0.2, 3, 9, -0.2, 3))
        broken_broken = xodr.Lane()
        broken_broken.add_roadmark(broken_broken_roadmark)
        centerlanes.append(broken_broken)

        # create the different lanesections and add them to the lanes
        lanes = xodr.Lanes()
        ls_start = 0
        for i in centerlanes:
            lanesection = xodr.LaneSection(ls_start, i)
            left_lane_with_roadmark = xodr.Lane(a=4)
            left_lane_with_roadmark.add_roadmark(xodr.std_roadmark_broken())

            right_lane_with_roadmark = xodr.Lane(a=4)
            right_lane_with_roadmark.add_roadmark(xodr.std_roadmark_solid())
            lanesection.add_left_lane(left_lane_with_roadmark)
            lanesection.add_right_lane(right_lane_with_roadmark)
            ls_start += 100

            lanes.add_lanesection(lanesection)

        # create the road
        road = xodr.Road(0, planview, lanes)

        # create the opendrive and add the road
        odr = xodr.OpenDrive("road with custom lanes")
        odr.add_road(road)

        # adjust the road
        odr.adjust_roads_and_lanes()

        return odr
    

class Scenario8(ScenarioGenerator):
    def __init__(self):
        super().__init__()

    def road(self, **kwargs):
        ## Multiple geometries in one only road.

        ##1. Create the planview
        planview = xodr.PlanView()

        ##2. Create some geometries and add them to the planview
        line1 = xodr.Line(100)
        arc1 = xodr.Arc(0.05, angle=np.pi / 2)
        line2 = xodr.Line(100)
        cloth1 = xodr.Spiral(0.05, -0.1, 30)
        line3 = xodr.Line(100)

        planview.add_geometry(line1)
        planview.add_geometry(arc1)
        planview.add_geometry(line2)
        planview.add_geometry(cloth1)
        planview.add_geometry(line3)

        ##3. Create a solid roadmark
        rm = xodr.RoadMark(xodr.RoadMarkType.solid, 0.2)

        ##4. Create centerlane
        centerlane = xodr.Lane(a=2)
        centerlane.add_roadmark(rm)

        ##5. Create lane section form the centerlane
        lanesec = xodr.LaneSection(0, centerlane)

        ##6. Create left and right lanes
        lane2 = xodr.Lane(a=3)
        lane2.add_roadmark(rm)
        lane3 = xodr.Lane(a=3)
        lane3.add_roadmark(rm)

        ##7. Add lanes to lane section
        lanesec.add_left_lane(lane2)
        lanesec.add_right_lane(lane3)

        ##8. Add lane section to Lanes
        lanes = xodr.Lanes()
        lanes.add_lanesection(lanesec)

        ##9. Create Road from Planview and Lanes
        road = xodr.Road(1, planview, lanes)

        ##10. Create the OpenDrive class (Master class)
        odr = xodr.OpenDrive("myroad")

        ##11. Finally add roads to Opendrive
        odr.add_road(road)

        ##12. Adjust initial positions of the roads looking at succ-pred logic
        odr.adjust_roads_and_lanes()

        return odr



def multiprocessing_func(data_files: list, cfg: dict, vectorized_dir: str, picture_dir: str, graph_dir: str, agent_dir: str, n_proc: int, proc_id: int):
    for index, file in enumerate(tqdm(data_files)):
        with open(file, "rb") as f:
            img_id = index*n_proc + proc_id
            img_color = Image.open(f).convert("RGB")

            # Vectorization
            vec_save_path = f'{vectorized_dir}/{img_id}.pkl'
            pic_save_path = f'{picture_dir}/{img_id}_process.png'
            graph_save_path = f'{graph_dir}/{img_id}_graph.pickle'
            agent_save_path = f'{agent_dir}/{img_id}_agents.npy'
            
            try:
                lanes, graph, agents, fig = vectorize(img_color, method=cfg['method'], map_range=cfg['map_range'], plot=cfg['plot'], pic_save_path=pic_save_path)
                
                if fig is not None:
                    fig.savefig(f'{picture_dir}/{img_id}.png', transparent=True, format="png")
                plt.close()
                
                if graph is not None:
                    try:
                        with open(graph_save_path, "wb") as f:
                            pickle.dump(graph, f)
                    except ValueError:
                        logger.error(f'Failed to save graph!')
                        continue
                
                if agents is not None and lanes is not None:
                    np.save(agent_save_path, np.array(agents))
                
            except OSError:
                logger.warning(f'File no. {img_id}: failed to be vectorized due to insufficient memory!')
                plt.close()
                break
            except Exception as e:
                logger.warning(f'File no. {img_id} failed to be vectorized due to {e}')
                plt.close()
                continue
            
            ## save the scenario
            output_dict = {}
            output_dict["scenario_id"] = index
            output_dict["sdc_track_index"] = 0
            output_dict["object_type"] = np.ones((len(agents)))
            output_dict["all_agent"] = agents
            output_dict["lane"] = lanes

            print(type(lanes), type(graph), type(agents), type(fig))
            print(type(lanes[0]))
            print(lanes[0]) # ndarray
            print(len(lanes[0]))

            global roads_out
            roads_out = lanes

            sce = Scenario()
            # Print the resulting xml
            prettyprint(sce.road().get_element())

            # write the OpenDRIVE file as xosc using current script name
            sce.generate(".")
            # xord_roads = []
            # for lane in lanes:
            #     xord_roads.append(
            #         xodr.create_road(

            #         )
            #         )

            torch.save(output_dict, vec_save_path)
            
    return


def chunks(input, n):
    """Yields successive n-sized chunks of input"""
    for i in range(0, len(input), n):
        yield input[i : i + n]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Vectorization')
    parser.add_argument('--load_path',default="/data/haibin/ML_DM/training_20s", type=str, help='path to dataset files')
    parser.add_argument('--save_path', default="/data/haibin/ML_DM/vectorization",type=str, help='path to save processed data')
    parser.add_argument('--cfg_file', default="./DriveSceneGen/config/vectorization.yaml",type=str, help='path to cfg file')
    
    sys_args = parser.parse_args()
    with open(sys_args.cfg_file, 'r') as file:
        cfg = yaml.safe_load(file)
    n_proc = cfg['n_proccess']
    
    map_range = cfg['vectoriztion']['map_range']
    
    # Generated Dataset Paths
    # input_dir = f'/data/haibin/ML_DM/generation/generated_{map_range}m_5k'
    input_dir = f'/data/haibin/ML_DM/test3'
    generated_imgs_dir = input_dir # os.path.join(input_dir, 'diffusion')
    outputs_dir = input_dir
    
    vectorized_output_dir = os.path.join(outputs_dir, "vectorized")
    vectorized_picture_dir = os.path.join(outputs_dir, "vectorized_pics")
    graph_dir = os.path.join(outputs_dir, "graph")
    agent_dir = os.path.join(outputs_dir, "agent")
    os.makedirs(vectorized_output_dir, exist_ok=True)
    os.makedirs(vectorized_picture_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)
    os.makedirs(agent_dir, exist_ok=True)

    ## get the list of all the files
    all_files = glob.glob(generated_imgs_dir + "/*")

    ## split the input files into n_proc chunks
    chunked_files = list(chunks(all_files, int(len(all_files) / n_proc) + 1))

    # Initialize the parallel processes list
    processes = []
# 
    # multiprocessing_func(chunked_files[proc_id], cfg['vectoriztion'], vectorized_output_dir, vectorized_picture_dir, graph_dir, agent_dir, n_proc, 1)
    for proc_id in np.arange(n_proc):
        """Execute the target function on the n_proc target processors using the splitted input"""
        p = multiprocessing.Process(
            target=multiprocessing_func,
            args=(chunked_files[proc_id], cfg['vectoriztion'], vectorized_output_dir, vectorized_picture_dir, graph_dir, agent_dir, n_proc, proc_id),
        )
        processes.append(p)
        p.start()
    for process in processes:
        process.join()

    print(f"Process finished!!!, results saved to: {vectorized_output_dir}")
