from scenariogeneration import xodr, prettyprint
from scenariogeneration.xodr import LaneType
import numpy as np
 
#创建planview
x_start = 234.234
y_strat = 555.343
h_start = 98
planview = xodr.PlanView(x_start, y_strat, h_start)
 
#描述参考线并将其添加到planview上
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
 
 
##创建中心车道
rm = xodr.RoadMark(xodr.RoadMarkType.solid, 0.2)  ##设置车道中心线路标属性
centerlane = xodr.Lane(a=2)
centerlane.add_roadmark(rm)
 
##创建车道组
lanesec = xodr.LaneSection(0, centerlane)
 
##添加左右车道
lane2 = xodr.Lane(LaneType.parking,a=3)
lane2.add_roadmark(rm)
lane3 = xodr.Lane(LaneType.shoulder,a=3)
lane3.add_roadmark(rm)
 
lanesec.add_left_lane(lane2)
lanesec.add_right_lane(lane3)
 
##将车道段添加到车道中
lanes = xodr.Lanes()
lanes.add_lanesection(lanesec)
 
##创建道路
road = xodr.Road(1, planview, lanes)
odr = xodr.OpenDrive("road1")
odr.add_road(road)
 
##根据前驱后继（如果有配置）调整道路的初始位置
odr.adjust_roads_and_lanes()
 
##打印地图文件
prettyprint(odr.get_element())
 
##保存地图文件
odr.write_xml("test.xodr")