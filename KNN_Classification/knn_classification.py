import os,glob
import numpy as np
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2
from sklearn.neighbors import NearestNeighbors
import numpy as np

def classify_scenario(start,change,process,end, lane_num):
    diff=abs(start-end)
    flag=False
    for p in process:
        if p<=np.pi/9 and p>=np.pi/60:
            flag=True
        if p>np.pi/9:
            flag=False
            break
    if diff>np.pi:
        diff=2*np.pi-diff
    if diff > np.pi/3 :
        return 'turn'
    elif change and flag:
        return 'lane change'
    elif diff<=np.pi/90:
        return 'straight'
    else:
        return 'Unknow'
    

def find_majority_lane_id(nearest_lane_ids):
    # 统计每个道路ID出现的次数
    lane_id_count = {}
    for lane_id in nearest_lane_ids:
        if lane_id in lane_id_count:
            lane_id_count[lane_id] += 1
        else:
            lane_id_count[lane_id] = 1
    
    # 找到出现次数最多的道路ID
    majority_lane_id = max(lane_id_count, key=lane_id_count.get)
    return majority_lane_id

def paint(label, save_data_path, save_image_path):
    raw_data_path = f'{save_data_path}{label}/'
    raw_data = glob.glob(os.path.join(raw_data_path, '*.tfrecord*'))
    raw_data.sort()
    count=0
    for data_file in raw_data:
        dataset = tf.data.TFRecordDataset(data_file, compression_type='')
        for _, data in enumerate(dataset):
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(bytearray(data.numpy()))
            print(scenario.scenario_id)
            plt.figure(figsize=(30,30))
            plt.rcParams['axes.facecolor']='grey'
            # 画地图线
            for i in range(len(scenario.map_features)):
                # 车道线
                if str(scenario.map_features[i].lane) != '':
                    line_x = [z.x for z in scenario.map_features[i].lane.polyline]
                    line_y = [z.y for z in scenario.map_features[i].lane.polyline]
                    plt.scatter(line_x, line_y, c='g', s=5)
                # 边界线
                if str(scenario.map_features[i].road_edge) != '':
                    road_edge_x = [polyline.x for polyline in scenario.map_features[i].road_edge.polyline]
                    road_edge_y = [polyline.y for polyline in scenario.map_features[i].road_edge.polyline]
                    plt.scatter(road_edge_x, road_edge_y)
                    if scenario.map_features[i].road_edge.type == 2:
                        plt.scatter(road_edge_x, road_edge_y, c='k')
                        
                    elif scenario.map_features[i].road_edge.type == 3:
                        plt.scatter(road_edge_x, road_edge_y, c='purple')
                        print(scenario.map_features[i].road_edge)
                    else:
                        plt.scatter(road_edge_x, road_edge_y, c='k')
                # 道路边界线
                if str(scenario.map_features[i].road_line) != '':
                    road_line_x = [j.x for j in scenario.map_features[i].road_line.polyline]
                    road_line_y = [j.y for j in scenario.map_features[i].road_line.polyline]
                    if scenario.map_features[i].road_line.type == 7:  # 双实黄线
                        plt.plot(road_line_x, road_line_y, c='y')
                    elif scenario.map_features[i].road_line.type == 8:  # 双虚实黄线
                        plt.plot(road_line_x, road_line_y, c='y') 
                    elif scenario.map_features[i].road_line.type == 6:  # 单实黄线
                        plt.plot(road_line_x, road_line_y, c='y')
                    elif scenario.map_features[i].road_line.type == 1:  # 单虚白线
                        for i in range(int(len(road_line_x)/7)):
                            plt.plot(road_line_x[i*7:5+i*7], road_line_y[i*7:5+i*7], color='w')
                    elif scenario.map_features[i].road_line.type == 2:  # 单实白线
                        plt.plot(road_line_x, road_line_y, c='w')
                    else:
                        plt.plot(road_line_x, road_line_y, c='w')
            
            # 画车及轨迹
            for i in range(len(scenario.tracks)):
                if i==scenario.sdc_track_index:
                    traj_x = [center.center_x for center in scenario.tracks[i].states if center.center_x != 0.0]
                    traj_y = [center.center_y for center in scenario.tracks[i].states if center.center_y != 0.0]
                    plt.scatter(traj_x[0], traj_y[0], s=140, c='r', marker='s')
                    plt.scatter(traj_x, traj_y, s=14, c='r')
                else:
                    traj_x = [center.center_x for center in scenario.tracks[i].states if center.center_x != 0.0]
                    traj_y = [center.center_y for center in scenario.tracks[i].states if center.center_y != 0.0]
                    plt.scatter(traj_x[0], traj_y[0], s=140, c='k', marker='s')
                    plt.scatter(traj_x, traj_y, s=14, c='b')    
            break
        file_name = os.path.basename(data_file)
        dir=os.path.join(save_image_path,label)
        os.makedirs(dir, exist_ok=True)
        plt.savefig(f'{dir}/show_map_{file_name.split("tfrecord-")[1]}.png', dpi=300, bbox_inches='tight')
        plt.close()
        count=count+1


def main():
    raw_data_path = './training_100/'  # 原始数据集位置
    save_data_path = './training_classify/'  # 分类后数据集复制到的位置
    save_image_path = './training_20s_save/'  # 保存图像位置

    raw_data = glob.glob(os.path.join(raw_data_path, '*.tfrecord*'))
    raw_data.sort()

    for data_file in raw_data:
        dataset = tf.data.TFRecordDataset(data_file, compression_type='')
        for _, data in enumerate(dataset):
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(bytearray(data.numpy()))


            vehicle_points = []
            for i in range(len(scenario.tracks)):
                if i==scenario.sdc_track_index:
                    traj_x = [center.center_x for center in scenario.tracks[i].states if center.center_x != 0.0]
                    traj_y = [center.center_y for center in scenario.tracks[i].states if center.center_y != 0.0]
                    head = [center.heading for center in scenario.tracks[i].states if center.center_y != 0.0]
                    start_head=head[0]
                    middle_head=head[int(len(head)/2)]
                    process_head=[abs(h-head[0]) for h in head]
                    end_head=head[-1]
                    print(start_head-end_head)
                    vehicle_points.append([traj_x, traj_y])
                    vehicle_points_correct=np.array([list(zip(traj_x, traj_y)) for traj_x, traj_y in vehicle_points]).reshape(-1, 2)
                    break

        # 提取 map_features 中 type 为 lane 或 edge 的点
            lane_points = []
            lane_ids = []
            lane_index=[]
            for i in range(len(scenario.map_features)):
                # 车道线
                if str(scenario.map_features[i].lane) != '':
                    line_x = [z.x for z in scenario.map_features[i].lane.polyline]
                    line_y = [z.y for z in scenario.map_features[i].lane.polyline]
                    lane_points.append([line_x, line_y])
                    lane_ids.extend([scenario.map_features[i].id] * len(line_x))
                    lane_index.extend([i]*len(line_x))

        

        # lane_points = np.array(lane_points)
            lane_points_corrected = np.array([[x, y] for lane in lane_points for x, y in zip(lane[0], lane[1])])

            # 使用 K-NN 算法找到最近的 K 个点
            k = 10  # 假设 K=10
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(lane_points_corrected)
            distances, indices = nbrs.kneighbors(vehicle_points_correct)
            # print(indices[0])
            # 使用索引从 lane_ids 中获取对应的 lane ID

            closest_lane_ids=[]
            closest_lane_index=[]
            # print(len(indices))
            for inid in indices:
                nearest_lane_ids = [lane_ids[ind] for ind in inid]  # 得到第一个点最近的10个lane点
                # print(nearest_lane_ids)
                nearest_lane_index = [lane_index[ind] for ind in inid]
                majority_lane_id = find_majority_lane_id(nearest_lane_ids)
                majority_lane_index = find_majority_lane_id(nearest_lane_index)
                closest_lane_ids.append(majority_lane_id)
                closest_lane_index.append(majority_lane_index)

            pass_lane_ids = list(set(closest_lane_ids))
            pass_lane_index=list(set(closest_lane_index))
            # 查找该pass_lane_ids的neighbour是否同样在pass_lane_ids中
            # print(pass_lane_ids)
            # exit()
            change_neighbour=False
            for i in pass_lane_index:
                left_neighbour_id=[ln.feature_id for ln in scenario.map_features[i].lane.left_neighbors]
                right_neighbors_id=[rn.feature_id for rn in scenario.map_features[i].lane.right_neighbors]
                for id in left_neighbour_id:
                    if id in pass_lane_ids:
                        change_neighbour=True
                        break
                if change_neighbour==True:
                    break
                for id in right_neighbors_id:
                    if id in pass_lane_ids:
                        change_neighbour=True
                        break
                if change_neighbour==True: 
                    break
                
            # print(change_neighbour)
            # exit()
            # print(start_head,end_head)
            
            label=classify_scenario(start_head,change_neighbour,process_head,end_head,len(pass_lane_ids))
            src = data_file
            dir = os.path.join(save_data_path, label)
            os.makedirs(dir, exist_ok=True)
            dst = os.path.join(dir, os.path.basename(data_file))
            shutil.copy(src, dst)
            break

    paint("straight",save_data_path,save_image_path)
    paint("turn",save_data_path,save_image_path)
    paint("lane change",save_data_path,save_image_path)
    paint("Unknow",save_data_path,save_image_path)


if __name__ == '__main__':
    main()