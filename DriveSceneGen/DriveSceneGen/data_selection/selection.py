import os, glob, pickle
import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2

raw_data_path = '/data/haibin/ML_DM/training_20s/'
process_data_path = '/data/haibin/ML_DM/process_selection'

raw_data = glob.glob(os.path.join(raw_data_path, '*.tfrecord*'))
raw_data.sort()

for data_file in raw_data:
    dataset = tf.data.TFRecordDataset(data_file, compression_type='')
    for cnt, data in enumerate(dataset):
        info = {}
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(bytearray(data.numpy()))
		
		# 一下几行代码可以将每个scenario的所有信息保存到txt文本文件中查看
        with open('./one_scenario.txt', 'w+') as f:
            f.write(str(scenario))
        f.close()
        print("--------------------- save successfully! ---------------------")
        quit()
        
        # print(type(scenario))
        info['scenario_id'] = scenario.scenario_id
        info['timestamps_seconds'] = list(scenario.timestamps_seconds)  # list of int of shape (91)
        info['current_time_index'] = scenario.current_time_index  # int, 10
        info['sdc_track_index'] = scenario.sdc_track_index  # int
        info['objects_of_interest'] = list(scenario.objects_of_interest)  # list, could be empty list

        info['tracks_to_predict'] = {
            'track_index': [cur_pred.track_index for cur_pred in scenario.tracks_to_predict],
            'difficulty': [cur_pred.difficulty for cur_pred in scenario.tracks_to_predict]
        }  # for training: suggestion of objects to train on, for val/test: need to be predicted
        info['tracks'] = list(scenario.tracks)
        info['dynamic_map_states'] = list(scenario.dynamic_map_states)
        # print(info)
        
        # output_file = os.path.join(process_data_path, f'sample_{scenario.scenario_id}.pkl')
        # with open(output_file, 'wb') as f:
        #     pickle.dump(info, f)
