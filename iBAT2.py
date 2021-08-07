"""
@Jingwei Wang
Improved algorithm-iBAT2
"""

import pandas as pd
import random
from random import choice
import math
import time
from sklearn import metrics


def iBAT2(input_path, output_path, at_ratio, true_file):
    """
    :param input_path: the input path of a dataset.
    :param output_path: the output path of results.
    :param at_ratio: the ratio of anomalous trajectories in this dataset.
    :param true_file: the file contains the true label of anomalous trajectories.
    :return: 
    """
    grid_file = input_path + '.csv'  
    print(grid_file)
    grid_df = pd.read_csv(grid_file)  
    # print(grid_df.head(2))
    grid_df.rename(columns={'index': 'traj_index'}, inplace=True)  
    # print(grid_df.head(2))
    df2dict = {}
    f = lambda x: df2dict.update({x.traj_index: x.grid_list})  
    grid_df.apply(f, axis=1) 

    cell_dict = {}
    for k, v in df2dict.items():
        raw_list = v
        raw_list = raw_list.strip('[]')
        raw_list = raw_list.split(',')
        if len(raw_list) == 0:
            cell_dict[k] = 0
        elif len(raw_list) == 1:
            cell_dict[k] = (raw_list[0])
        else:
            new_list = []
            for i in range(0, len(raw_list) - 1):
                new_list.append(int(raw_list[i])) 
            cell_dict[k] = new_list

    iso_df = pd.DataFrame(columns=('index', 'mean', 'max', 'min', 'mid', 'score', 'length', 'pred'))  
    start_time = time.time()

    for k, v in cell_dict.items():
        record_list = []  
        sample_num = 0  
        for i in range(100):
            num = len(cell_dict)
            key_list = []  
            for key in cell_dict.keys():
                key_list.append(key)
            key_list.remove(k)  
            sample = {}  
            if num > 256:
                sample_keys = random.sample(key_list, 256)
                sample_num = 256
                # print(sample)
                for k1 in sample_keys:
                    sample.setdefault(k1, cell_dict[k1])
            else:
                sample_keys = key_list
                sample_num = len(sample_keys)
                for k2 in sample_keys:
                    sample.setdefault(k2, cell_dict[k2])

            selected_cells = [] 
            cluster_dict = {1: sample_keys}  
            k_cluster = 1  
            former_cluster = 0  
            frozen_cluster = []  
            while k_cluster != former_cluster: 
                former_cluster = k_cluster 
                one_cell = choice(v) 
                while one_cell in selected_cells:
                    one_cell = choice(v)
                selected_cells.append(one_cell)  
                for cluster in range(1, k_cluster + 1):
                    if cluster in frozen_cluster:  
                        continue
                    else:
                        new_index_list = []  
                        cluster_size = len(cluster_dict[cluster])  
                        for v1 in cluster_dict[cluster]:
                            traj_list = sample[v1]   
                            if one_cell in traj_list:  
                                continue
                            else:
                                new_index_list.append(v1)  
                        if len(new_index_list) == 0:
                            frozen_cluster.append(cluster)
                        elif len(new_index_list) == cluster_size:
                            frozen_cluster.append(cluster)
                        else:
                            k_cluster = k_cluster + 1 
                            cluster_dict.update({k_cluster: new_index_list})
                            diff_list = list(
                                set(cluster_dict[cluster]).difference(set(new_index_list)))  
                            cluster_dict.pop(cluster)  
                            cluster_dict.update({cluster: diff_list})  

            record_list.append(len(selected_cells)) 

        index = k
        record_se = pd.Series(record_list)
        i_mean = record_se.mean()
        i_max = record_se.max()
        i_min = record_se.min()
        mid = record_se.median()
        score = 2 ** (-(i_mean / (2 * (math.log(sample_num - 1, math.e) + 0.57721566) - 2 * (sample_num - 1) / sample_num)))
        length = len(v) 
        iso_df = iso_df.append({'index': index, 'mean': i_mean, 'max': i_max, 'min': i_min, 'mid': mid, 'score': score, 'length': length, 'pred': 2},
                               ignore_index=True)  

    num_all_at = sum(at_ratio)  
    num_at04 = at_ratio[0] + at_ratio[3]  
    num_at13 = at_ratio[1] + at_ratio[2] 
    iso_df.sort_values(by=['score'], ascending=False, inplace=True)  

    at_all_df = iso_df.head(num_all_at) 
    at04_df = at_all_df.head(num_at04) 
    at13_df = at_all_df.tail(num_at13)  
    # the intersection of global anomalous trajectories (0: GD and 4:GS) and local anomalous trajectories (1: LD and 3: LS)
    print(pd.merge(at04_df, at13_df, on=['index']))

    at04_df.sort_values(by=['length'], ascending=False, inplace=True) 
    at13_df.sort_values(by=['length'], ascending=False, inplace=True)  

    at0_df = at04_df.head(at_ratio[0])  
    at0_df['pred'] = 0
    at4_df = at04_df.tail(at_ratio[3]) 
    at4_df['pred'] = 4
    at1_df = at13_df.head(at_ratio[1])  
    at1_df['pred'] = 1
    at3_df = at13_df.tail(at_ratio[2]) 
    at3_df['pred'] = 3

    num_traj = iso_df['index'].count()  
    norm_df = iso_df.tail(num_traj - num_all_at)  

    ibat2_df = pd.concat([at0_df, at1_df, norm_df, at3_df, at4_df], axis=0)
    save_file_name = output_path + '_iBAT2.csv'  
    ibat2_df.to_csv(save_file_name, sep=',', index=False, columns=['index', 'mean', 'max', 'min', 'mid', 'score', 'length', 'pred'])
    end_time = time.time()
    cost_time = end_time - start_time  
    print('%.2f ' % cost_time)  

    true_df = pd.read_csv(true_file) 
    true_df = true_df.loc[:, ['index', 'TRUE']]
    pred_df = pd.merge(ibat2_df, true_df, on='index')  

    target_names = ['class 0', 'class 1', 'class 3', 'class 4']  
    results = metrics.classification_report(list(pred_df['TRUE']), list(pred_df['pred']), labels=[0, 1, 3, 4],
                                            target_names=target_names, digits=2, output_dict=True)

    res_df = pd.DataFrame(results)
    res_df['time'] = cost_time
    print(res_df)
    save_file3 = output_path + '_results.csv' 
    res_df.to_csv(save_file3, sep=',',
                  columns=['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'micro avg', 'macro avg',
                           'weighted avg', 'time'])
    return res_df.head(1)

