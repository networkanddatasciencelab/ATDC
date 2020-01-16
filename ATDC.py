"""
2019-01-24 @Javy Wang
Functions for anomalous trajectory detection and classification
"""
import pandas as pd
import math
import time
from sklearn import metrics
import seaborn as sn
from matplotlib import pyplot as plt 


def ATDC(path_dict, theta, phi, kn):
    """
    :param path_dict: the path of trajectory
    :param theta: a set of thresholds needs to be used to classify those trajectories
    :param phi: a set of thresholds of "absolutely normal” trajectories
    :param kn: k trajectories with the largest intersection with one trajectory
    :return: res_f1: the f1-score of ATDC
    """

    grid_file = path_dict['input_path'] + '.csv'  # laod trajectories
    print(grid_file)
    grid_df = pd.read_csv(grid_file)    # all trajectories between a pair of source and distination

    true_file = path_dict['true_file']  # load the label of trajectories
    true_df = pd.read_csv(true_file, )  
    true_df = true_df.loc[:, ['index', 'TRUE']]

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

    iso_df = pd.DataFrame(columns=('index', 'diff', 'inter', 'ratio', 'pred'))  
    start_time = time.time()  

    # first round 
    for k, v in cell_dict.items():
        # num = len(cell_dict)
        key_list = []  
        for key in cell_dict.keys():
            key_list.append(key)
        key_list.remove(k)  
        sample = {}  
        sample_keys = key_list
        sample_num = len(sample_keys)
        for k2 in sample_keys:
            sample.setdefault(k2, cell_dict[k2])

        diff_list = []  # |A-B|-|B-A| 
        inter_list = []  # |A^B| 
        for k1, v1 in sample.items():
            diff = list(set(v).difference(set(v1)))
            diff1 = list(set(v1).difference(set(v)))
            inter = list(set(v).intersection(set(v1)))
            diff_list.append(len(diff) - len(diff1))  
            inter_list.append(len(inter))  

        index = k
        diff_mean = round(sum(diff_list) / sample_num, 4)  
        inter_mean = round(sum(inter_list) / sample_num, 4) 
        if int(inter_mean) == 0:
            ratio = 10000  
        else:
            ratio = round(diff_mean / inter_mean, 4)

        if ratio > theta[0]:
            pred = 0
        elif (ratio > theta[1]) & (ratio <= theta[0]):
            pred = 1
        elif (ratio <= theta[1]) & (ratio >= theta[2]):
            pred = 2
        elif (ratio < theta[2]) & (ratio >= theta[3]):
            pred = 3
        else:
            pred = 4

        iso_df = iso_df.append({'index': index, 'diff': diff_mean, 'inter': inter_mean, 'ratio': ratio, 'pred': pred},
                               ignore_index=True)  

    end_time1 = time.time()
    cost_time1 = end_time1 - start_time  # the time of the first round
    print(cost_time1)

    iso_df1 = pd.merge(iso_df, true_df, on='index')

    save_file1 = path_dict['output_path'] + '_ATP1.csv'  
    iso_df1.to_csv(save_file1, sep=',', index=False, columns=['index', 'diff', 'inter', 'ratio', 'pred', 'TRUE'])

    title1 = path_dict['title'] + '_ATP1.png'  
    savefig_path1 = path_dict['savefig_path'] + title1  

    res_dict1 = calculate_prec(iso_df1, title1, savefig_path1)

    # second round
    norm_df = iso_df1[(iso_df1['ratio'] >= phi[0]) & (iso_df1['ratio'] <= phi[1])] 
    norm_index = norm_df['index'].values 
    iso_df2 = pd.DataFrame(columns=('index', 'diff', 'inter', 'ratio', 'pred')) 
    start_time2 = time.time()

    for k, v in cell_dict.items():
        if k in norm_index:
            continue  # print('yes')
        else:
            sample = {}  
            # sample_num = len(norm_index)  
            for k2 in norm_index:
                sample.setdefault(k2, cell_dict[k2])

            diff_list = []  # |A-B|-|B-A| 
            inter_list = []  # |A^B| 
            for k1, v1 in sample.items():
                diff = list(set(v).difference(set(v1)))
                diff1 = list(set(v1).difference(set(v)))
                inter = list(set(v).intersection(set(v1)))
                diff_list.append(len(diff) - len(diff1))  
                inter_list.append(len(inter))

            index = k
            val_dict = {'diff': diff_list, 'inter': inter_list}
            val_df = pd.DataFrame(val_dict)  
            val_df.sort_values(by='inter', ascending=False, inplace=True)
            if kn >= val_df['inter'].count():
                kn = val_df['inter'].count()  
            top_k = val_df.head(kn)  
            diff_mean = round(sum(list(top_k['diff'])) / kn, 4)
            inter_mean = round(sum(list(top_k['inter'])) / kn, 4)

            if math.fabs(inter_mean) <= 0.0001:
                ratio = 10000  
            else:
                ratio = round(diff_mean / inter_mean, 4)

            # 轨迹类别判断
            if ratio > theta[0]:
                pred = 0
            elif (ratio > theta[1]) & (ratio <= theta[0]):
                pred = 1
            elif (ratio <= theta[1]) & (ratio >= theta[2]):
                pred = 2
            elif (ratio < theta[2]) & (ratio >= theta[3]):
                pred = 3
            else:
                pred = 4

            iso_df2 = iso_df2.append(
                {'index': index, 'diff': diff_mean, 'inter': inter_mean, 'ratio': ratio, 'pred': pred},
                ignore_index=True)  

    end_time2 = time.time()
    cost_time2 = end_time2 - start_time2  
    print(cost_time2) 

    iso_df3 = pd.concat([iso_df2, norm_df[['index', 'diff', 'inter', 'ratio', 'pred']]],
                        ignore_index=True) 

    iso_df4 = pd.merge(iso_df3, true_df, on='index')

    save_file2 = path_dict['output_path'] + '_ATP2.csv' 
    iso_df4.to_csv(save_file2, sep=',', index=False, columns=['index', 'diff', 'inter', 'ratio', 'pred', 'TRUE'])

    all_cost_time = end_time2 - start_time  
    print(all_cost_time)  

    # return the f1-score of second round
    title2 = path_dict['title'] + '_ATP2.png' 
    savefig_path2 = path_dict['savefig_path'] + title2 

    res_dict2 = calculate_prec(iso_df4, title2, savefig_path2)
    res_df1 = pd.DataFrame(res_dict1)
    res_df2 = pd.DataFrame(res_dict2)
    results = pd.concat([res_df1, res_df2], axis=0)
    save_file3 = path_dict['output_path'] + '_results.csv' 
    results.to_csv(save_file3, sep=',',
                   columns=['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'micro avg', 'macro avg',
                            'weighted avg'])

    para_dict = {'theta': theta, 'phi': phi, 'kn': kn, 'time': [cost_time1, cost_time2, all_cost_time]}
    save_file4 = path_dict['output_path'] + '_paras.txt'
    f = open(save_file4, 'w+')
    f.write(str(para_dict))
    f.close()

    res_f1 = res_df2.head(1)
    res_f1['all time'] = all_cost_time
    return res_f1


def calculate_prec(co_df, title, savefig_path):
    target_names = ['class 0', 'class 1', 'class 3', 'class 4'] 
    results = metrics.classification_report(list(co_df['TRUE']), list(co_df['pred']), labels=[0, 1, 3, 4],
                                            target_names=target_names, digits=2, output_dict=True)

    # cm = metrics.confusion_matrix(list(co_df['TRUE']), list(co_df['pred']))
    # cm_df = pd.DataFrame(cm)
    # plt.figure(figsize=(8, 8), dpi=120) 
    # sn.heatmap(cm_df, annot=True, vmax=1, square=True, cmap="Blues")
    # plt.title(title)
    # plt.savefig(savefig_path, format='png')
    # plt.show()

    return results


def main():
    data_path = {
        'data1': 'D:\\ftdd\\data\\ATDC\\figs\\',
        'data2': 'D:\\ftdd\\data\\ATDC\\Trajectory\\',
        'data3': 'D:\\ftdd\\data\\ATDC\\Label\\',
        'data4': 'D:\\ftdd\\data\\ATDC\\results\\',
    }

    theta = [0.5, 0.10, -0.11, -0.5]
    phi = [-0.05, 0.05]
    kn = 10

    data_set = ['3100-4421', '3099-4421', '3159-4421', '3159-4481', '4421-3099', '4421-3159']
    # grid_set = ['500', '400', '300', '200', '100'] 
    f1 = pd.DataFrame(columns=('class 0', 'class 1', 'class 3', 'class 4', 'micro avg', 'macro avg', 'weighted avg','all time'))
    for i in range(1, 7):
        true_file = data_path['data3'] + str(i) + '_' + '300_' + data_set[i - 1] + '_TRUE.csv'
        input_path = data_path['data2'] + 'index_grid300_' + data_set[i - 1]  
        output_path = data_path['data4'] + str(i) + '_' + '300' + '_' + data_set[i - 1]
        title = str(i) + '_' + '300' + '_' + data_set[i - 1]
        savefig_path = data_path['data1']
        path_dict = {'input_path': input_path, 'output_path': output_path, 'true_file': true_file, 'title': title, 'savefig_path': savefig_path}
        res_f1 = ATDC2(path_dict, theta, phi, kn)
        f1_name = '300' + '_f1'
        res_f1.index = pd.Series([f1_name])
        f1 = pd.concat([f1, res_f1])

    save_path = data_path['data4'] + 'ATDC_f1.csv'
    f1.to_csv(save_path, sep=',', columns=['class 0', 'class 1', 'class 3', 'class 4', 'micro avg', 'macro avg', 'weighted avg','all time'])


if __name__ == '__main__':
    main()
