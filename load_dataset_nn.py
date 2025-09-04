import numpy as np
import joblib,os
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
from dataset_split import loaddata,loaddata_preproc,loaddata_preproc_v2,loaddata_preproc_v3,extract_window_data
from modules_maxminscaler import datasetphysicalSharedMaxMinScalers

def load_dataset_zzm_train_valid_test(batch_size=32,window_size=50,dataset_style="node"):
    scalers_seq = []
    if dataset_style == "node":
        stand_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/zzm/45.bin',startpoint=0,endpoint=None,data_png_name='00.png')#静止站立  3000
        stand1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/zzm/46.bin',startpoint=0,endpoint=None,data_png_name='01.png')#静止站立label=0
        level_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/zzm/49.bin',startpoint=2000,endpoint=7000,data_png_name='10.png')#平地行走,label=1
        level1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/zzm/50.bin',startpoint=500,endpoint=5000,data_png_name='11.png')#平地行走,label=1
        upramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/zzm/51.bin',startpoint=500,endpoint=3500,data_png_name='20.png')#上坡,label=2
        upramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/zzm/53.bin',startpoint=500,endpoint=3500,data_png_name='21.png')#上坡,label=2
        downramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/zzm/52.bin',startpoint=500,endpoint=3500,data_png_name='30.png')#下坡,label=3
        downramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/zzm/55.bin',startpoint=500,endpoint=3500,data_png_name='31.png')#下坡,label=3
        upstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/zzm/63.bin',startpoint=500,endpoint=4000,data_png_name='40.png')#上楼梯,label=4
        downstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/zzm/64.bin',startpoint=2500,endpoint=6000,data_png_name='50.png')#下楼梯,label=5
        scalers_seq = [1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5] # 归一化
    else:
        stand_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/zzm/45.bin',startpoint=0,endpoint=None,data_png_name='00.png')#静止站立  3000
        stand1_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/zzm/46.bin',startpoint=0,endpoint=None,data_png_name='01.png')#静止站立label=0
        level_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/zzm/49.bin',startpoint=2000,endpoint=7000,data_png_name='10.png')#平地行走,label=1
        level1_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/zzm/50.bin',startpoint=500,endpoint=5000,data_png_name='11.png')#平地行走,label=1
        upramp_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/zzm/51.bin',startpoint=500,endpoint=3500,data_png_name='20.png')#上坡,label=2
        upramp1_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/zzm/53.bin',startpoint=500,endpoint=3500,data_png_name='21.png')#上坡,label=2
        downramp_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/zzm/52.bin',startpoint=500,endpoint=3500,data_png_name='30.png')#下坡,label=3
        downramp1_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/zzm/55.bin',startpoint=500,endpoint=3500,data_png_name='31.png')#下坡,label=3
        upstairs_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/zzm/63.bin',startpoint=500,endpoint=4000,data_png_name='40.png')#上楼梯,label=4
        downstairs_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/zzm/64.bin',startpoint=2500,endpoint=6000,data_png_name='50.png')#下楼梯,label=5
        scalers_seq = [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5] # 归一化

    dataset_all = np.concatenate((stand_raw,stand1_raw,
                                     level_raw,level1_raw,
                                     upramp_raw,upramp1_raw,
                                     downramp_raw,downramp1_raw,
                                     upstairs_raw,
                                     downstairs_raw,
                                     ),axis=0)
    # #注意，相同物理量纲的数据，采用共享的归一化参数
    # #应该对训练集进行归一化
    if os.path.exists("data_scaler.save"):
        scaler = datasetphysicalSharedMaxMinScalers(feature_range=(-1, 1),scalers_seq=scalers_seq,loaded_scaler_file="data_scaler.save")
        print("加载归一化文件")
    else:
        print("data_scaler.save文件未找到")
        scaler = datasetphysicalSharedMaxMinScalers(feature_range=(-1, 1),scalers_seq=scalers_seq)
    scaler.fit(dataset_all) # 计算数据集最大最小值
    joblib.dump(scaler, "data_scaler.save")# 方法1：使用 joblib (推荐) 或者使用 pickle # import pickle
    print("[归一化参数]：最小值", scaler.data_min_)  # 对于 MinMaxScaler
    print("[归一化参数]：最大值:", scaler.data_max_)  # 对于 MinMaxScaler

    train_data_stand,train_label_stand,validate_data_stand,validate_label_stand,test_data_stand,test_label_stand = extract_window_data(dataset=stand_raw,window_size=window_size,label=0)
    train_data_stand1,train_label_stand1,validate_data_stand1,validate_label_stand1,test_data_stand1,test_label_stand1 = extract_window_data(dataset=stand1_raw,window_size=window_size,label=0)
    train_data_level,train_label_level,validate_data_level,validate_label_level,test_data_level,test_label_level = extract_window_data(dataset=level_raw,window_size=window_size,label=1)
    train_data_level1,train_label_level1,validate_data_level1,validate_label_level1,test_data_level1,test_label_level1 = extract_window_data(dataset=level1_raw,window_size=window_size,label=1)
    train_data_upramp,train_label_upramp,validate_data_upramp,validate_label_upramp,test_data_upramp,test_label_upramp = extract_window_data(dataset=upramp_raw,window_size=window_size,label=2)
    train_data_upramp1,train_label_upramp1,validate_data_upramp1,validate_label_upramp1,test_data_upramp1,test_label_upramp1 = extract_window_data(dataset=upramp1_raw,window_size=window_size,label=2)
    train_data_downramp,train_label_downramp,validate_data_downramp,validate_label_downramp,test_data_downramp,test_label_downramp = extract_window_data(dataset=downramp_raw,window_size=window_size,label=3)
    train_data_downramp1,train_label_downramp1,validate_data_downramp1,validate_label_downramp1,test_data_downramp1,test_label_downramp1 = extract_window_data(dataset=downramp1_raw,window_size=window_size,label=3)
    train_data_upstairs,train_label_upstairs,validate_data_upstairs,validate_label_upstairs,test_data_upstairs,test_label_upstairs = extract_window_data(dataset=upstairs_raw,window_size=window_size,label=4)
    train_data_downstairs,train_label_downstairs,validate_data_downstairs,validate_label_downstairs,test_data_downstairs,test_label_downstairs = extract_window_data(dataset=downstairs_raw,window_size=window_size,label=5)

    train_data_raw = np.concatenate((train_data_stand,train_data_stand1,
                                     train_data_level,train_data_level1,
                                     train_data_upramp,train_data_upramp1,
                                     train_data_downramp,train_data_downramp1,
                                     train_data_upstairs,
                                     train_data_downstairs,
                                     ),axis=0)
    train_label_raw = np.concatenate((train_label_stand,train_label_stand1,
                                     train_label_level,train_label_level1,
                                     train_label_upramp,train_label_upramp1,
                                     train_label_downramp,train_label_downramp1,
                                     train_label_upstairs,
                                     train_label_downstairs,
                                     ),axis=0)
    
    validate_data_raw = np.concatenate((validate_data_stand,validate_data_stand1,
                                     validate_data_level,validate_data_level1,
                                     validate_data_upramp,validate_data_upramp1,
                                     validate_data_downramp,validate_data_downramp1,
                                     validate_data_upstairs,
                                     validate_data_downstairs,
                                     ),axis=0)
    validate_label_raw = np.concatenate((validate_label_stand,validate_label_stand1,
                                     validate_label_level,validate_label_level1,
                                     validate_label_upramp,validate_label_upramp1,
                                     validate_label_downramp,validate_label_downramp1,
                                     validate_label_upstairs,
                                     validate_label_downstairs,
                                     ),axis=0)

    test_data_raw = np.concatenate((test_data_stand,test_data_stand1,
                                     test_data_level,test_data_level1,
                                     test_data_upramp,test_data_upramp1,
                                     test_data_downramp,test_data_downramp1,
                                     test_data_upstairs,
                                     test_data_downstairs,
                                     ),axis=0)
    test_label_raw = np.concatenate((test_label_stand,test_label_stand1,
                                     test_label_level,test_label_level1,
                                     test_label_upramp,test_label_upramp1,
                                     test_label_downramp,test_label_downramp1,
                                     test_label_upstairs,
                                     test_label_downstairs,
                                     ),axis=0)
    
    # 应用标准化
    train_data_norm = np.zeros(train_data_raw.shape)
    for i in range(train_data_raw.shape[0]):
        train_data_norm[i] = scaler.transform(train_data_raw[i])
    validate_data_norm = np.zeros(validate_data_raw.shape)
    for i in range(validate_data_raw.shape[0]):
        validate_data_norm[i] = scaler.transform(validate_data_raw[i])
    test_data_norm = np.zeros(test_data_raw.shape)
    for i in range(test_data_raw.shape[0]):
        test_data_norm[i] = scaler.transform(test_data_raw[i])

    return train_data_norm,train_label_raw,validate_data_norm,validate_label_raw,test_data_norm,test_label_raw


def load_dataset_lyl_train_valid_test(batch_size=32,window_size=50,dataset_style="node"):
    scalers_seq = []
    if dataset_style == "node":
        stand_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/lyl/3.bin',startpoint=0,endpoint=None,data_png_name='00.png')#静止站立  3000
        stand1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/lyl/4.bin',startpoint=0,endpoint=None,data_png_name='01.png')#静止站立label=0
        level_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/lyl/5.bin',startpoint=500,endpoint=4500,data_png_name='10.png')#平地行走,label=1
        level1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/lyl/6.bin',startpoint=500,endpoint=4500,data_png_name='11.png')#平地行走,label=1
        upramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/lyl/7.bin',startpoint=500,endpoint=3500,data_png_name='20.png')#上坡,label=2
        upramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/lyl/9.bin',startpoint=500,endpoint=3500,data_png_name='21.png')#上坡,label=2
        downramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/lyl/8.bin',startpoint=500,endpoint=3500,data_png_name='30.png')#下坡,label=3
        downramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/lyl/10.bin',startpoint=500,endpoint=3500,data_png_name='31.png')#下坡,label=3
        upstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/lyl/11.bin',startpoint=1000,endpoint=5000,data_png_name='40.png')#上楼梯,label=4
        downstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/lyl/12.bin',startpoint=1000,endpoint=5000,data_png_name='50.png')#下楼梯,label=5
        scalers_seq = [1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5] # 归一化
    else:
        stand_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/lyl/3.bin',startpoint=0,endpoint=None,data_png_name='00.png')#静止站立  3000
        stand1_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/lyl/4.bin',startpoint=0,endpoint=None,data_png_name='01.png')#静止站立label=0
        level_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/lyl/5.bin',startpoint=500,endpoint=4500,data_png_name='10.png')#平地行走,label=1
        level1_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/lyl/6.bin',startpoint=500,endpoint=4500,data_png_name='11.png')#平地行走,label=1
        upramp_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/lyl/7.bin',startpoint=500,endpoint=3500,data_png_name='20.png')#上坡,label=2
        upramp1_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/lyl/9.bin',startpoint=500,endpoint=3500,data_png_name='21.png')#上坡,label=2
        downramp_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/lyl/8.bin',startpoint=500,endpoint=3500,data_png_name='30.png')#下坡,label=3
        downramp1_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/lyl/10.bin',startpoint=500,endpoint=3500,data_png_name='31.png')#下坡,label=3
        upstairs_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/lyl/11.bin',startpoint=1000,endpoint=5000,data_png_name='40.png')#上楼梯,label=4
        downstairs_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/lyl/12.bin',startpoint=1000,endpoint=5000,data_png_name='50.png')#下楼梯,label=5
        scalers_seq = [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5] # 归一化

    dataset_all = np.concatenate((stand_raw,stand1_raw,#stand2_raw,
                                     level_raw,level1_raw,
                                     upramp_raw,upramp1_raw,
                                     downramp_raw,downramp1_raw,
                                     upstairs_raw,
                                     downstairs_raw,#downstairs1_raw,downstairs2_raw,downstairs3_raw,downstairs4_raw,downstairs5_raw,downstairs6_raw,downstairs7_raw
                                     ),axis=0)
    
    # #注意，相同物理量纲的数据，采用共享的归一化参数
    # #应该对训练集进行归一化
    if os.path.exists("data_scaler.save"):
        scaler = datasetphysicalSharedMaxMinScalers(feature_range=(-1, 1),scalers_seq=scalers_seq,loaded_scaler_file="data_scaler.save")
        print("加载归一化文件")
    else:
        print("data_scaler.save文件未找到")
        scaler = datasetphysicalSharedMaxMinScalers(feature_range=(-1, 1),scalers_seq=scalers_seq)
    scaler.fit(dataset_all) # 计算数据集最大最小值
    joblib.dump(scaler, "data_scaler.save")# 方法1：使用 joblib (推荐) 或者使用 pickle # import pickle
    print("[归一化参数]：最小值", scaler.data_min_)  # 对于 MinMaxScaler
    print("[归一化参数]：最大值:", scaler.data_max_)  # 对于 MinMaxScaler

    train_data_stand,train_label_stand,validate_data_stand,validate_label_stand,test_data_stand,test_label_stand = extract_window_data(dataset=stand_raw,window_size=window_size,label=0)
    train_data_stand1,train_label_stand1,validate_data_stand1,validate_label_stand1,test_data_stand1,test_label_stand1 = extract_window_data(dataset=stand1_raw,window_size=window_size,label=0)
    train_data_level,train_label_level,validate_data_level,validate_label_level,test_data_level,test_label_level = extract_window_data(dataset=level_raw,window_size=window_size,label=1)
    train_data_level1,train_label_level1,validate_data_level1,validate_label_level1,test_data_level1,test_label_level1 = extract_window_data(dataset=level1_raw,window_size=window_size,label=1)
    train_data_upramp,train_label_upramp,validate_data_upramp,validate_label_upramp,test_data_upramp,test_label_upramp = extract_window_data(dataset=upramp_raw,window_size=window_size,label=2)
    train_data_upramp1,train_label_upramp1,validate_data_upramp1,validate_label_upramp1,test_data_upramp1,test_label_upramp1 = extract_window_data(dataset=upramp1_raw,window_size=window_size,label=2)
    train_data_downramp,train_label_downramp,validate_data_downramp,validate_label_downramp,test_data_downramp,test_label_downramp = extract_window_data(dataset=downramp_raw,window_size=window_size,label=3)
    train_data_downramp1,train_label_downramp1,validate_data_downramp1,validate_label_downramp1,test_data_downramp1,test_label_downramp1 = extract_window_data(dataset=downramp1_raw,window_size=window_size,label=3)
    train_data_upstairs,train_label_upstairs,validate_data_upstairs,validate_label_upstairs,test_data_upstairs,test_label_upstairs = extract_window_data(dataset=upstairs_raw,window_size=window_size,label=4)
    train_data_downstairs,train_label_downstairs,validate_data_downstairs,validate_label_downstairs,test_data_downstairs,test_label_downstairs = extract_window_data(dataset=downstairs_raw,window_size=window_size,label=5)
    
    train_data_raw = np.concatenate((train_data_stand,train_data_stand1,
                                     train_data_level,train_data_level1,
                                     train_data_upramp,train_data_upramp1,
                                     train_data_downramp,train_data_downramp1,
                                     train_data_upstairs,
                                     train_data_downstairs,
                                     ),axis=0)
    train_label_raw = np.concatenate((train_label_stand,train_label_stand1,
                                     train_label_level,train_label_level1,
                                     train_label_upramp,train_label_upramp1,
                                     train_label_downramp,train_label_downramp1,
                                     train_label_upstairs,
                                     train_label_downstairs,
                                     ),axis=0)
    
    validate_data_raw = np.concatenate((validate_data_stand,validate_data_stand1,
                                     validate_data_level,validate_data_level1,
                                     validate_data_upramp,validate_data_upramp1,
                                     validate_data_downramp,validate_data_downramp1,
                                     validate_data_upstairs,
                                     validate_data_downstairs,
                                     ),axis=0)
    validate_label_raw = np.concatenate((validate_label_stand,validate_label_stand1,
                                     validate_label_level,validate_label_level1,
                                     validate_label_upramp,validate_label_upramp1,
                                     validate_label_downramp,validate_label_downramp1,
                                     validate_label_upstairs,
                                     validate_label_downstairs,
                                     ),axis=0)

    test_data_raw = np.concatenate((test_data_stand,test_data_stand1,
                                     test_data_level,test_data_level1,
                                     test_data_upramp,test_data_upramp1,
                                     test_data_downramp,test_data_downramp1,
                                     test_data_upstairs,
                                     test_data_downstairs,
                                     ),axis=0)
    test_label_raw = np.concatenate((test_label_stand,test_label_stand1,
                                     test_label_level,test_label_level1,
                                     test_label_upramp,test_label_upramp1,
                                     test_label_downramp,test_label_downramp1,
                                     test_label_upstairs,
                                     test_label_downstairs,
                                     ),axis=0)
    
    # 应用标准化
    train_data_norm = np.zeros(train_data_raw.shape)
    for i in range(train_data_raw.shape[0]):
        train_data_norm[i] = scaler.transform(train_data_raw[i])
    validate_data_norm = np.zeros(validate_data_raw.shape)
    for i in range(validate_data_raw.shape[0]):
        validate_data_norm[i] = scaler.transform(validate_data_raw[i])
    test_data_norm = np.zeros(test_data_raw.shape)
    for i in range(test_data_raw.shape[0]):
        test_data_norm[i] = scaler.transform(test_data_raw[i])

    return train_data_norm,train_label_raw,validate_data_norm,validate_label_raw,test_data_norm,test_label_raw


def load_dataset_szh_train_valid_test(batch_size=32,window_size=50,dataset_style="node"):
    scalers_seq = []
    if dataset_style=="node":
        stand_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/szh/13.bin',startpoint=0,endpoint=None,data_png_name='00.png')#静止站立  3000
        stand1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/szh/14.bin',startpoint=0,endpoint=None,data_png_name='01.png')#静止站立label=0
        level_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/szh/16.bin',startpoint=500,endpoint=4500,data_png_name='10.png')#平地行走,label=1
        level1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/szh/17.bin',startpoint=500,endpoint=4500,data_png_name='11.png')#平地行走,label=1
        upramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/szh/18.bin',startpoint=500,endpoint=3500,data_png_name='20.png')#上坡,label=2
        upramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/szh/20.bin',startpoint=500,endpoint=3500,data_png_name='21.png')#上坡,label=2
        downramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/szh/19.bin',startpoint=500,endpoint=3500,data_png_name='30.png')#下坡,label=3
        downramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/szh/21.bin',startpoint=500,endpoint=3500,data_png_name='31.png')#下坡,label=3
        upstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/szh/22.bin',startpoint=1000,endpoint=5000,data_png_name='40.png')#上楼梯,label=4
        downstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/szh/23.bin',startpoint=1000,endpoint=5000,data_png_name='50.png')#下楼梯,label=5
        scalers_seq = [1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5] # 归一化
    else:
        stand_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/szh/13.bin',startpoint=0,endpoint=None,data_png_name='00.png')#静止站立  3000
        stand1_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/szh/14.bin',startpoint=0,endpoint=None,data_png_name='01.png')#静止站立label=0
        level_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/szh/16.bin',startpoint=500,endpoint=4500,data_png_name='10.png')#平地行走,label=1
        level1_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/szh/17.bin',startpoint=500,endpoint=4500,data_png_name='11.png')#平地行走,label=1
        upramp_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/szh/18.bin',startpoint=500,endpoint=3500,data_png_name='20.png')#上坡,label=2
        upramp1_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/szh/20.bin',startpoint=500,endpoint=3500,data_png_name='21.png')#上坡,label=2
        downramp_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/szh/19.bin',startpoint=500,endpoint=3500,data_png_name='30.png')#下坡,label=3
        downramp1_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/szh/21.bin',startpoint=500,endpoint=3500,data_png_name='31.png')#下坡,label=3
        upstairs_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/szh/22.bin',startpoint=1000,endpoint=5000,data_png_name='40.png')#上楼梯,label=4
        downstairs_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/szh/23.bin',startpoint=1000,endpoint=5000,data_png_name='50.png')#下楼梯,label=5
        scalers_seq = [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5] # 归一化

    dataset_all = np.concatenate((stand_raw,stand1_raw,
                                     level_raw,level1_raw,
                                     upramp_raw,upramp1_raw,
                                     downramp_raw,downramp1_raw,
                                     upstairs_raw,
                                     downstairs_raw,
                                     ),axis=0)
    # #注意，相同物理量纲的数据，采用共享的归一化参数
    # #应该对训练集进行归一化
    if os.path.exists("data_scaler.save"):
        scaler = datasetphysicalSharedMaxMinScalers(feature_range=(-1, 1),scalers_seq=scalers_seq,loaded_scaler_file="data_scaler.save")
        print("加载归一化文件")
    else:
        print("data_scaler.save文件未找到")
        scaler = datasetphysicalSharedMaxMinScalers(feature_range=(-1, 1),scalers_seq=scalers_seq)
    scaler.fit(dataset_all) # 计算数据集最大最小值
    joblib.dump(scaler, "data_scaler.save")# 方法1：使用 joblib (推荐) 或者使用 pickle # import pickle
    print("[归一化参数]：最小值", scaler.data_min_)  # 对于 MinMaxScaler
    print("[归一化参数]：最大值:", scaler.data_max_)  # 对于 MinMaxScaler

    train_data_stand,train_label_stand,validate_data_stand,validate_label_stand,test_data_stand,test_label_stand = extract_window_data(dataset=stand_raw,window_size=window_size,label=0)
    train_data_stand1,train_label_stand1,validate_data_stand1,validate_label_stand1,test_data_stand1,test_label_stand1 = extract_window_data(dataset=stand1_raw,window_size=window_size,label=0)
    train_data_level,train_label_level,validate_data_level,validate_label_level,test_data_level,test_label_level = extract_window_data(dataset=level_raw,window_size=window_size,label=1)
    train_data_level1,train_label_level1,validate_data_level1,validate_label_level1,test_data_level1,test_label_level1 = extract_window_data(dataset=level1_raw,window_size=window_size,label=1)
    train_data_upramp,train_label_upramp,validate_data_upramp,validate_label_upramp,test_data_upramp,test_label_upramp = extract_window_data(dataset=upramp_raw,window_size=window_size,label=2)
    train_data_upramp1,train_label_upramp1,validate_data_upramp1,validate_label_upramp1,test_data_upramp1,test_label_upramp1 = extract_window_data(dataset=upramp1_raw,window_size=window_size,label=2)
    train_data_downramp,train_label_downramp,validate_data_downramp,validate_label_downramp,test_data_downramp,test_label_downramp = extract_window_data(dataset=downramp_raw,window_size=window_size,label=3)
    train_data_downramp1,train_label_downramp1,validate_data_downramp1,validate_label_downramp1,test_data_downramp1,test_label_downramp1 = extract_window_data(dataset=downramp1_raw,window_size=window_size,label=3)
    train_data_upstairs,train_label_upstairs,validate_data_upstairs,validate_label_upstairs,test_data_upstairs,test_label_upstairs = extract_window_data(dataset=upstairs_raw,window_size=window_size,label=4)
    train_data_downstairs,train_label_downstairs,validate_data_downstairs,validate_label_downstairs,test_data_downstairs,test_label_downstairs = extract_window_data(dataset=downstairs_raw,window_size=window_size,label=5)
    
    train_data_raw = np.concatenate((train_data_stand,train_data_stand1,
                                     train_data_level,train_data_level1,
                                     train_data_upramp,train_data_upramp1,
                                     train_data_downramp,train_data_downramp1,
                                     train_data_upstairs,
                                     train_data_downstairs,
                                     ),axis=0)
    train_label_raw = np.concatenate((train_label_stand,train_label_stand1,
                                     train_label_level,train_label_level1,
                                     train_label_upramp,train_label_upramp1,
                                     train_label_downramp,train_label_downramp1,
                                     train_label_upstairs,
                                     train_label_downstairs,
                                     ),axis=0)
    
    validate_data_raw = np.concatenate((validate_data_stand,validate_data_stand1,
                                     validate_data_level,validate_data_level1,
                                     validate_data_upramp,validate_data_upramp1,
                                     validate_data_downramp,validate_data_downramp1,
                                     validate_data_upstairs,
                                     validate_data_downstairs,
                                     ),axis=0)
    validate_label_raw = np.concatenate((validate_label_stand,validate_label_stand1,
                                     validate_label_level,validate_label_level1,
                                     validate_label_upramp,validate_label_upramp1,
                                     validate_label_downramp,validate_label_downramp1,
                                     validate_label_upstairs,
                                     validate_label_downstairs,
                                     ),axis=0)

    test_data_raw = np.concatenate((test_data_stand,test_data_stand1,
                                     test_data_level,test_data_level1,
                                     test_data_upramp,test_data_upramp1,
                                     test_data_downramp,test_data_downramp1,
                                     test_data_upstairs,
                                     test_data_downstairs,
                                     ),axis=0)
    test_label_raw = np.concatenate((test_label_stand,test_label_stand1,
                                     test_label_level,test_label_level1,
                                     test_label_upramp,test_label_upramp1,
                                     test_label_downramp,test_label_downramp1,
                                     test_label_upstairs,
                                     test_label_downstairs,
                                     ),axis=0)
    
    # 应用标准化
    train_data_norm = np.zeros(train_data_raw.shape)
    for i in range(train_data_raw.shape[0]):
        train_data_norm[i] = scaler.transform(train_data_raw[i])
    validate_data_norm = np.zeros(validate_data_raw.shape)
    for i in range(validate_data_raw.shape[0]):
        validate_data_norm[i] = scaler.transform(validate_data_raw[i])
    test_data_norm = np.zeros(test_data_raw.shape)
    for i in range(test_data_raw.shape[0]):
        test_data_norm[i] = scaler.transform(test_data_raw[i])

    return train_data_norm,train_label_raw,validate_data_norm,validate_label_raw,test_data_norm,test_label_raw


def load_dataset_kdl_train_valid_test(batch_size=32,window_size=50,dataset_style="node"):
    scalers_seq = []
    if dataset_style=="node":
        stand_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/kdl/34.bin',startpoint=0,endpoint=None,data_png_name='00.png')#静止站立  3000
        stand1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/kdl/35.bin',startpoint=0,endpoint=None,data_png_name='01.png')#静止站立label=0
        level_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/kdl/36.bin',startpoint=500,endpoint=4500,data_png_name='10.png')#平地行走,label=1
        level1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/kdl/37.bin',startpoint=500,endpoint=3500,data_png_name='11.png')#平地行走,label=1
        upramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/kdl/38.bin',startpoint=500,endpoint=3500,data_png_name='20.png')#上坡,label=2
        upramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/kdl/40.bin',startpoint=500,endpoint=3500,data_png_name='21.png')#上坡,label=2
        downramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/kdl/39.bin',startpoint=500,endpoint=3500,data_png_name='30.png')#下坡,label=3
        downramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/kdl/41.bin',startpoint=500,endpoint=3500,data_png_name='31.png')#下坡,label=3
        upstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/kdl/42.bin',startpoint=1000,endpoint=5000,data_png_name='40.png')#上楼梯,label=4
        downstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/kdl/43.bin',startpoint=1000,endpoint=5000,data_png_name='50.png')#下楼梯,label=5
        scalers_seq = [1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5] # 归一化
    else:
        stand_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/kdl/34.bin',startpoint=0,endpoint=None,data_png_name='00.png')#静止站立  3000
        stand1_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/kdl/35.bin',startpoint=0,endpoint=None,data_png_name='01.png')#静止站立label=0
        level_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/kdl/36.bin',startpoint=500,endpoint=4500,data_png_name='10.png')#平地行走,label=1
        level1_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/kdl/37.bin',startpoint=500,endpoint=3500,data_png_name='11.png')#平地行走,label=1
        upramp_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/kdl/38.bin',startpoint=500,endpoint=3500,data_png_name='20.png')#上坡,label=2
        upramp1_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/kdl/40.bin',startpoint=500,endpoint=3500,data_png_name='21.png')#上坡,label=2
        downramp_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/kdl/39.bin',startpoint=500,endpoint=3500,data_png_name='30.png')#下坡,label=3
        downramp1_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/kdl/41.bin',startpoint=500,endpoint=3500,data_png_name='31.png')#下坡,label=3
        upstairs_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/kdl/42.bin',startpoint=1000,endpoint=5000,data_png_name='40.png')#上楼梯,label=4
        downstairs_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/kdl/43.bin',startpoint=1000,endpoint=5000,data_png_name='50.png')#下楼梯,label=5
        scalers_seq = [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5] # 归一化

    dataset_all = np.concatenate((stand_raw,stand1_raw,
                                     level_raw,level1_raw,
                                     upramp_raw,upramp1_raw,
                                     downramp_raw,downramp1_raw,
                                     upstairs_raw,
                                     downstairs_raw,
                                     ),axis=0)
    # #注意，相同物理量纲的数据，采用共享的归一化参数
    # #应该对训练集进行归一化
    if os.path.exists("data_scaler.save"):
        scaler = datasetphysicalSharedMaxMinScalers(feature_range=(-1, 1),scalers_seq=scalers_seq,loaded_scaler_file="data_scaler.save")
        print("加载归一化文件")
    else:
        print("data_scaler.save文件未找到")
        scaler = datasetphysicalSharedMaxMinScalers(feature_range=(-1, 1),scalers_seq=scalers_seq)
    scaler.fit(dataset_all) # 计算数据集最大最小值
    joblib.dump(scaler, "data_scaler.save")# 方法1：使用 joblib (推荐) 或者使用 pickle # import pickle
    print("[归一化参数]：最小值", scaler.data_min_)  # 对于 MinMaxScaler
    print("[归一化参数]：最大值:", scaler.data_max_)  # 对于 MinMaxScaler

    train_data_stand,train_label_stand,validate_data_stand,validate_label_stand,test_data_stand,test_label_stand = extract_window_data(dataset=stand_raw,window_size=window_size,label=0)
    train_data_stand1,train_label_stand1,validate_data_stand1,validate_label_stand1,test_data_stand1,test_label_stand1 = extract_window_data(dataset=stand1_raw,window_size=window_size,label=0)
    train_data_level,train_label_level,validate_data_level,validate_label_level,test_data_level,test_label_level = extract_window_data(dataset=level_raw,window_size=window_size,label=1)
    train_data_level1,train_label_level1,validate_data_level1,validate_label_level1,test_data_level1,test_label_level1 = extract_window_data(dataset=level1_raw,window_size=window_size,label=1)
    train_data_upramp,train_label_upramp,validate_data_upramp,validate_label_upramp,test_data_upramp,test_label_upramp = extract_window_data(dataset=upramp_raw,window_size=window_size,label=2)
    train_data_upramp1,train_label_upramp1,validate_data_upramp1,validate_label_upramp1,test_data_upramp1,test_label_upramp1 = extract_window_data(dataset=upramp1_raw,window_size=window_size,label=2)
    train_data_downramp,train_label_downramp,validate_data_downramp,validate_label_downramp,test_data_downramp,test_label_downramp = extract_window_data(dataset=downramp_raw,window_size=window_size,label=3)
    train_data_downramp1,train_label_downramp1,validate_data_downramp1,validate_label_downramp1,test_data_downramp1,test_label_downramp1 = extract_window_data(dataset=downramp1_raw,window_size=window_size,label=3)
    train_data_upstairs,train_label_upstairs,validate_data_upstairs,validate_label_upstairs,test_data_upstairs,test_label_upstairs = extract_window_data(dataset=upstairs_raw,window_size=window_size,label=4)
    train_data_downstairs,train_label_downstairs,validate_data_downstairs,validate_label_downstairs,test_data_downstairs,test_label_downstairs = extract_window_data(dataset=downstairs_raw,window_size=window_size,label=5)

    train_data_raw = np.concatenate((train_data_stand,train_data_stand1,
                                     train_data_level,train_data_level1,
                                     train_data_upramp,train_data_upramp1,
                                     train_data_downramp,train_data_downramp1,
                                     train_data_upstairs,
                                     train_data_downstairs,
                                     ),axis=0)
    train_label_raw = np.concatenate((train_label_stand,train_label_stand1,
                                     train_label_level,train_label_level1,
                                     train_label_upramp,train_label_upramp1,
                                     train_label_downramp,train_label_downramp1,
                                     train_label_upstairs,
                                     train_label_downstairs,
                                     ),axis=0)
    
    validate_data_raw = np.concatenate((validate_data_stand,validate_data_stand1,
                                     validate_data_level,validate_data_level1,
                                     validate_data_upramp,validate_data_upramp1,
                                     validate_data_downramp,validate_data_downramp1,
                                     validate_data_upstairs,
                                     validate_data_downstairs,
                                     ),axis=0)
    validate_label_raw = np.concatenate((validate_label_stand,validate_label_stand1,
                                     validate_label_level,validate_label_level1,
                                     validate_label_upramp,validate_label_upramp1,
                                     validate_label_downramp,validate_label_downramp1,
                                     validate_label_upstairs,
                                     validate_label_downstairs,
                                     ),axis=0)

    test_data_raw = np.concatenate((test_data_stand,test_data_stand1,
                                     test_data_level,test_data_level1,
                                     test_data_upramp,test_data_upramp1,
                                     test_data_downramp,test_data_downramp1,
                                     test_data_upstairs,
                                     test_data_downstairs,
                                     ),axis=0)
    test_label_raw = np.concatenate((test_label_stand,test_label_stand1,
                                     test_label_level,test_label_level1,
                                     test_label_upramp,test_label_upramp1,
                                     test_label_downramp,test_label_downramp1,
                                     test_label_upstairs,
                                     test_label_downstairs,
                                     ),axis=0)
    
    # 应用标准化
    train_data_norm = np.zeros(train_data_raw.shape)
    for i in range(train_data_raw.shape[0]):
        train_data_norm[i] = scaler.transform(train_data_raw[i])
    validate_data_norm = np.zeros(validate_data_raw.shape)
    for i in range(validate_data_raw.shape[0]):
        validate_data_norm[i] = scaler.transform(validate_data_raw[i])
    test_data_norm = np.zeros(test_data_raw.shape)
    for i in range(test_data_raw.shape[0]):
        test_data_norm[i] = scaler.transform(test_data_raw[i])

    return train_data_norm,train_label_raw,validate_data_norm,validate_label_raw,test_data_norm,test_label_raw


def load_dataset_wdh_train_valid_test(batch_size=32,window_size=50,dataset_style="node"):
    scalers_seq = []
    if dataset_style == "node":
        stand_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/wdh/209.bin',startpoint=0,endpoint=None,data_png_name='00.png')#静止站立  3000
        stand1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/wdh/210.bin',startpoint=0,endpoint=None,data_png_name='01.png')#静止站立label=0
        level_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/wdh//211.bin',startpoint=500,endpoint=4500,data_png_name='10.png')#平地行走,label=1
        level1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/wdh//212.bin',startpoint=500,endpoint=4500,data_png_name='11.png')#平地行走,label=1
        upramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/wdh//220.bin',startpoint=500,endpoint=3500,data_png_name='20.png')#上坡,label=2
        upramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/wdh//222.bin',startpoint=500,endpoint=3500,data_png_name='21.png')#上坡,label=2
        downramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/wdh//221.bin',startpoint=500,endpoint=3500,data_png_name='30.png')#下坡,label=3
        downramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/wdh//223.bin',startpoint=500,endpoint=3500,data_png_name='31.png')#下坡,label=3
        upstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/wdh//218.bin',startpoint=1000,endpoint=5000,data_png_name='40.png')#上楼梯,label=4
        downstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/wdh//219.bin',startpoint=1000,endpoint=5000,data_png_name='50.png')#下楼梯,label=5
        scalers_seq = [1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5] # 归一化
    else:
        stand_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/wdh/209.bin',startpoint=0,endpoint=None,data_png_name='00.png')#静止站立  3000
        stand1_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/wdh/210.bin',startpoint=0,endpoint=None,data_png_name='01.png')#静止站立label=0
        level_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/wdh//211.bin',startpoint=500,endpoint=4500,data_png_name='10.png')#平地行走,label=1
        level1_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/wdh//212.bin',startpoint=500,endpoint=4500,data_png_name='11.png')#平地行走,label=1
        upramp_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/wdh//220.bin',startpoint=500,endpoint=3500,data_png_name='20.png')#上坡,label=2
        upramp1_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/wdh//222.bin',startpoint=500,endpoint=3500,data_png_name='21.png')#上坡,label=2
        downramp_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/wdh//221.bin',startpoint=500,endpoint=3500,data_png_name='30.png')#下坡,label=3
        downramp1_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/wdh//223.bin',startpoint=500,endpoint=3500,data_png_name='31.png')#下坡,label=3
        upstairs_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/wdh//218.bin',startpoint=1000,endpoint=5000,data_png_name='40.png')#上楼梯,label=4
        downstairs_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/wdh//219.bin',startpoint=1000,endpoint=5000,data_png_name='50.png')#下楼梯,label=5
        scalers_seq = [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5] # 归一化

    dataset_all = np.concatenate((stand_raw,stand1_raw,
                                     level_raw,level1_raw,
                                     upramp_raw,upramp1_raw,
                                     downramp_raw,downramp1_raw,
                                     upstairs_raw,
                                     downstairs_raw,
                                     ),axis=0)
    # #注意，相同物理量纲的数据，采用共享的归一化参数
    # #应该对训练集进行归一化
    if os.path.exists("data_scaler.save"):
        scaler = datasetphysicalSharedMaxMinScalers(feature_range=(-1, 1),scalers_seq=scalers_seq,loaded_scaler_file="data_scaler.save")
        print("加载归一化文件")
    else:
        print("data_scaler.save文件未找到")
        scaler = datasetphysicalSharedMaxMinScalers(feature_range=(-1, 1),scalers_seq=scalers_seq)
    scaler.fit(dataset_all) # 计算数据集最大最小值
    joblib.dump(scaler, "data_scaler.save")# 方法1：使用 joblib (推荐) 或者使用 pickle # import pickle
    print("[归一化参数]：最小值", scaler.data_min_)  # 对于 MinMaxScaler
    print("[归一化参数]：最大值:", scaler.data_max_)  # 对于 MinMaxScaler

    train_data_stand,train_label_stand,validate_data_stand,validate_label_stand,test_data_stand,test_label_stand = extract_window_data(dataset=stand_raw,window_size=window_size,label=0)
    train_data_stand1,train_label_stand1,validate_data_stand1,validate_label_stand1,test_data_stand1,test_label_stand1 = extract_window_data(dataset=stand1_raw,window_size=window_size,label=0)
    train_data_level,train_label_level,validate_data_level,validate_label_level,test_data_level,test_label_level = extract_window_data(dataset=level_raw,window_size=window_size,label=1)
    train_data_level1,train_label_level1,validate_data_level1,validate_label_level1,test_data_level1,test_label_level1 = extract_window_data(dataset=level1_raw,window_size=window_size,label=1)
    train_data_upramp,train_label_upramp,validate_data_upramp,validate_label_upramp,test_data_upramp,test_label_upramp = extract_window_data(dataset=upramp_raw,window_size=window_size,label=2)
    train_data_upramp1,train_label_upramp1,validate_data_upramp1,validate_label_upramp1,test_data_upramp1,test_label_upramp1 = extract_window_data(dataset=upramp1_raw,window_size=window_size,label=2)
    train_data_downramp,train_label_downramp,validate_data_downramp,validate_label_downramp,test_data_downramp,test_label_downramp = extract_window_data(dataset=downramp_raw,window_size=window_size,label=3)
    train_data_downramp1,train_label_downramp1,validate_data_downramp1,validate_label_downramp1,test_data_downramp1,test_label_downramp1 = extract_window_data(dataset=downramp1_raw,window_size=window_size,label=3)
    train_data_upstairs,train_label_upstairs,validate_data_upstairs,validate_label_upstairs,test_data_upstairs,test_label_upstairs = extract_window_data(dataset=upstairs_raw,window_size=window_size,label=4)
    train_data_downstairs,train_label_downstairs,validate_data_downstairs,validate_label_downstairs,test_data_downstairs,test_label_downstairs = extract_window_data(dataset=downstairs_raw,window_size=window_size,label=5)

    train_data_raw = np.concatenate((train_data_stand,train_data_stand1,
                                     train_data_level,train_data_level1,
                                     train_data_upramp,train_data_upramp1,
                                     train_data_downramp,train_data_downramp1,
                                     train_data_upstairs,
                                     train_data_downstairs,
                                     ),axis=0)
    train_label_raw = np.concatenate((train_label_stand,train_label_stand1,
                                     train_label_level,train_label_level1,
                                     train_label_upramp,train_label_upramp1,
                                     train_label_downramp,train_label_downramp1,
                                     train_label_upstairs,
                                     train_label_downstairs,
                                     ),axis=0)
    
    validate_data_raw = np.concatenate((validate_data_stand,validate_data_stand1,
                                     validate_data_level,validate_data_level1,
                                     validate_data_upramp,validate_data_upramp1,
                                     validate_data_downramp,validate_data_downramp1,
                                     validate_data_upstairs,
                                     validate_data_downstairs,
                                     ),axis=0)
    validate_label_raw = np.concatenate((validate_label_stand,validate_label_stand1,
                                     validate_label_level,validate_label_level1,
                                     validate_label_upramp,validate_label_upramp1,
                                     validate_label_downramp,validate_label_downramp1,
                                     validate_label_upstairs,
                                     validate_label_downstairs,
                                     ),axis=0)

    test_data_raw = np.concatenate((test_data_stand,test_data_stand1,
                                     test_data_level,test_data_level1,
                                     test_data_upramp,test_data_upramp1,
                                     test_data_downramp,test_data_downramp1,
                                     test_data_upstairs,
                                     test_data_downstairs,
                                     ),axis=0)
    test_label_raw = np.concatenate((test_label_stand,test_label_stand1,
                                     test_label_level,test_label_level1,
                                     test_label_upramp,test_label_upramp1,
                                     test_label_downramp,test_label_downramp1,
                                     test_label_upstairs,
                                     test_label_downstairs,
                                     ),axis=0)
    
    # 应用标准化
    train_data_norm = np.zeros(train_data_raw.shape)
    for i in range(train_data_raw.shape[0]):
        train_data_norm[i] = scaler.transform(train_data_raw[i])
    validate_data_norm = np.zeros(validate_data_raw.shape)
    for i in range(validate_data_raw.shape[0]):
        validate_data_norm[i] = scaler.transform(validate_data_raw[i])
    test_data_norm = np.zeros(test_data_raw.shape)
    for i in range(test_data_raw.shape[0]):
        test_data_norm[i] = scaler.transform(test_data_raw[i])

    return train_data_norm,train_label_raw,validate_data_norm,validate_label_raw,test_data_norm,test_label_raw


def load_dataset_ly_train_valid_test(batch_size=32,window_size=50,dataset_style="node"):
    scalers_seq = []
    if dataset_style == "node":
        stand_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/ly/224.bin',startpoint=0,endpoint=None,data_png_name='00.png')#静止站立  3000
        stand1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/ly/225.bin',startpoint=0,endpoint=None,data_png_name='01.png')#静止站立label=0
        level_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/ly/226.bin',startpoint=500,endpoint=4500,data_png_name='10.png')#平地行走,label=1
        level1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/ly/227.bin',startpoint=500,endpoint=4500,data_png_name='11.png')#平地行走,label=1
        upramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/ly/228.bin',startpoint=500,endpoint=3000,data_png_name='20.png')#上坡,label=2
        upramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/ly/230.bin',startpoint=500,endpoint=3000,data_png_name='21.png')#上坡,label=2
        downramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/ly/229.bin',startpoint=500,endpoint=3000,data_png_name='30.png')#下坡,label=3
        downramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/ly/231.bin',startpoint=500,endpoint=3000,data_png_name='31.png')#下坡,label=3
        upstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/ly/232.bin',startpoint=1000,endpoint=5000,data_png_name='40.png')#上楼梯,label=4
        downstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/ly/233.bin',startpoint=1000,endpoint=5000,data_png_name='50.png')#下楼梯,label=5
        scalers_seq = [1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5] # 归一化
    else:
        stand_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/ly/224.bin',startpoint=0,endpoint=None,data_png_name='00.png')#静止站立  3000
        stand1_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/ly/225.bin',startpoint=0,endpoint=None,data_png_name='01.png')#静止站立label=0
        level_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/ly/226.bin',startpoint=500,endpoint=4500,data_png_name='10.png')#平地行走,label=1
        level1_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/ly/227.bin',startpoint=500,endpoint=4500,data_png_name='11.png')#平地行走,label=1
        upramp_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/ly/228.bin',startpoint=500,endpoint=3000,data_png_name='20.png')#上坡,label=2
        upramp1_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/ly/230.bin',startpoint=500,endpoint=3000,data_png_name='21.png')#上坡,label=2
        downramp_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/ly/229.bin',startpoint=500,endpoint=3000,data_png_name='30.png')#下坡,label=3
        downramp1_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/ly/231.bin',startpoint=500,endpoint=3000,data_png_name='31.png')#下坡,label=3
        upstairs_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/ly/232.bin',startpoint=1000,endpoint=5000,data_png_name='40.png')#上楼梯,label=4
        downstairs_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/ly/233.bin',startpoint=1000,endpoint=5000,data_png_name='50.png')#下楼梯,label=5
        scalers_seq = [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5] # 归一化

    dataset_all = np.concatenate((stand_raw,stand1_raw,
                                     level_raw,level1_raw,
                                     upramp_raw,upramp1_raw,
                                     downramp_raw,downramp1_raw,
                                     upstairs_raw,
                                     downstairs_raw,
                                     ),axis=0)
    # #注意，相同物理量纲的数据，采用共享的归一化参数
    # #应该对训练集进行归一化
    if os.path.exists("data_scaler.save"):
        scaler = datasetphysicalSharedMaxMinScalers(feature_range=(-1, 1),scalers_seq=scalers_seq,loaded_scaler_file="data_scaler.save")
        print("加载归一化文件")
    else:
        print("data_scaler.save文件未找到")
        scaler = datasetphysicalSharedMaxMinScalers(feature_range=(-1, 1),scalers_seq=scalers_seq)
    scaler.fit(dataset_all) # 计算数据集最大最小值
    joblib.dump(scaler, "data_scaler.save")# 方法1：使用 joblib (推荐) 或者使用 pickle # import pickle
    print("[归一化参数]：最小值", scaler.data_min_)  # 对于 MinMaxScaler
    print("[归一化参数]：最大值:", scaler.data_max_)  # 对于 MinMaxScaler

    train_data_stand,train_label_stand,validate_data_stand,validate_label_stand,test_data_stand,test_label_stand = extract_window_data(dataset=stand_raw,window_size=window_size,label=0)
    train_data_stand1,train_label_stand1,validate_data_stand1,validate_label_stand1,test_data_stand1,test_label_stand1 = extract_window_data(dataset=stand1_raw,window_size=window_size,label=0)
    train_data_level,train_label_level,validate_data_level,validate_label_level,test_data_level,test_label_level = extract_window_data(dataset=level_raw,window_size=window_size,label=1)
    train_data_level1,train_label_level1,validate_data_level1,validate_label_level1,test_data_level1,test_label_level1 = extract_window_data(dataset=level1_raw,window_size=window_size,label=1)
    train_data_upramp,train_label_upramp,validate_data_upramp,validate_label_upramp,test_data_upramp,test_label_upramp = extract_window_data(dataset=upramp_raw,window_size=window_size,label=2)
    train_data_upramp1,train_label_upramp1,validate_data_upramp1,validate_label_upramp1,test_data_upramp1,test_label_upramp1 = extract_window_data(dataset=upramp1_raw,window_size=window_size,label=2)
    train_data_downramp,train_label_downramp,validate_data_downramp,validate_label_downramp,test_data_downramp,test_label_downramp = extract_window_data(dataset=downramp_raw,window_size=window_size,label=3)
    train_data_downramp1,train_label_downramp1,validate_data_downramp1,validate_label_downramp1,test_data_downramp1,test_label_downramp1 = extract_window_data(dataset=downramp1_raw,window_size=window_size,label=3)
    train_data_upstairs,train_label_upstairs,validate_data_upstairs,validate_label_upstairs,test_data_upstairs,test_label_upstairs = extract_window_data(dataset=upstairs_raw,window_size=window_size,label=4)
    train_data_downstairs,train_label_downstairs,validate_data_downstairs,validate_label_downstairs,test_data_downstairs,test_label_downstairs = extract_window_data(dataset=downstairs_raw,window_size=window_size,label=5)
    
    train_data_raw = np.concatenate((train_data_stand,train_data_stand1,
                                     train_data_level,train_data_level1,
                                     train_data_upramp,train_data_upramp1,
                                     train_data_downramp,train_data_downramp1,
                                     train_data_upstairs,
                                     train_data_downstairs,
                                     ),axis=0)
    train_label_raw = np.concatenate((train_label_stand,train_label_stand1,
                                     train_label_level,train_label_level1,
                                     train_label_upramp,train_label_upramp1,
                                     train_label_downramp,train_label_downramp1,
                                     train_label_upstairs,
                                     train_label_downstairs,
                                     ),axis=0)
    
    validate_data_raw = np.concatenate((validate_data_stand,validate_data_stand1,
                                     validate_data_level,validate_data_level1,
                                     validate_data_upramp,validate_data_upramp1,
                                     validate_data_downramp,validate_data_downramp1,
                                     validate_data_upstairs,
                                     validate_data_downstairs,
                                     ),axis=0)
    validate_label_raw = np.concatenate((validate_label_stand,validate_label_stand1,
                                     validate_label_level,validate_label_level1,
                                     validate_label_upramp,validate_label_upramp1,
                                     validate_label_downramp,validate_label_downramp1,
                                     validate_label_upstairs,
                                     validate_label_downstairs,
                                     ),axis=0)

    test_data_raw = np.concatenate((test_data_stand,test_data_stand1,
                                     test_data_level,test_data_level1,
                                     test_data_upramp,test_data_upramp1,
                                     test_data_downramp,test_data_downramp1,
                                     test_data_upstairs,
                                     test_data_downstairs,
                                     ),axis=0)
    test_label_raw = np.concatenate((test_label_stand,test_label_stand1,
                                     test_label_level,test_label_level1,
                                     test_label_upramp,test_label_upramp1,
                                     test_label_downramp,test_label_downramp1,
                                     test_label_upstairs,
                                     test_label_downstairs,
                                     ),axis=0)
    
    # 应用标准化
    train_data_norm = np.zeros(train_data_raw.shape)
    for i in range(train_data_raw.shape[0]):
        train_data_norm[i] = scaler.transform(train_data_raw[i])
    validate_data_norm = np.zeros(validate_data_raw.shape)
    for i in range(validate_data_raw.shape[0]):
        validate_data_norm[i] = scaler.transform(validate_data_raw[i])
    test_data_norm = np.zeros(test_data_raw.shape)
    for i in range(test_data_raw.shape[0]):
        test_data_norm[i] = scaler.transform(test_data_raw[i])

    return train_data_norm,train_label_raw,validate_data_norm,validate_label_raw,test_data_norm,test_label_raw


def load_dataset_lys_train_valid_test(batch_size=32,window_size=50,dataset_style="node"):
    scalers_seq = []
    if dataset_style == "node":
        stand_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/lys/75.bin',startpoint=0,endpoint=None,data_png_name='00.png')#静止站立  3000
        stand1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/lys/76.bin',startpoint=0,endpoint=None,data_png_name='01.png')#静止站立label=0
        level_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/lys/77.bin',startpoint=500,endpoint=4500,data_png_name='10.png')#平地行走,label=1
        level1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/lys/78.bin',startpoint=500,endpoint=4500,data_png_name='11.png')#平地行走,label=1
        upramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/lys/79.bin',startpoint=500,endpoint=3500,data_png_name='20.png')#上坡,label=2
        upramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/lys/81.bin',startpoint=500,endpoint=3500,data_png_name='21.png')#上坡,label=2
        downramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/lys/80.bin',startpoint=500,endpoint=3500,data_png_name='30.png')#下坡,label=3
        downramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/lys/82.bin',startpoint=500,endpoint=3500,data_png_name='31.png')#下坡,label=3
        upstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/lys/83.bin',startpoint=1000,endpoint=5000,data_png_name='40.png')#上楼梯,label=4
        downstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/lys/86.bin',startpoint=1000,endpoint=5000,data_png_name='50.png')#下楼梯,label=5
        scalers_seq = [1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5] # 归一化
    else:
        stand_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/lys/75.bin',startpoint=0,endpoint=None,data_png_name='00.png')#静止站立  3000
        stand1_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/lys/76.bin',startpoint=0,endpoint=None,data_png_name='01.png')#静止站立label=0
        level_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/lys/77.bin',startpoint=500,endpoint=4500,data_png_name='10.png')#平地行走,label=1
        level1_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/lys/78.bin',startpoint=500,endpoint=4500,data_png_name='11.png')#平地行走,label=1
        upramp_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/lys/79.bin',startpoint=500,endpoint=3500,data_png_name='20.png')#上坡,label=2
        upramp1_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/lys/81.bin',startpoint=500,endpoint=3500,data_png_name='21.png')#上坡,label=2
        downramp_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/lys/80.bin',startpoint=500,endpoint=3500,data_png_name='30.png')#下坡,label=3
        downramp1_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/lys/82.bin',startpoint=500,endpoint=3500,data_png_name='31.png')#下坡,label=3
        upstairs_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/lys/83.bin',startpoint=1000,endpoint=5000,data_png_name='40.png')#上楼梯,label=4
        downstairs_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/lys/86.bin',startpoint=1000,endpoint=5000,data_png_name='50.png')#下楼梯,label=5
        scalers_seq = [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5] # 归一化
 
    dataset_all = np.concatenate((stand_raw,stand1_raw,#stand2_raw,
                                     level_raw,level1_raw,
                                     upramp_raw,upramp1_raw,
                                     downramp_raw,downramp1_raw,
                                     upstairs_raw,
                                     downstairs_raw,#downstairs1_raw,downstairs2_raw,downstairs3_raw,downstairs4_raw,downstairs5_raw,downstairs6_raw,downstairs7_raw
                                     ),axis=0)
    
    # #注意，相同物理量纲的数据，采用共享的归一化参数
    # #应该对训练集进行归一化
    if os.path.exists("data_scaler.save"):
        scaler = datasetphysicalSharedMaxMinScalers(feature_range=(-1, 1),scalers_seq=scalers_seq,loaded_scaler_file="data_scaler.save")
        print("加载归一化文件")
    else:
        print("data_scaler.save文件未找到")
        scaler = datasetphysicalSharedMaxMinScalers(feature_range=(-1, 1),scalers_seq=scalers_seq)
    scaler.fit(dataset_all) # 计算数据集最大最小值
    joblib.dump(scaler, "data_scaler.save")# 方法1：使用 joblib (推荐) 或者使用 pickle # import pickle
    print("[归一化参数]：最小值", scaler.data_min_)  # 对于 MinMaxScaler
    print("[归一化参数]：最大值:", scaler.data_max_)  # 对于 MinMaxScaler

    train_data_stand,train_label_stand,validate_data_stand,validate_label_stand,test_data_stand,test_label_stand = extract_window_data(dataset=stand_raw,window_size=window_size,label=0)
    train_data_stand1,train_label_stand1,validate_data_stand1,validate_label_stand1,test_data_stand1,test_label_stand1 = extract_window_data(dataset=stand1_raw,window_size=window_size,label=0)
    train_data_level,train_label_level,validate_data_level,validate_label_level,test_data_level,test_label_level = extract_window_data(dataset=level_raw,window_size=window_size,label=1)
    train_data_level1,train_label_level1,validate_data_level1,validate_label_level1,test_data_level1,test_label_level1 = extract_window_data(dataset=level1_raw,window_size=window_size,label=1)
    train_data_upramp,train_label_upramp,validate_data_upramp,validate_label_upramp,test_data_upramp,test_label_upramp = extract_window_data(dataset=upramp_raw,window_size=window_size,label=2)
    train_data_upramp1,train_label_upramp1,validate_data_upramp1,validate_label_upramp1,test_data_upramp1,test_label_upramp1 = extract_window_data(dataset=upramp1_raw,window_size=window_size,label=2)
    train_data_downramp,train_label_downramp,validate_data_downramp,validate_label_downramp,test_data_downramp,test_label_downramp = extract_window_data(dataset=downramp_raw,window_size=window_size,label=3)
    train_data_downramp1,train_label_downramp1,validate_data_downramp1,validate_label_downramp1,test_data_downramp1,test_label_downramp1 = extract_window_data(dataset=downramp1_raw,window_size=window_size,label=3)
    train_data_upstairs,train_label_upstairs,validate_data_upstairs,validate_label_upstairs,test_data_upstairs,test_label_upstairs = extract_window_data(dataset=upstairs_raw,window_size=window_size,label=4)
    train_data_downstairs,train_label_downstairs,validate_data_downstairs,validate_label_downstairs,test_data_downstairs,test_label_downstairs = extract_window_data(dataset=downstairs_raw,window_size=window_size,label=5)

    train_data_raw = np.concatenate((train_data_stand,train_data_stand1,
                                     train_data_level,train_data_level1,
                                     train_data_upramp,train_data_upramp1,
                                     train_data_downramp,train_data_downramp1,
                                     train_data_upstairs,
                                     train_data_downstairs,
                                     ),axis=0)
    train_label_raw = np.concatenate((train_label_stand,train_label_stand1,
                                     train_label_level,train_label_level1,
                                     train_label_upramp,train_label_upramp1,
                                     train_label_downramp,train_label_downramp1,
                                     train_label_upstairs,
                                     train_label_downstairs,
                                     ),axis=0)
    
    validate_data_raw = np.concatenate((validate_data_stand,validate_data_stand1,
                                     validate_data_level,validate_data_level1,
                                     validate_data_upramp,validate_data_upramp1,
                                     validate_data_downramp,validate_data_downramp1,
                                     validate_data_upstairs,
                                     validate_data_downstairs,
                                     ),axis=0)
    validate_label_raw = np.concatenate((validate_label_stand,validate_label_stand1,
                                     validate_label_level,validate_label_level1,
                                     validate_label_upramp,validate_label_upramp1,
                                     validate_label_downramp,validate_label_downramp1,
                                     validate_label_upstairs,
                                     validate_label_downstairs,
                                     ),axis=0)

    test_data_raw = np.concatenate((test_data_stand,test_data_stand1,
                                     test_data_level,test_data_level1,
                                     test_data_upramp,test_data_upramp1,
                                     test_data_downramp,test_data_downramp1,
                                     test_data_upstairs,
                                     test_data_downstairs,
                                     ),axis=0)
    test_label_raw = np.concatenate((test_label_stand,test_label_stand1,
                                     test_label_level,test_label_level1,
                                     test_label_upramp,test_label_upramp1,
                                     test_label_downramp,test_label_downramp1,
                                     test_label_upstairs,
                                     test_label_downstairs,
                                     ),axis=0)
    
    # 应用标准化
    train_data_norm = np.zeros(train_data_raw.shape)
    for i in range(train_data_raw.shape[0]):
        train_data_norm[i] = scaler.transform(train_data_raw[i])
    validate_data_norm = np.zeros(validate_data_raw.shape)
    for i in range(validate_data_raw.shape[0]):
        validate_data_norm[i] = scaler.transform(validate_data_raw[i])
    test_data_norm = np.zeros(test_data_raw.shape)
    for i in range(test_data_raw.shape[0]):
        test_data_norm[i] = scaler.transform(test_data_raw[i])

    return train_data_norm,train_label_raw,validate_data_norm,validate_label_raw,test_data_norm,test_label_raw


def load_dataset_yhl_train_valid_test(batch_size=32,window_size=50,dataset_style="node"):
    scalers_seq = []
    if dataset_style == "node":
        stand_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/yhl/65.bin',startpoint=0,endpoint=None,data_png_name='00.png')#静止站立  3000
        stand1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/yhl/66.bin',startpoint=0,endpoint=None,data_png_name='01.png')#静止站立label=0
        level_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/yhl/67.bin',startpoint=500,endpoint=4500,data_png_name='10.png')#平地行走,label=1
        level1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/yhl/68.bin',startpoint=500,endpoint=4500,data_png_name='11.png')#平地行走,label=1
        upramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/yhl/69.bin',startpoint=500,endpoint=3500,data_png_name='20.png')#上坡,label=2
        upramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/yhl/71.bin',startpoint=500,endpoint=3500,data_png_name='21.png')#上坡,label=2
        downramp_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/yhl/70.bin',startpoint=500,endpoint=3000,data_png_name='30.png')#下坡,label=3
        downramp1_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/yhl/72.bin',startpoint=500,endpoint=2500,data_png_name='31.png')#下坡,label=3
        upstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/yhl/73.bin',startpoint=1000,endpoint=5000,data_png_name='40.png')#上楼梯,label=4
        downstairs_raw = loaddata_preproc_v2(filepath='./data_v2_20250808/yhl/74.bin',startpoint=1000,endpoint=5000,data_png_name='50.png')#下楼梯,label=5
        scalers_seq = [1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5] # 归一化
    else:
        stand_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/yhl/65.bin',startpoint=0,endpoint=None,data_png_name='00.png')#静止站立  3000
        stand1_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/yhl/66.bin',startpoint=0,endpoint=None,data_png_name='01.png')#静止站立label=0
        level_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/yhl/67.bin',startpoint=500,endpoint=4500,data_png_name='10.png')#平地行走,label=1
        level1_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/yhl/68.bin',startpoint=500,endpoint=4500,data_png_name='11.png')#平地行走,label=1
        upramp_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/yhl/69.bin',startpoint=500,endpoint=3500,data_png_name='20.png')#上坡,label=2
        upramp1_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/yhl/71.bin',startpoint=500,endpoint=3500,data_png_name='21.png')#上坡,label=2
        downramp_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/yhl/70.bin',startpoint=500,endpoint=3000,data_png_name='30.png')#下坡,label=3
        downramp1_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/yhl/72.bin',startpoint=500,endpoint=2500,data_png_name='31.png')#下坡,label=3
        upstairs_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/yhl/73.bin',startpoint=1000,endpoint=5000,data_png_name='40.png')#上楼梯,label=4
        downstairs_raw = loaddata_preproc_v3(filepath='./data_v2_20250808/yhl/74.bin',startpoint=1000,endpoint=5000,data_png_name='50.png')#下楼梯,label=5
        scalers_seq = [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5] # 归一化

    dataset_all = np.concatenate((stand_raw,stand1_raw,
                                     level_raw,level1_raw,
                                     upramp_raw,upramp1_raw,
                                     downramp_raw,downramp1_raw,
                                     upstairs_raw,
                                     downstairs_raw,
                                     ),axis=0)
    
    # #注意，相同物理量纲的数据，采用共享的归一化参数
    # #应该对训练集进行归一化
    if os.path.exists("data_scaler.save"):
        scaler = datasetphysicalSharedMaxMinScalers(feature_range=(-1, 1),scalers_seq=scalers_seq,loaded_scaler_file="data_scaler.save")
        print("加载归一化文件")
    else:
        print("data_scaler.save文件未找到")
        scaler = datasetphysicalSharedMaxMinScalers(feature_range=(-1, 1),scalers_seq=scalers_seq)
    scaler.fit(dataset_all) # 计算数据集最大最小值
    joblib.dump(scaler, "data_scaler.save")# 方法1：使用 joblib (推荐) 或者使用 pickle # import pickle
    print("[归一化参数]：最小值", scaler.data_min_)  # 对于 MinMaxScaler
    print("[归一化参数]：最大值:", scaler.data_max_)  # 对于 MinMaxScaler

    train_data_stand,train_label_stand,validate_data_stand,validate_label_stand,test_data_stand,test_label_stand = extract_window_data(dataset=stand_raw,window_size=window_size,label=0)
    train_data_stand1,train_label_stand1,validate_data_stand1,validate_label_stand1,test_data_stand1,test_label_stand1 = extract_window_data(dataset=stand1_raw,window_size=window_size,label=0)
    train_data_level,train_label_level,validate_data_level,validate_label_level,test_data_level,test_label_level = extract_window_data(dataset=level_raw,window_size=window_size,label=1)
    train_data_level1,train_label_level1,validate_data_level1,validate_label_level1,test_data_level1,test_label_level1 = extract_window_data(dataset=level1_raw,window_size=window_size,label=1)
    train_data_upramp,train_label_upramp,validate_data_upramp,validate_label_upramp,test_data_upramp,test_label_upramp = extract_window_data(dataset=upramp_raw,window_size=window_size,label=2)
    train_data_upramp1,train_label_upramp1,validate_data_upramp1,validate_label_upramp1,test_data_upramp1,test_label_upramp1 = extract_window_data(dataset=upramp1_raw,window_size=window_size,label=2)
    train_data_downramp,train_label_downramp,validate_data_downramp,validate_label_downramp,test_data_downramp,test_label_downramp = extract_window_data(dataset=downramp_raw,window_size=window_size,label=3)
    train_data_downramp1,train_label_downramp1,validate_data_downramp1,validate_label_downramp1,test_data_downramp1,test_label_downramp1 = extract_window_data(dataset=downramp1_raw,window_size=window_size,label=3)
    train_data_upstairs,train_label_upstairs,validate_data_upstairs,validate_label_upstairs,test_data_upstairs,test_label_upstairs = extract_window_data(dataset=upstairs_raw,window_size=window_size,label=4)
    train_data_downstairs,train_label_downstairs,validate_data_downstairs,validate_label_downstairs,test_data_downstairs,test_label_downstairs = extract_window_data(dataset=downstairs_raw,window_size=window_size,label=5)

    train_data_raw = np.concatenate((train_data_stand,train_data_stand1,
                                     train_data_level,train_data_level1,
                                     train_data_upramp,train_data_upramp1,
                                     train_data_downramp,train_data_downramp1,
                                     train_data_upstairs,
                                     train_data_downstairs,
                                     ),axis=0)
    train_label_raw = np.concatenate((train_label_stand,train_label_stand1,
                                     train_label_level,train_label_level1,
                                     train_label_upramp,train_label_upramp1,
                                     train_label_downramp,train_label_downramp1,
                                     train_label_upstairs,
                                     train_label_downstairs,
                                     ),axis=0)
    
    validate_data_raw = np.concatenate((validate_data_stand,validate_data_stand1,
                                     validate_data_level,validate_data_level1,
                                     validate_data_upramp,validate_data_upramp1,
                                     validate_data_downramp,validate_data_downramp1,
                                     validate_data_upstairs,
                                     validate_data_downstairs,
                                     ),axis=0)
    validate_label_raw = np.concatenate((validate_label_stand,validate_label_stand1,
                                     validate_label_level,validate_label_level1,
                                     validate_label_upramp,validate_label_upramp1,
                                     validate_label_downramp,validate_label_downramp1,
                                     validate_label_upstairs,
                                     validate_label_downstairs,
                                     ),axis=0)

    test_data_raw = np.concatenate((test_data_stand,test_data_stand1,
                                     test_data_level,test_data_level1,
                                     test_data_upramp,test_data_upramp1,
                                     test_data_downramp,test_data_downramp1,
                                     test_data_upstairs,
                                     test_data_downstairs,
                                     ),axis=0)
    test_label_raw = np.concatenate((test_label_stand,test_label_stand1,
                                     test_label_level,test_label_level1,
                                     test_label_upramp,test_label_upramp1,
                                     test_label_downramp,test_label_downramp1,
                                     test_label_upstairs,
                                     test_label_downstairs,
                                     ),axis=0)
    
    # 应用标准化
    train_data_norm = np.zeros(train_data_raw.shape)
    for i in range(train_data_raw.shape[0]):
        train_data_norm[i] = scaler.transform(train_data_raw[i])
    validate_data_norm = np.zeros(validate_data_raw.shape)
    for i in range(validate_data_raw.shape[0]):
        validate_data_norm[i] = scaler.transform(validate_data_raw[i])
    test_data_norm = np.zeros(test_data_raw.shape)
    for i in range(test_data_raw.shape[0]):
        test_data_norm[i] = scaler.transform(test_data_raw[i])

    return train_data_norm,train_label_raw,validate_data_norm,validate_label_raw,test_data_norm,test_label_raw


def load_dataset_test():
    print("load dataset test")
