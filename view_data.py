from dataset_split import loaddata,extract_window_data


stand_raw = loaddata(filepath='./data20250602/bt_msg_71.bin',startpoint=0,endpoint=None,data_png_name='00.png')#静止站立  3000
stand1_raw = loaddata(filepath='./data20250602/bt_msg_73.bin',startpoint=0,endpoint=None,data_png_name='01.png')#静止站立label=0
# stand2_raw = loaddata(filepath='./data20250601/bt_msg_40.bin',startpoint=2494,endpoint=3362)#静止站立label=0

level_raw = loaddata(filepath='./data20250602/bt_msg_74.bin',startpoint=0,endpoint=None,data_png_name='10.png')#平地,label=1
level1_raw = loaddata(filepath='./data20250602/bt_msg_75.bin',startpoint=0,endpoint=None,data_png_name='11.png')#平地,label=1

upstairs_raw = loaddata(filepath='./data20250602/bt_msg_77.bin',startpoint=0,endpoint=None,data_png_name='20.png')#上楼梯,label=2
# upstairs_raw = loaddata(filepath='./data20250601/bt_msg_58.bin',startpoint=0,endpoint=None,data_png_name='21.png')#上楼梯,label=2

# downstairs_raw = loaddata(filepath='./data20250601/bt_msg_57.bin',startpoint=0,endpoint=None,data_png_name='30.png')#下楼梯,label=3
# downstairs1_raw = loaddata(filepath='./data20250601/bt_msg_59.bin',startpoint=0,endpoint=None,data_png_name='31.png')#下楼梯,label=3
