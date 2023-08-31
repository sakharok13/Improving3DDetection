import pickle


semi_data = "/home/junbo/ssd/repository/SemiDet3D/data/waymo/waymo_infos_train_D5.pkl"
openpcdet_list_path = "/home/junbo/ssd/repository/CenterPoint-Waymo/data/Waymo/openpcdet/openpcdet_train_scene_list.txt"

semi_data=pickle.load(open('/home/junbo/ssd/repository/SemiDet3D/data/waymo/waymo_infos_train_D5.pkl','rb'))
semi_data = [ele['metadata']['context_name']+'-'+str(ele['metadata']['timestamp_micros']) for ele in semi_data]
openpcdet_list = [x.strip() for x in open(openpcdet_list_path).readlines()]