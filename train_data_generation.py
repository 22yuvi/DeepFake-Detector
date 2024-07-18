import os
import numpy as np
from model_arch import *

classifier = MesoInception4()
classifier.load('/content/drive/MyDrive/DeepFakeDetection/MesoInception_F2F.h5')

real_video_dir = "/content/drive/MyDrive/DeepFakeDetection/video/real/"
real_vid_list = os.listdir(real_video_dir)
real_vid_list.sort()
real_vid_dict = {x:real_video_dir + x for x in real_vid_list}
fake_video_dir = "/content/drive/MyDrive/DeepFakeDetection/video/fake/"
fake_vid_list = os.listdir(fake_video_dir)
fake_vid_list.sort()
fake_vid_dict = {x:fake_video_dir + x for x in fake_vid_list}
video_dict = real_vid_dict
video_dict.update(fake_vid_dict)

fake_mit_dir = "/content/drive/MyDrive/DeepFakeDetection/video/fake/"
fake_map_dir = "/content/drive/MyDrive/DeepFakeDetection/map/fake/"
fake_mit_list = os.listdir(fake_mit_dir)
fake_mit_list.sort()
fake_mit_dict = {x:fake_map_dir + x + '.npy' for x in fake_mit_list}
real_mit_dir = "/content/drive/MyDrive/DeepFakeDetection/video/real/"
real_map_dir = "/content/drive/MyDrive/DeepFakeDetection/map/real/"
real_mit_list = os.listdir(real_mit_dir)
real_mit_list.sort()
real_mit_dict = {x:real_map_dir + x + '.npy' for x in real_mit_list}

mit_dict = fake_mit_dict
mit_dict.update(real_mit_dict)

data_name_list = real_vid_list + fake_vid_list

data_Meso_set = []
data_mit_set = []
data_y_set = []
data_name_set = []

for vid_name in data_name_list:
    print("video {}:".format(vid_name), end='', flush=True)
    if vid_name in mit_dict:
        if vid_name in fake_mit_dict:
          video_dir_name = 'fake/'
        if vid_name in real_mit_dict:
          video_dir_name = 'real/'
        meta_dir = '/content/drive/MyDrive/DeepFakeDetection/resize/' + video_dir_name
        vid_path = meta_dir + vid_name[:-4]
        print(vid_path)
        img_list = os.listdir(vid_path)
        if img_list==[]:
            continue
        img_list.sort()
        data_Meso = np.ones(300) * 0.5
        try:
            for img_name in img_list:
                if int(img_name[:-4]) >= 300:
                    continue
                img_path = vid_path +'/'+ img_name
                img = cv2.resize(cv2.imread(img_path), (256, 256))
                pred = classifier.predict(np.array([img]))
                data_Meso[int(img_name[:-4])] = pred
        except Exception:
            print("hello")
            continue
        data_Meso_set.append(data_Meso)
        data_mit = np.load(mit_dict[vid_name])
        data_mit_set.append(data_mit)

        if video_dir_name == 'fake/':
            data_y_set.append(1)
            print(1)
        elif video_dir_name == 'real/':
            data_y_set.append(0)
            print(0)

        data_name_set.append(vid_name)

    print("  ", np.shape(data_Meso_set))
    print("  ", np.shape(data_mit_set))
    print("  ", np.shape(data_y_set))
    print("  ", np.shape(data_name_set))

save_path = "/content/drive/MyDrive/DeepFakeDetection/"
np.save(save_path+"Meso.npy", data_Meso_set)
np.save(save_path+"mit.npy", data_mit_set)
np.save(save_path+"y.npy", data_y_set)
np.save(save_path+"name.npy", data_y_set)
