import os
import json
import dlib
from facenet_pytorch import MTCNN
import numpy as np
import cv2
import torch
import time
import streamlit as st

from magnify import Magnify
from metadata import MetaData
from utilities import *
from model_arch import *

working_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(working_dir, "target.mp4")
meta_dir = os.path.join(working_dir, "faces/")
resize_dir = os.path.join(working_dir, "resize/")
newviddir = os.path.join(working_dir, "video.avi")
mag_path = os.path.join(working_dir, "magnified_video.avi")
map_path = os.path.join(working_dir, "map")
model_path = "/content/drive/MyDrive/Model/model_Meso.h5"
predictor_name = 'shape_predictor_81_face_landmarks.dat'
predictor_path = os.path.join(working_dir, predictor_name)
mesoIncep_name = 'MesoInception_F2F.h5'
mesoIncep_path = os.path.join(working_dir, mesoIncep_name)

print("Preparing dlib ... ", end='', flush=True)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
print("Done")

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

print("Preparing MTCNN ... ", end='', flush=True)
mtcnn = MTCNN(thresholds=[0.3, 0.3, 0.3], margin=20, keep_all=True, post_process=False, select_largest=False, device=torch.device('cuda', 0))
print("Done")

def generate_align_face(data_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    try:
        align_face = preprocess_video(vidname[:-4], detector, predictor, mtcnn, data_path, save_path)
    except:
        print("Error caused !!")

def resize_frame(align_path, resize_path):
    if not os.path.exists(resize_path):
        os.makedirs(resize_path)
    imglist = os.listdir(align_path)
    imglist.sort()
    f_h, f_w = None, None
    for imgname in imglist:
        imgpath = align_path + imgname
        img = cv2.imread(imgpath)
        if f_h is None:
            f_h, f_w = img.shape[:2]
            f_h = ((int(f_h/10)+1)*10)
            f_w = ((int(f_w/10)+1)*10)
        img = cv2.resize(img, (f_w, f_h))
        save_img_path = resize_path + imgname
        cv2.imwrite(save_img_path, img)

    idx = None
    save_vid_path = resize_path
    imglist = os.listdir(resize_path)
    imglist.sort()
    imglist.append("0300.jpg")
    for imgname in imglist:
        if int(imgname[:4]) >= 300:
            break
        if idx is None and int(imgname[:4])!=0:
            for i in range(0, int(imgname[:4])):
                ori_img = save_vid_path + imgname
                new_img = save_vid_path + ("%04d.jpg"%i)
                os.system("cp " + ori_img + " " + new_img)
        if idx is not None and idx < int(imgname[:4]):
            for i in range(idx, int(imgname[:4])):
                ori_img = save_vid_path + ("%04d.jpg"%(idx-1))
                new_img = save_vid_path + ("%04d.jpg"%i)
                os.system("cp " + ori_img + " " + new_img)
        idx = int(imgname[:4]) + 1

def generate_align_video(video_frame_path, video_store_path):
    os.system("ffmpeg -i {}%04d.jpg {}".format(video_frame_path, video_store_path))

def generate_mag_video(vid_path, mag_path):
        Magnify(MetaData(file_name=vid_path, low=0.833, high=2, levels=1,
                    amplification=10,  output_folder=mag_path, target = mag_path)).do_magnify()

def generate_mmst_map(mag_path, map_path):
    ROI_h, ROI_w = 5, 5
    vid = cv2.VideoCapture(mag_path)
    full_st_map = np.zeros((300, 25, 3))
    idx = 0
    while idx < 300:
        success, frame = vid.read()
        if not success:
            break
        frame_seg = get_frame_seg(frame, ROI_h, ROI_w)
        full_st_map[idx] = frame_seg
        idx += 1
    full_st_map = normalization(full_st_map)
    np.save(map_path + ".npy", full_st_map)

def run() -> int:
    with st.spinner('Creating MMST Map......'):
        generate_align_face(dataset_dir, meta_dir)
        resize_frame(meta_dir, resize_dir)
        generate_align_video(resize_dir, newviddir)
        generate_mag_video(newviddir, mag_path)
        generate_mmst_map(mag_path, map_path)

    with st.spinner('Preparing Data.....'):
        classifier = MesoInception4()
        classifier.load(mesoIncep_path)
        
        img_list = os.listdir(resize_dir)
        if img_list==[]:
            return -1
        img_list.sort()
        data_Meso = np.ones(300) * 0.5
        try:
            for img_name in img_list:
                if int(img_name[:-4]) >= 300:
                    continue
                img_path = resize_dir +'/'+ img_name
                img = cv2.resize(cv2.imread(img_path), (256, 256))
                pred = classifier.predict(np.array([img]))
                data_Meso[int(img_name[:-4])] = pred
        except Exception:
            print("Error in predicting")
            return -1    
        data_mit = np.load(map_path)

    with st.spinner('Predicting.....'):
        model = load_model(model_path, custom_objects={'multiply':multiply, 'Add':Add, 'X_plus_Layer':X_plus_Layer})
        prediction = model.predict([data_mit, data_Meso])
    return prediction
