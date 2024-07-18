import os
import json
import dlib
from facenet_pytorch import MTCNN
import numpy as np
import cv2
import torch
import time

from magnify import Magnify
from metadata import MetaData
from detector import *

print("Preparing dlib ... ", end='', flush=True)
detector = dlib.get_frontal_face_detector()
predictor_path = '/shape_predictor_81_face_landmarks.dat'
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
    print("Done")

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

  
