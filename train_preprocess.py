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
from utilities import *

predictor_name = 'shape_predictor_81_face_landmarks.dat'
predictor_path = os.path.join(working_dir, predictor_name)

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
    vidlist = os.listdir(data_path)
    vidlist.sort()
    for vidname in vidlist:
        vidpath = data_path + vidname
        save_vid_path = meta_dir + vidname[:-4] + '/'
        if os.path.exists(save_vid_path):
            continue
        else:
            os.mkdir(save_vid_path)
        try:
            align_face = preprocess_video(vidname[:-4], detector, predictor, mtcnn, vidpath, save_vid_path)
        except:
            print("Error caused !!")

def resize_frame(align_path, resize_path):
    if not os.path.exists(resize_path):
        os.makedirs(resize_path)
    vidlist = os.listdir(align_path)
    vidlist.sort()
    for vidname in vidlist:
        print("{} - {} ... ".format(align_path, vidname), end='', flush=True)
        vidpath = align_path + vidname + '/'
        save_vid_path = resize_path + vidname + '/'
        if not os.path.exists(save_vid_path):
            os.mkdir(save_vid_path)
        imglist = os.listdir(vidpath)
        imglist.sort()
        f_h, f_w = None, None
        for imgname in imglist:
            if imgname.endswith('.npy'):
                continue
            if imgname.endswith('_face.jpg'):
                continue
            imgpath = vidpath + imgname
            img = cv2.imread(imgpath)
            if f_h is None:
                f_h, f_w = img.shape[:2]
                f_h = ((int(f_h/10)+1)*10)
                f_w = ((int(f_w/10)+1)*10)
            img = cv2.resize(img, (f_w, f_h))
            save_img_path = save_vid_path + imgname
            cv2.imwrite(save_img_path, img)
            
        idx = None
        imglist = os.listdir(save_vid_path)
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
        if idx is not None and idx < 300:
            for i in range(idx, 300):
                ori_img = save_vid_path + ("%04d.jpg"%(idx-1))
                new_img = save_vid_path + ("%04d.jpg"%i)
                os.system("cp " + ori_img + " " + new_img)
        print("Done")

def generate_align_video(video_frame_path, video_store_path):
    if not os.path.exists(video_store_path):
        os.makedirs(video_store_path)
    vidlist = os.listdir(video_frame_path)
    vidlist.sort()
    for vidname in vidlist:
        print("{} - {} ... ".format(video_frame_path, vidname), end='', flush=True)
        vidpath = video_frame_path + vidname +'/'
        save_vid_path = video_store_path + vidname + '.avi'
        os.system("ffmpeg -i {}%04d.jpg {}".format(vidpath, save_vid_path))
        print("Done")

def generate_mag_video(vid_path, mag_path):
    if not os.path.exists(mag_path):
        os.makedirs(mag_path)
    vidlist = os.listdir(vid_path)
    vidlist.sort()
    for vidname in vidlist:
        print("{} - {} ... ".format(video_dir_name, vidname), end='', flush=True)
        vidpath = vid_path + vidname
        save_vid_path = mag_path + vidname
        Magnify(MetaData(file_name=vidpath, low=0.833, high=2, levels=1,
                    amplification=10,  output_folder=mag_path, target = save_vid_path)).do_magnify()
        print("Done")

def generate_mmst_map(mag_path, map_path):
    if not os.path.exists(map_path):
        os.makedirs(map_path)
    ROI_h, ROI_w = 5, 5
    vidlist = os.listdir(mag_path)
    vidlist.sort()
    for vidname in vidlist:
        print("{} - {} ... ".format(mag_path, vidname), end='', flush=True)
        vidpath = mag_path + vidname
        vid = cv2.VideoCapture(vidpath)
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
        save_vid_path = map_path + vidname
        np.save(save_vid_path + ".npy", full_st_map)
        print("Done")
