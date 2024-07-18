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

print("Preparing dlib ... ", end='', flush=True)
detector = dlib.get_frontal_face_detector()
predictor_path = '/shape_predictor_81_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)
print("Done")

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

print("Preparing MTCNN ... ", end='', flush=True)
mtcnn = MTCNN(thresholds=[0.3, 0.3, 0.3], margin=20, keep_all=True, post_process=False, select_largest=False, device=torch.device('cuda', 0))
print("Done")

def get_landmark(img, detector, predictor):
    height, width = np.shape(img)[:2]
    dets = detector(img, 0)
    landmarks = None
    for k, det in enumerate(dets):
        shape = predictor(img, det)
        landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])
        for i in range(81):
            for j in range(2):
                if landmarks[i, j]<0:
                    landmarks[i, j] = 0
            if landmarks[i, 1]>height:
                landmarks[i, 1] = height
            if landmarks[i, 0]>width:
                landmarks[i, 0] = width
    return landmarks

def trans_landmark(landmark, trans_mat, height, width):
    trans_land = np.zeros(np.shape(landmark))
    for i in range(np.shape(landmark)[0]):
        trans_land[i] = [landmark[i,0]+trans_mat[0,2], landmark[i,1]+trans_mat[1,2]]
    trans_land = np.array(trans_land, 'int32')
    for i in range(trans_land.shape[0]):
        if trans_land[i, 0]<0:
            trans_land[i, 0]=0
        if trans_land[i, 1]<0:
            trans_land[i, 1]=0
        if trans_land[i, 0]>width:
            trans_land[i, 0]=width
        if trans_land[i, 1]>height:
            trans_land[i, 1]=height
    return trans_land

def rotate_landmark(landmark, rotate_mat, height, width):
    rotate_land = np.zeros(np.shape(landmark))
    for i in range(np.shape(landmark)[0]):
        rotate_land[i] = np.around(np.array([landmark[i,0], landmark[i,1], 1]).dot(rotate_mat.T))
    rotate_land = np.array(rotate_land, 'int32')
    for i in range(rotate_land.shape[0]):
        if rotate_land[i, 0]<0:
            rotate_land[i, 0]=0
        if rotate_land[i, 1]<0:
            rotate_land[i, 1]=0
        if rotate_land[i, 0]>width:
            rotate_land[i, 0]=width
        if rotate_land[i, 1]>height:
            rotate_land[i, 1]=height
    return rotate_land

def get_rotate_img(img, landmark, first_frame_distance, first_frame_center, idx):
    height, width = np.shape(img)[:2]

    center_left_eye = [np.average(landmark[36:42, 0]), np.average(landmark[36:42, 1])]
    center_right_eye = [np.average(landmark[42:48, 0]), np.average(landmark[42:48, 1])]
    center_eyes = [(center_left_eye[0]+center_right_eye[0])/2, (center_left_eye[1]+center_right_eye[1])/2]
    distance = ((center_eyes[0]-center_right_eye[0])**2 + (center_eyes[1]-center_right_eye[1])**2)**(1/2)
    center = (center_eyes[0], center_eyes[1])

    radius = np.arctan((center_right_eye[1]-center_eyes[1])/(center_right_eye[0]-center_eyes[0]))
    degree = np.degrees(radius)

    if first_frame_distance is None:
        scale = 1
    else :
        scale = first_frame_distance / distance
    rotate_mat = cv2.getRotationMatrix2D(center, degree, scale)
    rotate_img = cv2.warpAffine(img, rotate_mat, (width, height))

    if first_frame_center is not None:
        trans_mat = np.float32([[1,0,first_frame_center[0]-center[0]],[0,1,first_frame_center[1]-center[1]]])
        trans_img = cv2.warpAffine(rotate_img, trans_mat, (width, height))
    else:
        trans_img = img

    face_land = rotate_landmark(landmark, rotate_mat, height, width)
    if first_frame_center is not None:
        face_land = trans_landmark(face_land, trans_mat, height, width)

    return trans_img, distance, center, face_land

def calculate_bounding_landmark(landmark, start_point, bound_w, bound_h):
    bounding_land = np.zeros(landmark.shape)
    for i in range(landmark.shape[0]):
        bounding_land[i] = [landmark[i,0]-start_point[1], landmark[i,1]-start_point[0]]
        if bounding_land[i,0] < 0:
            bounding_land[i,0] = 0
        if bounding_land[i,0] > bound_w:
            bounding_land[i,0] = bound_w
        if bounding_land[i,1] < 0:
            bounding_land[i,1] = 0
        if bounding_land[i,1] > bound_h:
            bounding_land[i,1] = bound_h
    bounding_land = np.array(bounding_land, 'int32')
    return bounding_land

def get_bounding_image(img, landmark):
    height, width, channels = np.shape(img)
    bound_w = max(landmark[:, 0]) - min(landmark[:, 0])
    bound_h = max(landmark[:, 1]) - min(landmark[:, 1])
    bound_w = int(bound_w)
    bound_h = int(bound_h)

    st = [min(landmark[:, 1]), min(landmark[:, 0])]
    bound_img = np.zeros((bound_h, bound_w, channels))
    bound_img = img[st[0]:st[0]+bound_h, st[1]:st[1]+bound_w, :]
    bounding_land = calculate_bounding_landmark(landmark, st, bound_w, bound_h)

    return bound_img, bounding_land

def remove_background(img, landmark):
    height, width, channels = np.shape(img)
    mask = np.zeros((height, width))

    OUT_POINT = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,78,74,79,73,72,80,71,70,69,68,76,75,77]
    outline = np.array(landmark[OUT_POINT])
    cv2.fillPoly(mask, [outline], (255, 255, 255))

    eyeline = np.array(landmark[36:42])
    cv2.fillPoly(mask, [eyeline], (0, 0, 0))
    eyeline = np.array(landmark[42:48])
    cv2.fillPoly(mask, [eyeline], (0, 0, 0))

    nbg_img = np.zeros((height, width, channels))
    for h in range(height):
        for w in range(width):
            if mask[h][w]==0:
                continue
            nbg_img[h][w] = img[h][w]
    return nbg_img

def preprocess_img(img, detector, predictor, first_frame_distance, first_frame_center, idx):
    landmark = get_landmark(img, detector, predictor)
    if landmark is None:
        return None, None, None, None

    rotate_img, distance, center, rotate_land = get_rotate_img(img, landmark, first_frame_distance, first_frame_center, idx)
    bound_img, bounding_land = get_bounding_image(rotate_img, rotate_land)
    no_bg_img = remove_background(bound_img, bounding_land)

    return no_bg_img, distance, center, landmark

def preprocess_video(video_file_name, detector, predictor, mtcnn, vidpath, facepath):
    print("Processing "+video_file_name+":")

    video_file_path = vidpath
    face_align_dir = facepath

    margin = (20, 20, 20, 20)
    batch_size = 30

    first_frame_distance = None
    video = cv2.VideoCapture(video_file_path)
    face_imgs = []
    idx = 0

    stime = time.time()
    frames = []
    while True:
        success, frame = video.read()
        if not success:
            break
        height, width = frame.shape[:2]
        frames.append(frame)
        idx += 1
        if idx > 300:
            break
        if idx % batch_size == 0:
            first_frame_center = None
            batch_boxes, _ = mtcnn.detect(frames)
            pos = 0
            for boxes in batch_boxes:
                print("  %04d ... " % (idx - batch_size + pos), end='', flush=True)
                if boxes is None:
                    print("Done, there is no face")
                    continue
                st = [int(boxes[0,0]-margin[0]), int(boxes[0,1]-margin[1])]
                ed = [int(boxes[0,2]+margin[2]), int(boxes[0,3]+margin[3])]
                if st[0]<0:
                    st[0] = 0
                if st[1]<0:
                    st[1] = 0
                if ed[0]>width:
                    ed[0] = width
                if ed[1]>height:
                    ed[1] = height
                box = [st[0], st[1], ed[0], ed[1]]
                face_img = frames[pos][st[1]:ed[1], st[0]:ed[0], :]
                # cv2.imwrite(face_align_dir + ("%04d_face.jpg" % (idx - batch_size + pos)), face_img)

                pre_img, distance, center, landmark = preprocess_img(face_img, detector, predictor, first_frame_distance, first_frame_center, (idx - batch_size + pos))
                if first_frame_distance is None:
                    first_frame_distance = distance
                if first_frame_center is None:
                    first_frame_center = center
                if pre_img is not None:
                    print("Done, saving ... ", end='', flush=True)
                    cv2.imwrite(face_align_dir + ("%04d.jpg" % (idx - batch_size + pos)), pre_img)
                    # np.save(face_align_dir + ("%04d_landmark.npy" % (idx - batch_size + pos)), landmark)
                    print("Done")
                else :
                    print("Done, no face")
                pos += 1
            frames = []

    print("-- Using time:", time.time() - stime)

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

def calculate_ROI(img, block_h, block_w):
    height, width, channels = np.shape(img)
    blk_h, blk_w = int(height/block_h), int(width/block_w)
    roi_seg = np.zeros((block_h*block_w, channels))
    for bh in range(block_h):
        for bw in range(block_w):
            roi_seg[bh*block_w+bw] = [np.average(img[blk_h*bh:blk_h*(bh+1),blk_w*bw:blk_w*(bw+1),i]) for i in range(3)]
    return roi_seg

def reshape_ROI_SEG(roi_seg):
    bh, bw, channels = np.shape(roi_seg)
    frame_seg = np.zeros((bh*bw, channels))
    for h in range(bh):
        for w in range(bw):
            frame_seg[h*bw+w] = roi_seg[h][w]
    return frame_seg

def get_frame_seg(img, ROI_h, ROI_w):
    ROI_seg = calculate_ROI(img, ROI_h, ROI_w)
    return ROI_seg

def normalization(st_map):
    time, block, channels = np.shape(st_map)
    nom_map = np.zeros((time, block, channels))
    for blk in range(block):
        for c in range(channels):
            t_seq = [x[blk, c] for x in st_map]
            for t in range(time):
                nom_map[t][blk][c] = (t_seq[t] - min(t_seq)) / (max(t_seq) - min(t_seq)) * 255
    return nom_map

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

  
