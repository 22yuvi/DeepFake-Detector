import os
import json
import dlib
from facenet_pytorch import MTCNN
import numpy as np
import cv2
import torch
import time

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
