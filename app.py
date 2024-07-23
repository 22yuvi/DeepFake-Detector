import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import tempfile
from tqdm import tqdm
import time
import cv2
import moviepy.editor as mp
from PIL import Image
import glob
import requests
from streamlit_lottie import st_lottie
from pytube import YouTube
from predictor import run

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def download_video(link):
    yt = YouTube(link)
    video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download(filename="video.mp4")
    return video
  
st.set_page_config(page_title="Deepfake Detector",
                   layout="wide",
                   page_icon="🧑‍⚕️")

with st.sidebar:
    VidOptSel = option_menu('Video Options',
                           ['Upload Custom Video',
                            'Input Youtube Url'],
                           menu_icon='list',
                           icons=['upload', 'person'],
                           default_index=0)

st.title('Deepfake Detector')
col1, col2 = st.columns([1, 3])
url = "https://lottie.host/dd7d9b7b-7713-423a-b3a9-8c78c21a1644/yVdrKmczvO.json"
with col1:
    lottie = load_lottieurl(url)
    st_lottie(lottie)
with col2:
    st.write("""
    ## Face Swap Detector
    ##### Upload a video to detect whether it is real or fake.
    """)

working_dir = os.path.dirname(os.path.abspath(__file__))

link = None
uploaded_file = None
                
if VidOptSel == 'Upload Custom Video':
    uploaded_file = st.file_uploader("Upload your video here...", type=['mp4', 'mov', 'avi', 'mkv'])
if VidOptSel == 'Input Youtube Url':
    link = st.text_input("YouTube Link (The longer the video, the longer the processing time)")
    
if st.button("Predict"):
    if link:
        yt_video = download_video(link)
        video = cv2.VideoCapture("video.mp4") 
        col1, col2 = st.columns([0.5, 0.5])
        with col1:
            st.markdown('<p style="text-align: center;">Video</p>', unsafe_allow_html=True)
            st.video(yt_video)   
    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(uploaded_file.read())
            audio = mp.AudioFileClip(temp_file.name)
            video = cv2.VideoCapture(temp_file.name)
            col1, col2 = st.columns([0.5, 0.5])
            with col1:
                st.markdown('<p style="text-align: center;">Video</p>', unsafe_allow_html=True)
                st.video(temp_file.name)
    if uploaded_file is not None or link is not None:
        with col2:
            st.markdown('<p style="text-align: center;">Magnified Video</p>', unsafe_allow_html=True)
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = video.get(cv2.CAP_PROP_FPS)
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            output_path = os.path.join(working_dir, "target.mp4")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            for _ in tqdm(range(total_frames), unit='frame', desc="Progress"):
                ret, frame = video.read()
                out.write(frame) 
                if not ret:
                    break
            out.release()
            pred = run()
            video.release()
        if(pred == -1):
            st.write('Prediction Failed')
        elif(pred > 0.5 ):
            st.write(f'Fake with accuracy : {pred*100}')
        else:
            st.write(f'Real with accuracy : {100-pred*100}')
            
