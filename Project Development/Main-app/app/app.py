# Import all of the dependencies
import streamlit as st
import os 
import imageio 

import tensorflow as tf 
from functions import load_data, num_to_char
from ml_model import load_model

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar: 
    st.image('https://www.itransition.com/static/2d784bcbdae971858f7f3e23d560f0fc/machine-learning-statistics-header.jpg')
    st.title('LipReader')
    st.info('This web application employs machine learning techniques to analyze and interpret lip movements from video content.')

st.title('LipReader  WebApp') 


# Generating a list of options or videos 
options = os.listdir(os.path.join('/Users/jewel/Downloads/LipNet-main/data/s1'))
selected_video = st.selectbox('Available Videos', options)


# Generate two columns 
col1, col2 = st.columns(2)

if options: 

    # Rendering the video 
    with col1: 
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('..','data','s1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')
        # Rendering inside of the app
        video = open('test_video.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)


    with col2: 
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        st.info('Raw output from the ML model')
        model = load_model()
        pred = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(pred, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info('Raw output converted into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
        
