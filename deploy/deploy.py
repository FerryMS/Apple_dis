import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np

model=tf.keras.models.load_model('model_apple.h5')

st.title('Apple Disease Classifier')

st.header ('Apple Disease Classifier')

st.text ('Identifies the following categories: Apple rot leaves, Healthy leaves, Leaf blotch, Scab leaveas')

input_img = st.file_uploader('Upload apple leaf image here:', type=['jpg','jpeg','png'])

if st.button('Submit'):

    img = tf.keras.preprocessing.image.load_img(input_img,target_size=(224,224))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0) 

    images = np.vstack([x])
    classes = model.predict(images)

    X = np.argmax(classes)

    if X == 0:
        st.image(img, use_column_width=True)
        st.text ('Apple rot leaves')
    elif X == 1:
        st.image(img, use_column_width=True)
        st.text ('Healthy leaves')
    elif X == 2:
        st.image(img, use_column_width=True)
        st.text ('Leaf blotch')
    elif X == 3:
        st.image(img, use_column_width=True)
        st.text ('Scab Leaves')
    else:
        st.text ('Invalid input, Please upload an image.')