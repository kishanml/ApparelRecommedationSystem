import streamlit as st
#from rembg import remove
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from charset_normalizer import detect
import pixellib 
from pixellib.tune_bg import alter_bg
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

def background_changer():
    change_bg = alter_bg(model_type = "pb")
    change_bg.load_pascalvoc_model("xception_pascalvoc.pb")
    change_bg.change_bg_img(f_image_path = "uploads/opencv_frame_0.png",b_image_path = "background.jpeg", output_image_name="train/train_img.jpg",detect="person")

def main():
    activiteis = ["Home", "Use Webcam", "Local Upload","Test Results","About"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    st.sidebar.markdown(
        """ SKILL LAB PROJECT 
            Members:\n

            \n1.Akash Sahoo

            \n2.Kishan Mishra

            \n3.Sanikesh Mohanty
            \n4.Soumava Dhabal""")
    if choice == "Home":
        st.title("Apparel Recommendation System")
        st.image(
            "https://github.com/KishanMishra1/Datasets-Here/blob/main/Gray%20and%20black%20online%20fashion%20store%20promotion%20video%20template%20(1).gif?raw=true", # I prefer to load the GIFs using GIPHY
            width=500,use_column_width=True # The actual size of most gifs on GIPHY are really small, and using the column-width parameter would make it weirdly big. So I would suggest adjusting the width manually!
        )
        #st.markdown("![Alt Text](https://github.com/KishanMishra1/Datasets-Here/blob/main/Gray%20and%20black%20online%20fashion%20store%20promotion%20video%20template%20(1).gif?raw=true)")
        
        html_temp_about1= """
                             		<div style="background-color:#98AFC7;padding:15px">
                             		<h4 style="color:white;text-align:center;">Model Used : ResNet50 - 48 Convolution layers along with 1 MaxPool and 1 Average Pool layer.<h4>
                             		<h4 style="color:white;text-align:center;">Dataset Used : Myntra Products Image Dataset</h4.
                             		</div>
"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)

        
    elif choice == "Use Webcam":
        st.title("Apparel Recommendation System")
        st.caption("Press Start button to access webcam !")
        if st.button('Start'):
            os.system('python try.py')
            st.caption("Webcam snap")
            st.image("uploads/opencv_frame_0.png")
            
            st.caption("Image object")
            background_changer()
            st.image("train/train_img.jpg")
            features = feature_extraction(os.path.join("train","train_img.jpg"),model)
                #st.text(features)
                # recommendention
            indices = recommend(features,feature_list)
            # show
            col1,col2,col3,col4,col5 = st.columns(5)

            with col1:
                st.header('I')
                st.image(filenames[indices[0][0]])
            with col2:
                st.header('II')
                st.image(filenames[indices[0][1]])
            with col3:
                st.header('III')
                st.image(filenames[indices[0][2]])
            with col4:
                st.header('IV')
                st.image(filenames[indices[0][3]])
            with col5:
                st.header('V')
                st.image(filenames[indices[0][4]])

    elif choice == 'Local Upload':
        uploaded_file = st.file_uploader("Choose an image")
        if uploaded_file is not None:
            if save_uploaded_file(uploaded_file):
                # display the file
                display_image = Image.open(uploaded_file)
                st.image(display_image)
                # feature extract
                features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
                #st.text(features)
                # recommendention
                indices = recommend(features,feature_list)
                # show
                col1,col2,col3,col4,col5 = st.columns(5)

                with col1:
                    st.header('I')
                    st.image(filenames[indices[0][0]])
                with col2:
                    st.header('II')
                    st.image(filenames[indices[0][1]])
                with col3:
                    st.header('III')
                    st.image(filenames[indices[0][2]])
                with col4:
                    st.header('IV')
                    st.image(filenames[indices[0][3]])
                with col5:
                    st.header('V')
                    st.image(filenames[indices[0][4]])
            else:
                st.header("Some error occured in file upload")

    elif choice=="Test Results":
        st.caption("Test 1")
        st.image('ztest/Screenshot 2022-05-22 at 8.19.21 PM.png')
        st.caption("Test 2")
        st.image('ztest/Screenshot 2022-05-23 at 8.58.27 AM.png')
        st.caption("Test 3")
        st.image('ztest/Screenshot 2022-05-23 at 9.09.27 AM.png')
        st.caption("Test 4")
        st.image('ztest/Screenshot 2022-05-23 at 9.17.21 AM.png')



    elif choice == "About":
        st.subheader("About this app")
        html_temp_about1= """
                             		<div style="background-color:#98AFC7;padding:15px">
                             		<h4 style="color:white;text-align:center;">This project is purely made by a group of engineers of Silicon Institute Of Technology, Bhubaneswar. The final project of SKILL-LAB 2022-23, made to solve the chaos of choosing a good outfit from the online sites. Operating the site is as simple as getting ignorance from your crush, just capture your present outfit image or upload the image of the type of outfit you want, and it will recommend you similar outfits that you can try.</h4>
                             		<h4 style="color:white;text-align:center;">If you like the project or want its source code, do ping me at</h4>
                                     <center> <a style="color:white;" href="mailto: mishrakishan2017@gmail.com">Email ID </a> </center>
                                     <h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                             		</div>
                             		<br></br>
                             		<br></br>
"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)


    else:
        pass


if __name__ == "__main__":
    main()
