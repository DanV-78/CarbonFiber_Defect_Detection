#This is the main file to run for the streamlit app 
import datetime, os, random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import matplotlib.image as mpimg
from PIL import Image
import streamlit as st

st.set_page_config(
    page_title = "Introducing the Carbon Fiber Defect Detection Demonstration.",
    layout = "wide",
    menu_items = {
        "About":"Copyright Engineering and Data Limited",
        "Get Help": "https://www.engndata.com"
            
    }
)
st.title("Introducing the Carbon Fiber Defect Detection Demo.")

st.markdown(
    '''
    This web page showcases a potential industrial application 
    for automated defect detection in carbon fiber products.   
    The machine learning model used was trained offline using a dataset uploaded online by RWTH Aachen University under the CC BY 4.0 license.  

    Full Dataset available at:  
    https://data.niaid.nih.gov/resources?id=ZENODO_7970489 

    For information or queries email: info@engndata.com  
    Copyright: Engineering and Data Limited  
    Website: www.engndata.com  
    ''' 
)



st.divider()
regular1 = mpimg.imread('./samples/1_regular.png')
regular2 = mpimg.imread('./samples/2_regular.png')
fold1 = mpimg.imread('./samples/fold_01.png')
fold2 = mpimg.imread('./samples/fold_02.png')
gap1 = mpimg.imread('./samples/gap_01.png')
gap2 = mpimg.imread('./samples/gap_02.png')

fig, ax = plt.subplots(3,2,figsize=(5,5), tight_layout=True)

ax[0,0].imshow(regular1,cmap='gray')
ax[0,0].set_title('REGULAR')
ax[0,0].axis("off")

ax[0,1].imshow(regular2,cmap='gray')
ax[0,1].set_title('REGULAR')
ax[0,1].axis("off")

ax[1,0].imshow(fold1)
ax[1,0].set_title('FOLD')
ax[1,0].axis("off")

ax[1,1].imshow(fold2)
ax[1,1].set_title('FOLD')
ax[1,1].axis("off")

ax[2,0].imshow(gap1)
ax[2,0].set_title('GAP')
ax[2,0].axis("off")

ax[2,1].imshow(gap2)
ax[2,1].set_title('GAP')
ax[2,1].axis("off")


with st.container(height=500,border=True):
    st.subheader("Samples space")
    st.pyplot(fig)



st.divider()
st.markdown(
    '''
    Clicking the button below will randomly select a picture from the folder,  
    pass it to the model for labelling according to the three categories [Regular, Gap, Fold] 
      
    '''
)
infer_dataset = ('./infer-images')
loaded_model = keras.saving.load_model('./model_drop.keras')
class_names = ['fold', 'gap', 'regular']

def check():
    random_image = random.choice(os.listdir(infer_dataset))
    print(f'Image {random_image} has been selected')
    rand = './infer-images/'+random_image
    img_bw = keras.utils.load_img(rand, color_mode='grayscale')
    in_arr = keras.utils.img_to_array(img_bw)
    in_arr = np.array([in_arr])
    predictions = loaded_model.predict(in_arr)
    score = tf.nn.softmax([predictions[0]])
    st.write("This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100*np.max(score))
    )
    print_image =(mpimg.imread(infer_dataset+'/'+random_image))
    fig2, ax2 = plt.subplots(figsize=(1,1))
    ax2.imshow(print_image,cmap='gray')
    ax2.axis("off")
   
    st.pyplot(fig2)

with st.container(height=550, border=True):
    if st.button('Check Picture', use_container_width=False):
         check()


st.divider()