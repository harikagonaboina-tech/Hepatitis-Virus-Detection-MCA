import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

model=load_model("daliver_cnn_model.keras")

img_size=(128,128)

def predict_image(path):

    img=load_img(path,target_size=img_size)

    img=img_to_array(img)/255.0

    img=np.expand_dims(img,axis=0)

    prediction=model.predict(img)

    index=np.argmax(prediction)

    if index==0:
        print("No Virus Detected")
    else:
        print("Virus Detected")

predict_image("images/9.png")
