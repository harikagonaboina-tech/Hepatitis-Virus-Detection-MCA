import gradio as gr
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

model=load_model("daliver_cnn_model.keras")

img_size=(128,128)

class_names=["No Virus","Virus"]

def predict_infection(image):

    image=image.resize(img_size)

    img_array=img_to_array(image)/255.0

    img_array=np.expand_dims(img_array,axis=0)

    prediction=model.predict(img_array)

    index=np.argmax(prediction)

    label=class_names[index]

    confidence=round(np.max(prediction)*100,2)

    return f"{label} ({confidence}%)"

interface=gr.Interface(
    fn=predict_infection,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Liver Infection Detection using CNN",
    description="Upload liver biopsy image to detect virus."
)

interface.launch()
