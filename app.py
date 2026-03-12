import gradio as gr
import random
from PIL import Image

# Demo class names
class_names = ["No Virus", "Virus"]

def predict_infection(image):

    # Demo prediction (random result)
    predicted_label = random.choice(class_names)

    confidence = round(random.uniform(80, 99), 2)

    return f"Prediction: {predicted_label} (Confidence: {confidence}%)"


interface = gr.Interface(

    fn=predict_infection,

    inputs=gr.Image(type="pil", label="Upload Liver Image"),

    outputs=gr.Textbox(label="Prediction Result"),

    title="Liver Infection Detection using CNN",

    description="Upload a liver biopsy image and the system will predict Virus or No Virus (Demo Version)."

)

interface.launch()
