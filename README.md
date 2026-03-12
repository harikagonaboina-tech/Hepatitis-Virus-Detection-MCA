# Hepatitis Virus Detection Using Deep Learning

This project detects hepatitis virus infection from liver biopsy images using a **Convolutional Neural Network (CNN)**.
The system analyzes liver biopsy images and predicts whether the infection is **Virus** or **No Virus**.

---

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Gradio

---

## Features

* Liver biopsy image classification
* CNN-based deep learning model
* Prediction from new images
* Simple web interface using Gradio

---

## Project Files

```
Hepatitis-Virus-Detection-MCA
│
├── app.py
├── train_model.py
├── predict.py
├── requirements.txt
├── README.md
├── documentation.pdf
├── ppt.pptx
└── images.zip
```

---

## Dataset

The liver biopsy image dataset used for training is large and therefore **not included in this repository**.

To run the training code, users need to provide:

* `data.csv`
* liver biopsy images inside an **images** folder.

---

## How to Run the Project

### 1 Install Required Libraries

```
pip install -r requirements.txt
```

### 2 Run the Web Application

```
python app.py
```

This will start a **Gradio web interface** where users can upload a liver biopsy image and see the prediction result.

---

## Author

**Harika Gonaboina**
MCA Final Year Project
Hepatitis Virus Detection Using Deep Learning
