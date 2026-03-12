import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

csv_path = "data.csv"
images_path = "images"

df = pd.read_csv(csv_path)

img_size = (128,128)

X=[]
y=[]

for index,row in df.iterrows():

    img_path=os.path.join(images_path,f"{row['ID']}.png")

    if os.path.exists(img_path):

        img=load_img(img_path,target_size=img_size)

        img=img_to_array(img)/255.0

        X.append(img)

        y.append(row['Infection'])

X=np.array(X)

le=LabelEncoder()

y=le.fit_transform(y)

y=to_categorical(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

model=Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(128,activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(len(le.classes_),activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train,y_train,epochs=10,batch_size=32)

model.save("daliver_cnn_model.keras")

print("Model saved successfully")
