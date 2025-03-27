from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import os
import re
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model, model_from_json
import pickle

main = tkinter.Tk()
main.title("Satellite Image Deforestation Analysis with Deep Learning")
main.geometry("1300x1200")

global filename
global deep_learning_acc
global classifier
global X, Y

labels = ['Urban Land', 'Agricultural Land', 'Range Land', 'Forest Land']

def upload():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END, filename + " loaded\n")

def extractFeatures():
    global X, Y
    text.delete('1.0', END)
    X = np.load('model/X.txt.npy')
    Y = np.load('model/Y.txt.npy')
    text.insert(END, "Total Images Found in dataset: " + str(len(X)) + "\n")

def build_model():
    model = Sequential([
        Input(shape=(64, 64, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(4, activation='softmax')
    ])
    return model

def runDeepLabV3Plus():  # Renamed from runCNN
    global X, Y
    global neural_network_acc
    global classifier
    Y1 = to_categorical(Y)
    weights_file = 'model/model.weights.h5'
    json_file = 'model/model.json'
    history_file = 'model/history.pckl'

    if os.path.exists(json_file):
        with open(json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        classifier.load_weights(weights_file)
        print(classifier.summary())
        with open(history_file, 'rb') as f:
            data = pickle.load(f)
        acc = data['accuracy']
        neural_network_acc = acc[-1] * 100
        text.insert(END, "DeepLab V3+ Neural Networks Accuracy: " + str(neural_network_acc) + "\n")  # Renamed
    else:
        classifier = build_model()
        classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        hist = classifier.fit(X, Y1, batch_size=32, epochs=20, shuffle=True, verbose=2)
        classifier.save_weights(weights_file)
        model_json = classifier.to_json()
        with open(json_file, "w") as json_file:
            json_file.write(model_json)
        with open(history_file, 'wb') as f:
            pickle.dump(hist.history, f)
        with open(history_file, 'rb') as f:
            data = pickle.load(f)
        acc = data['accuracy']
        neural_network_acc = acc[-1] * 100
        text.insert(END, "DeepLab V3+ Neural Networks Accuracy: " + str(neural_network_acc) + "\n")  # Renamed

def graph():
    with open('model/history.pckl', 'rb') as f:
        data = pickle.load(f)
    acc = data['accuracy']
    loss = data['loss']
    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy/Loss')
    plt.plot(acc, 'ro-', color='green')
    plt.plot(loss, 'ro-', color='blue')
    plt.legend(['Accuracy', 'Loss'], loc='upper left')
    plt.title('DeepLab V3+ Accuracy & Loss Graph')  # Renamed
    plt.show()

def calculate_forest_area(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([36, 25, 25])  # Lower bound for green (forest)
    upper = np.array([70, 255, 255])  # Upper bound for green (forest)
    mask = cv2.inRange(hsv, lower, upper)
    forest_pixels = cv2.countNonZero(mask)
    total_pixels = image.shape[0] * image.shape[1]
    forest_percentage = (forest_pixels / total_pixels) * 100
    return forest_percentage, mask

def predict():
    text.delete('1.0', END)
    filename1 = filedialog.askopenfilename(initialdir="sampleImages", title="Select First Image (Earlier)")
    filename2 = filedialog.askopenfilename(initialdir="sampleImages", title="Select Second Image (Later)")

    # Load and preprocess first image
    image1 = cv2.imread(filename1)
    img1 = cv2.resize(image1, (64, 64))
    im2arr1 = np.array(img1)
    im2arr1 = im2arr1.reshape(1, 64, 64, 3)
    img1 = np.asarray(im2arr1)
    img1 = img1.astype('float32') / 255
    preds1 = classifier.predict(img1)
    predict1 = np.argmax(preds1)

    # Load and preprocess second image
    image2 = cv2.imread(filename2)
    img2 = cv2.resize(image2, (64, 64))
    im2arr2 = np.array(img2)
    im2arr2 = im2arr2.reshape(1, 64, 64, 3)
    img2 = np.asarray(im2arr2)
    img2 = img2.astype('float32') / 255
    preds2 = classifier.predict(img2)
    predict2 = np.argmax(preds2)

    # Use original images for display (no resizing)
    img_display1 = image1.copy()
    img_display2 = image2.copy()

    # Calculate forest area for both images
    forest_percentage1, mask1 = calculate_forest_area(img_display1)
    forest_percentage2, mask2 = calculate_forest_area(img_display2)

    # Calculate deforestation percentage
    deforestation_percentage = max(0, forest_percentage1 - forest_percentage2)

    # Draw bounding boxes and labels on first image
    contours1, _ = cv2.findContours(mask1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours1:
        if len(contour) > 10:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img_display1, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(img_display1, f'Class: {labels[predict1]} ({forest_percentage1:.2f}% Forest)', 
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw bounding boxes and labels on second image
    contours2, _ = cv2.findContours(mask2.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours2:
        if len(contour) > 10:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img_display2, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(img_display2, f'Class: {labels[predict2]} ({forest_percentage2:.2f}% Forest)', 
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display images in windows matching their original size
    cv2.imshow('Earlier Image', img_display1)
    cv2.imshow('Later Image', img_display2)

    # Wait for key press and close windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Output results to text widget
    text.insert(END, f"Earlier Image Classified as: {labels[predict1]} with {forest_percentage1:.2f}% Forest\n")
    text.insert(END, f"Later Image Classified as: {labels[predict2]} with {forest_percentage2:.2f}% Forest\n")
    text.insert(END, f"Deforestation Percentage: {deforestation_percentage:.2f}%\n")

font = ('times', 14, 'bold')
title = Label(main, text='Detecting Deforestation from Satellite Images')
title.config(bg='yellow3', fg='white')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=0, y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Satellite Images Dataset", command=upload)
uploadButton.place(x=50, y=100)
uploadButton.config(font=font1)

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')
pathlabel.config(font=font1)
pathlabel.place(x=460, y=100)

featuresButton = Button(main, text="Extract Features from Images", command=extractFeatures)
featuresButton.place(x=50, y=150)
featuresButton.config(font=font1)

deepLabV3PlusButton = Button(main, text="Train DeepLab V3+ Algorithm", command=runDeepLabV3Plus)  # Renamed from cnnButton
deepLabV3PlusButton.place(x=310, y=150)
deepLabV3PlusButton.config(font=font1)

graphbutton = Button(main, text="Accuracy Graph", command=graph)
graphbutton.place(x=50, y=200)
graphbutton.config(font=font1)

predictb = Button(main, text="Upload Two Images & Analyze Deforestation", command=predict)
predictb.place(x=310, y=200)
predictb.config(font=font1)

font1 = ('times', 12, 'bold')
text = Text(main, height=20, width=150)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10, y=250)
text.config(font=font1)

main.config(bg='burlywood2')
main.mainloop()