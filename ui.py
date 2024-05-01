import tkinter as tk
from tkinter import filedialog
import shutil
import os
from PIL import Image, ImageTk
import cv2
import numpy as np
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import matplotlib.pyplot as plt


# Define global variables for better organization
IMG_SIZE = 50
LR = 1e-3
MODEL_NAME = 'healthyvsunhealthy-{}-{}.model'.format(LR, '2conv-basic')

def open_photo():
    dir_path = "testpicture"
    for file_name in os.listdir(dir_path):
        os.remove(os.path.join(dir_path, file_name))
    
    file_name = filedialog.askopenfilename(initialdir='C:/Users/HP/Downloads', title='Select image for analysis ',
                                           filetypes=[('Image files', '.jpg')])
    if not file_name:
        return  # User canceled file selection

    dst = "/home/pi/project/PlantDiseaseDetection/testpicture.jpg"
    shutil.copy(file_name, dst)

    load = Image.open(dst)
    render = ImageTk.PhotoImage(load)
    img_label.configure(image=render)
    img_label.image = render

    title_label.destroy()
    button_select_photo.destroy()
    button_analyse_image.grid(column=0, row=2, padx=10, pady=10)

def analyse_image():
    verify_dir = 'testpicture'
    verifying_data = []

    for img in os.listdir(verify_dir):
        path = os.path.join(verify_dir, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        verifying_data.append([np.array(img), img_num])

    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')
    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)
    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)
    convnet = conv_2d(convnet, 128, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)
    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)
    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)
    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)
    convnet = fully_connected(convnet, 4, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('Model loaded!')

    fig = plt.figure()

    for num, data in enumerate(verifying_data):
        img_data, img_num = data

        y = fig.add_subplot(3, 4, num + 1)
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
        model_out = model.predict([data])[0]

        if np.argmax(model_out) == 0:
            str_label = 'healthy'
        elif np.argmax(model_out) == 1:
            str_label = 'bacterial'
        elif np.argmax(model_out) == 2:
            str_label = 'viral'
        elif np.argmax(model_out) == 3:
            str_label = 'lateblight'

        if str_label == 'healthy':
            status = "HEALTHY"
        else:
            status = "UNHEALTHY"

        message_label.config(text='Status: ' + status)

        if str_label == 'bacterial':
            diseasename = "Bacterial Spot "
            disease_label.config(text='Disease Name: ' + diseasename)
            button_remedies.config(command=bact)
        elif str_label == 'viral':
            diseasename = "Yellow leaf curl virus "
            disease_label.config(text='Disease Name: ' + diseasename)
            button_remedies.config(command=vir)
        elif str_label == 'lateblight':
            diseasename = "Late Blight "
            disease_label.config(text='Disease Name: ' + diseasename)
            button_remedies.config(command=latebl)
        else:
            r_label.config(text='Plant is healthy')

def bact():
    show_remedies("Bacterial Spot", "Discard or destroy any affected plants. Do not compost them. Rotate your tomato plants yearly to prevent re-infection next year. Use copper fungicides.")

def vir():
    show_remedies("Yellow leaf curl virus", "Monitor the field, handpick diseased plants and bury them. Use sticky yellow plastic traps. Spray insecticides such as organophosphates, carbamates during the seedling stage. Use copper fungicides.")

def latebl():
    show_remedies("Late Blight", "Monitor the field, remove and destroy infected leaves. Treat organically with copper spray. Use chemical fungicides, the best of which for tomatoes is chlorothalonil.")

def show_remedies(disease_name, remedies):
    window_remedies = tk.Toplevel()
    window_remedies.title("Remedies")
    window_remedies.geometry("400x300")
    window_remedies.configure(background="lightgreen")

    label_disease = tk.Label(window_remedies, text="Disease: " + disease_name, background="lightgreen", fg="Brown", font=("", 15))
    label_disease.pack(pady=10)

    label_remedies = tk.Label(window_remedies, text="Remedies:\n" + remedies, background="lightgreen", fg="Black", font=("", 12))
    label_remedies.pack(pady=10)

    button_exit = tk.Button(window_remedies, text="Exit", command=window_remedies.destroy)
    button_exit.pack(pady=10)

window = tk.Tk()
window.title("Leaf Disease Detection")
window.geometry("500x510")
window.configure(background="lightgreen")

title_label = tk.Label(window, text="Click below to choose picture for testing disease....", background="lightgreen", fg="Brown", font=("", 15))
title_label.grid()

button_select_photo = tk.Button(window, text="Select Photo", command=open_photo)
button_select_photo.grid(column=0, row=1, padx=10, pady=10)

button_analyse_image = tk.Button(window, text="Analyse Image", command=analyse_image)

img_label = tk.Label(window, background="lightgreen")
img_label.grid(column=0, row=1, padx=10, pady=10)

message_label = tk.Label(window, text="", background="lightgreen", fg="Brown", font=("", 15))
message_label.grid(column=0, row=3, padx=10, pady=10)

disease_label = tk.Label(window, text="", background="lightgreen", fg="Black", font=("", 15))
disease_label.grid(column=0, row=4, padx=10, pady=10)

r_label = tk.Label(window, text="", background="lightgreen", fg="Black", font=("", 15))
r_label.grid(column=0, row=4, padx=10, pady=10)

button_remedies = tk.Button(window, text="Remedies", command=None)  # Command will be set dynamically
button_remedies.grid(column=0, row=6, padx=10, pady=10)

window.mainloop()
