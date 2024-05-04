import tkinter as tk
from tkinter.filedialog import askopenfilename
import shutil
import os
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf

# Initialize Tkinter window
window = tk.Tk()
window.title("Dr. Plant")
window.geometry("500x510")
window.configure(background="lightgreen")

# Function to update GUI with disease information
def update_gui(model_out):
    if np.argmax(model_out) == 0:
        str_label = "healthy"
    elif np.argmax(model_out) == 1:
        str_label = "bacterial"
    elif np.argmax(model_out) == 2:
        str_label = "viral"
    elif np.argmax(model_out) == 3:
        str_label = "lateblight"

    if str_label == "healthy":
        status = "HEALTHY"
    else:
        status = "UNHEALTHY"

    message.config(text="Status: " + status)

    if str_label in ["bacterial", "viral", "lateblight"]:
        show_disease_info(str_label)

# Function to show disease information
def show_disease_info(disease_name):
    if disease_name == "bacterial":
        diseasename = "Bacterial Spot"
    elif disease_name == "viral":
        diseasename = "Yellow leaf curl virus"
    elif disease_name == "lateblight":
        diseasename = "Late Blight"

    disease.config(text="Disease Name: " + diseasename)
    remedies.config(text="Click below for remedies...")
    button3.config(command=lambda: show_remedies(disease_name))

# Function to show remedies
def show_remedies(disease_name):
    if disease_name == "bacterial":
        bact()
    elif disease_name == "viral":
        vir()
    elif disease_name == "lateblight":
        latebl()

# Function to handle "Bacterial Spot" remedies
def bact():
    window1 = tk.Tk()
    window1.title("Leaf Disease Detection")
    window1.geometry("500x510")
    window1.configure(background="lightgreen")

    rem = "The remedies for Bacterial Spot are:\n\n "
    remedies = tk.Label(
        text=rem, background="lightgreen", fg="Brown", font=("", 15)
    )
    remedies.grid(column=0, row=7, padx=10, pady=10)
    rem1 = (
        " Discard or destroy any affected plants. \n  Do not compost them. \n  Rotate yoour tomato plants yearly to prevent re-infection next year. \n Use copper fungicites"
    )
    remedies1 = tk.Label(
        text=rem1, background="lightgreen", fg="Black", font=("", 12)
    )
    remedies1.grid(column=0, row=8, padx=10, pady=10)

    button = tk.Button(text="Exit", command=window1.destroy)
    button.grid(column=0, row=9, padx=20, pady=20)

    window1.mainloop()

# Function to handle "Yellow leaf curl virus" remedies
def vir():
    window1 = tk.Tk()
    window1.title("Dr. Plant")
    window1.geometry("650x510")
    window1.configure(background="lightgreen")

    rem = "The remedies for Yellow leaf curl virus are: "
    remedies = tk.Label(
        text=rem, background="lightgreen", fg="Brown", font=("", 15)
    )
    remedies.grid(column=0, row=7, padx=10, pady=10)
    rem1 = (
        " Monitor the field, handpick diseased plants and bury them. \n  Use sticky yellow plastic traps. \n  Spray insecticides such as organophosphates, carbametes during the seedliing stage. \n Use copper fungicites"
    )
    remedies1 = tk.Label(
        text=rem1, background="lightgreen", fg="Black", font=("", 12)
    )
    remedies1.grid(column=0, row=8, padx=10, pady=10)

    button = tk.Button(text="Exit", command=window1.destroy)
    button.grid(column=0, row=9, padx=20, pady=20)

    window1.mainloop()

# Function to handle "Late Blight" remedies
def latebl():
    window1 = tk.Tk()
    window1.title("Dr. Plant")
    window1.geometry("520x510")
    window1.configure(background="lightgreen")

    rem = "The remedies for Late Blight are: "
    remedies = tk.Label(
        text=rem, background="lightgreen", fg="Brown", font=("", 15)
    )
    remedies.grid(column=0, row=7, padx=10, pady=10)

    rem1 = (
        " Monitor the field, remove and destroy infected leaves. \n  Treat organically with copper spray. \n  Use chemical fungicides,the best of which for tomatoes is chlorothalonil."
    )
    remedies1 = tk.Label(
        text=rem1, background="lightgreen", fg="Black", font=("", 12)
    )
    remedies1.grid(column=0, row=8, padx=10, pady=10)

    button = tk.Button(text="Exit", command=window1.destroy)
    button.grid(column=0, row=9, padx=20, pady=20)

    window1.mainloop()

 def process_verify_data():
     verify_dir = 'testpicture'
     verifying_data = []
     for img in tqdm(os.listdir(verify_dir)):
         path = os.path.join(verify_dir, img)
         img_num = img.split('.')[0]
         img = cv2.imread(path, cv2.IMREAD_COLOR)
         img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
         verifying_data.append([np.array(img), img_num])
     np.save('verify_data.npy', verifying_data)
     return verifying_data

 verify_data = process_verify_data()
 #verify_data = np.load('verify_data.npy')

# Function to analyze images
def analysis(verify_data):
    loading_message.config(text="Loading...")

    num_iterations = min(len(verify_data), 12)

    for num, data in enumerate(verify_data[:num_iterations]):
        print("Processing image", num + 1)

        try:
            img_num = data[1]
            img_data = data[0]

            y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            print("Data shape:", data.shape)

            model_out = model.predict([data])[0]
            print("Model output:", model_out)

            update_gui(model_out)

        except Exception as e:
            print("Error processing image:", e)

        window.update()
analysis(verify_data)

# Function to open photo for analysis
def openphoto():
    dirPath = "testpicture"
    fileList = os.listdir(dirPath)
    for fileName in fileList:
        os.remove(dirPath + "/" + fileName)

    fileName = askopenfilename(
        initialdir="C:/Users/HP/Downloads/images",
        title="Select image for analysis ",
        filetypes=[("image files", ".jpg")],
    )
    dst = "/home/pi/project/PlantDiseaseDetection/testpicture.jpg"
    shutil.copy(fileName, dst)
    load = Image.open(fileName)
    render = ImageTk.PhotoImage(load)
    img = tk.Label(image=render, height="250", width="500")
    img.image = render
    img.place(x=0, y=0)
    img.grid(column=0, row=1, padx=10, pady=10)
    title.destroy()
    button1.destroy()
    button2 = tk.Button(text="Analyse Image", command=analysis)
    button2.grid(column=0, row=2, padx=10, pady=10)

# GUI elements
title = tk.Label(
    text="Click below to choose picture for testing disease....",
    background="lightgreen",
    fg="Brown",
    font=("", 15)
)
title.grid()

button1 = tk.Button(text="Get Photo", command=openphoto)
button1.grid(column=0, row=1, padx=10, pady=10)

message = tk.Label(
    text="",
    background="lightgreen",
    fg="Brown",
    font=("", 15),
)
message.grid(column=0, row=3, padx=10, pady=10)

disease = tk.Label(
    text="",
    background="lightgreen",
    fg="Black",
    font=("", 15),
)
disease.grid(column=0, row=4, padx=10, pady=10)

remedies = tk.Label(
    text="",
    background="lightgreen",
    fg="Brown",
    font=("", 15),
)
remedies.grid(column=0, row=5, padx=10, pady=10)

button3 = tk.Button(text="Remedies", command=lambda: None)
button3.grid(column=0, row=6, padx=10, pady=10)

loading_message = tk.Label(
    text="",
    background="lightgreen",
    fg="Brown",
    font=("", 15),
)
loading_message.grid(column=0, row=3, padx=10, pady=10)

window.mainloop()
