import tkinter as tk
from tkinter.filedialog import askopenfilename
import shutil
import os
import cv2
import numpy as np
from PIL import Image, ImageTk
import cnn  # Import the CNN module

def display_remedies(disease_name):
    window = tk.Tk()
    window.title("Remedies")
    window.geometry("500x300")
    window.configure(background="lightgreen")

    remedies_text = {
        "bacterial": "Remedies for Bacterial Spot:\n\n"
                     "1. Discard or destroy any affected plants.\n"
                     "2. Do not compost them.\n"
                     "3. Rotate your tomato plants yearly to prevent re-infection next year.\n"
                     "4. Use copper fungicides.",
        "viral": "Remedies for Yellow Leaf Curl Virus:\n\n"
                 "1. Monitor the field, handpick diseased plants, and bury them.\n"
                 "2. Use sticky yellow plastic traps.\n"
                 "3. Spray insecticides such as organophosphates, carbamates during the seedling stage.\n"
                 "4. Use copper fungicides.",
        "lateblight": "Remedies for Late Blight:\n\n"
                      "1. Monitor the field, remove and destroy infected leaves.\n"
                      "2. Treat organically with copper spray.\n"
                      "3. Use chemical fungicides, the best of which for tomatoes is chlorothalonil."
    }

    if disease_name in remedies_text:
        remedies_label = tk.Label(window, text=remedies_text[disease_name], background="lightgreen", fg="black",
                                  font=("", 12))
        remedies_label.pack(padx=10, pady=10)
    else:
        remedies_label = tk.Label(window, text="No remedies found for this disease.", background="lightgreen",
                                  fg="black", font=("", 12))
        remedies_label.pack(padx=10, pady=10)

    window.mainloop()

def analyze_image():
    if os.path.exists("testpicture/testpicture.jpg"):
        disease_name = cnn.analysis()
        display_remedies(disease_name)
    else:
        message = tk.Label(text="Please select an image first.", background="lightgreen", fg="red", font=("", 12))
        message.grid(column=0, row=2, padx=10, pady=10)

def open_photo():
    dir_path = "testpicture"
    file_list = os.listdir(dir_path)
    for file_name in file_list:
        os.remove(os.path.join(dir_path, file_name))

    file_name = askopenfilename(initialdir='C:/Users/sagpa/Downloads/images', title='Select image for analysis',
                                filetypes=[('image files', '.jpg')])
    dst = "/home/pi/project/PlantDiseaseDetection/testpicture/testpicture.jpg"
    shutil.copy(file_name, dst)
    load = Image.open(file_name)
    render = ImageTk.PhotoImage(load)
    img = tk.Label(image=render, height="250", width="500")
    img.image = render
    img.place(x=0, y=0)
    img.grid(column=0, row=1, padx=10, pady=10)
    title.destroy()
    button1.destroy()
    button2 = tk.Button(text="Analyze Image", command=analyze_image)
    button2.grid(column=0, row=2, padx=10, pady=10)

window = tk.Tk()
window.title("Dr. Plant")
window.geometry("500x510")
window.configure(background="lightgreen")

title = tk.Label(text="Click below to choose picture for testing disease....", background="lightgreen", fg="Brown",
                 font=("", 15))
title.grid()

button1 = tk.Button(text="Get Photo", command=open_photo)
button1.grid(column=0, row=1, padx=10, pady=10)

window.mainloop()
