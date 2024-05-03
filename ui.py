import tkinter as tk
from tkinter.filedialog import askopenfilename
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
from tqdm import tqdm

# Define the main application window
class LeafDiseaseDetectionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Leaf Disease Detection")
        self.geometry("500x510")
        self.configure(background="lightgreen")

        self.title_label = tk.Label(self, text="Click below to choose picture for testing disease....",
                                    background="lightgreen", fg="Brown", font=("", 15))
        self.title_label.grid(column=0, row=0, padx=10, pady=10)

        self.button_get_photo = tk.Button(self, text="Get Photo", command=self.open_photo)
        self.button_get_photo.grid(column=0, row=1, padx=10, pady=10)

    # Function to open photo and start analysis
    def open_photo(self):
        dir_path = "testpicture"
        for file_name in os.listdir(dir_path):
            os.remove(os.path.join(dir_path, file_name))
        
        # Select image for analysis
        file_name = askopenfilename(initialdir='C:/Users/HP/Downloads', title='Select image for analysis ',
                                    filetypes=[('image files', '.jpg')])
        dst = "/home/pi/project/PlantDiseaseDetection/testpicture.jpg"
        shutil.copy(file_name, dst)

        # Load and display the selected image
        load = Image.open(dst)
        render = ImageTk.PhotoImage(load)
        img_label = tk.Label(image=render, height="250", width="500")
        img_label.image = render
        img_label.grid(column=0, row=2, padx=10, pady=10)
        self.title_label.destroy()
        self.button_get_photo.destroy()

        # Analyze the image
        self.analyze_image()

    # Function to analyze the selected image
    def analyze_image(self):
        verify_dir = 'testpicture'
        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'healthyvsunhealthy-{}-{}.model'.format(LR, '2conv-basic')

        def process_verify_data():
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

        print("Loading model...")
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
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy',
                             name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            print("Model found, loading...")
            model.load(MODEL_NAME)
            print('Model loaded!')
        else:
            print("Model file not found!")

        fig = plt.figure()

        for num, data in enumerate(verify_data):
            img_num = data[1]
            img_data = data[0]

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

            message = tk.Label(text='Status: ' + status, background="lightgreen",
                               fg="Brown", font=("", 15))
            message.grid(column=0, row=3, padx=10, pady=10)
            if str_label == 'bacterial':
                diseasename = "Bacterial Spot "
                disease = tk.Label(text='Disease Name: ' + diseasename, background="lightgreen",
                                   fg="Black", font=("", 15))
                disease.grid(column=0, row=4, padx=10, pady=10)
                r = tk.Label(text='Click below for remedies...', background="lightgreen", fg="Brown", font=("", 15))
                r.grid(column=0, row=5, padx=10, pady=10)
                button3 = tk.Button(text="Remedies", command=self.bact)
                button3.grid(column=0, row=6, padx=10, pady=10)
            elif str_label == 'viral':
                diseasename = "Yellow leaf curl virus "
                disease = tk.Label(text='Disease Name: ' + diseasename, background="lightgreen",
                                   fg="Black", font=("", 15))
                disease.grid(column=0, row=4, padx=10, pady=10)
                r = tk.Label(text='Click below for remedies...', background="lightgreen", fg="Brown",
                             font=("", 15))
                r.grid(column=0, row=5, padx=10, pady=10)
                button3 = tk.Button(text="Remedies", command=self.vir)
                button3.grid(column=0, row=6, padx=10, pady=10)
            elif str_label == 'lateblight':
                diseasename = "Late Blight "
                disease = tk.Label(text='Disease Name: ' + diseasename, background="lightgreen",
                                   fg="Black", font=("", 15))
                disease.grid(column=0, row=4, padx=10, pady=10)
                r = tk.Label(text='Click below for remedies...', background="lightgreen", fg="Brown",
                             font=("", 15))
                r.grid(column=0, row=5, padx=10, pady=10)
                button3 = tk.Button(text="Remedies", command=self.latebl)
                button3.grid(column=0, row=6, padx=10, pady=10)
            else:
                r = tk.Label(text='Plant is healthy', background="lightgreen", fg="Black",
                             font=("", 15))
                r.grid(column=0, row=4, padx=10, pady=10)
                button = tk.Button(text="Exit", command=self.exit)
                button.grid(column=0, row=9, padx=20, pady=20)

    # Function to display remedies for Bacterial Spot
    def bact(self):
        window1 = tk.Toplevel()
        window1.title("Remedies for Bacterial Spot")
        window1.geometry("500x510")
        window1.configure(background="lightgreen")

        def exit():
            window1.destroy()

        rem = "The remedies for Bacterial Spot are:\n\n "
        remedies = tk.Label(window1, text=rem, background="lightgreen", fg="Brown", font=("", 15))
        remedies.grid(column=0, row=0, padx=10, pady=10)

        rem1 = " Discard or destroy any affected plants. \n  Do not compost them. \n  Rotate your tomato plants yearly to prevent re-infection next year. \n Use copper fungicides"
        remedies1 = tk.Label(window1, text=rem1, background="lightgreen", fg="Black", font=("", 12))
        remedies1.grid(column=0, row=1, padx=10, pady=10)

        button = tk.Button(window1, text="Exit", command=exit)
        button.grid(column=0, row=2, padx=20, pady=20)

        window1.mainloop()

    # Function to display remedies for Yellow leaf curl virus
    def vir(self):
        window1 = tk.Toplevel()
        window1.title("Remedies for Yellow Leaf Curl Virus")
        window1.geometry("650x510")
        window1.configure(background="lightgreen")

        def exit():
            window1.destroy()

        rem = "The remedies for Yellow leaf curl virus are: "
        remedies = tk.Label(window1, text=rem, background="lightgreen", fg="Brown", font=("", 15))
        remedies.grid(column=0, row=0, padx=10, pady=10)

        rem1 = " Monitor the field, handpick diseased plants and bury them. \n  Use sticky yellow plastic traps. \n  Spray insecticides such as organophosphates, carbamates during the seedling stage. \n Use copper fungicides"
        remedies1 = tk.Label(window1, text=rem1, background="lightgreen", fg="Black", font=("", 12))
        remedies1.grid(column=0, row=1, padx=10, pady=10)

        button = tk.Button(window1, text="Exit", command=exit)
        button.grid(column=0, row=2, padx=20, pady=20)

        window1.mainloop()

    # Function to display remedies for Late Blight
    def latebl(self):
        window1 = tk.Toplevel()
        window1.title("Remedies for Late Blight")
        window1.geometry("520x510")
        window1.configure(background="lightgreen")

        def exit():
            window1.destroy()

        rem = "The remedies for Late Blight are: "
        remedies = tk.Label(window1, text=rem, background="lightgreen", fg="Brown", font=("", 15))
        remedies.grid(column=0, row=0, padx=10, pady=10)

        rem1 = " Monitor the field, remove and destroy infected leaves. \n  Treat organically with copper spray. \n  Use chemical fungicides, the best of which for tomatoes is chlorothalonil."
        remedies1 = tk.Label(window1, text=rem1, background="lightgreen", fg="Black", font=("", 12))
        remedies1.grid(column=0, row=1, padx=10, pady=10)

        button = tk.Button(window1, text="Exit", command=exit)
        button.grid(column=0, row=2, padx=20, pady=20)

        window1.mainloop()

    # Function to exit the application
    def exit(self):
        self.destroy()


if __name__ == "__main__":
    app = LeafDiseaseDetectionApp()
    app.mainloop()
