import tkinter as tk
import adafruit_dht
import board
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

def exit_window(window):
    window.destroy()

# Load the model
def load_model():
    LR = 1e-3
    IMG_SIZE = 50
    MODEL_NAME = 'healthyvsunhealthy-{}-{}.model'.format(LR, '2conv-basic')

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
        print('Model loaded successfully!')
    else:
        print('Model not found!')

    return model

# Analyze image using loaded model
def analyze_image(image_path, model):
    IMG_SIZE = 50
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.array(img).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

    model_out = model.predict(img)[0]
    class_index = np.argmax(model_out)

    class_labels = {0: 'Healthy', 1: 'Bacterial Spot', 2: 'Yellow leaf curl virus', 3: 'Late Blight'}

    return class_labels[class_index]

# Open photo and analyze
def open_photo():
    # Select the image for analysis
    file_path = askopenfilename(initialdir='C:/Users/HP/Downloads/images', title='Select image for analysis', filetypes=[('image files', '.jpg')])
    
    # Load the model
    model = load_model()
    
    # Analyze the selected image
    disease_label = analyze_image(file_path, model)

    # Display the selected image
    img = Image.open(file_path)
    img = img.resize((300, 300))  # Resize the image
    img = ImageTk.PhotoImage(img)

    # Create a new window to display the image
    window_img = tk.Toplevel()
    window_img.title("Selected Image")
    window_img.geometry("500x500")
    window_img.configure(background="lightgreen")

    img_label = tk.Label(window_img, image=img)
    img_label.image = img
    img_label.pack(pady=10)

    # Analyze and display the disease result
    analyze_and_display_result(disease_label)




# Display remedies
def display_remedies(disease_name):
    remedies_text = {
        "Bacterial Spot": "The remedies for Bacterial Spot are:\n\n"
                          "1. Discard or destroy any affected plants.\n"
                          "2. Do not compost them.\n"
                          "3. Rotate your tomato plants yearly to prevent re-infection next year.\n"
                          "4. Use copper fungicides.",
        "Yellow leaf curl virus": "The remedies for Yellow leaf curl virus are:\n\n"
                                  "1. Monitor the field, handpick diseased plants and bury them.\n"
                                  "2. Use sticky yellow plastic traps.\n"
                                  "3. Spray insecticides such as organophosphates, carbametes during the seedling stage.\n"
                                  "4. Use copper fungicides.",
        "Late Blight": "The remedies for Late Blight are:\n\n"
                       "1. Monitor the field, remove and destroy infected leaves.\n"
                       "2. Treat organically with copper spray.\n"
                       "3. Use chemical fungicides, the best of which for tomatoes is chlorothalonil."
    }

    window = tk.Toplevel()
    window.title("Remedies")
    window.geometry("500x300")
    window.configure(background="lightgreen")

    if disease_name in remedies_text:
        remedies_label = tk.Label(window, text=remedies_text[disease_name], background="lightgreen", fg="Brown", font=("", 12))
        remedies_label.pack(padx=10, pady=10)
    else:
        no_remedies_label = tk.Label(window, text="No remedies found for this disease.", background="lightgreen", fg="Black", font=("", 12))
        no_remedies_label.pack(padx=10, pady=10)


# Analyze and display result
def analyze_and_display_result(disease_label):
    window = tk.Tk()
    window.title("Disease Detection")
    window.geometry("500x500")
    window.configure(background="lightgreen")

    # Read DHT11 sensor data
    dht11_pin = board.D4  # GPIO pin connected to the DHT11 sensor
    temperature_c, temperature_f, humidity = read_dht11_sensor(dht11_pin)

    # Display DHT11 sensor readings
    if temperature_c is not None and humidity is not None:
        temperature_label = tk.Label(window, text=f"Temperature: {temperature_c:.1f}°C / {temperature_f:.1f}°F", background="lightgreen", fg="Black", font=("", 15))
        temperature_label.grid(column=0, row=4, padx=10, pady=10)

        humidity_label = tk.Label(window, text=f"Humidity: {humidity:.1f}%", background="lightgreen", fg="Black", font=("", 15))
        humidity_label.grid(column=0, row=5, padx=10, pady=10)
    else:
        error_label = tk.Label(window, text="Failed to read DHT11 sensor data", background="lightgreen", fg="Red", font=("", 15))
        error_label.grid(column=0, row=5, padx=10, pady=10)


    status_label = tk.Label(window, text=f"Status: {disease_label}", background="lightgreen", fg="Brown", font=("", 15))
    status_label.grid(column=0, row=0, padx=10, pady=10)

    if disease_label in ["Bacterial Spot", "Yellow leaf curl virus", "Late Blight"]:
        disease_name_label = tk.Label(window, text=f"Disease Name: {disease_label}", background="lightgreen", fg="Black", font=("", 15))
        disease_name_label.grid(column=0, row=1, padx=10, pady=10)

        remedies_button = tk.Button(window, text="Remedies", command=lambda: display_remedies(disease_label))
        remedies_button.grid(column=0, row=2, padx=10, pady=10)
    else:
        healthy_label = tk.Label(window, text="Plant is healthy", background="lightgreen", fg="Black", font=("", 15))
        healthy_label.grid(column=0, row=1, padx=10, pady=10)

    exit_button = tk.Button(window, text="Exit", command=lambda: exit_window(window))
    exit_button.grid(column=0, row=3, padx=10, pady=10)

def read_dht11_sensor(pin):
    try:
        sensor = adafruit_dht.DHT11(pin)
        temperature_c = sensor.temperature
        temperature_f = temperature_c * (9 / 5) + 32
        humidity = sensor.humidity
        return temperature_c, temperature_f, humidity
    except RuntimeError as error:
        # Errors happen fairly often, DHT's are hard to read, just keep going
        print(error.args[0])
        return None, None, None
    except Exception as error:
        sensor.exit()
        raise error



# Create the UI
window = tk.Tk()
window.title("Dr. Plant")
window.geometry("500x510")
window.configure(background="lightgreen")

title = tk.Label(text="Click below to choose picture for testing disease....", background="lightgreen", fg="Brown", font=("", 15))
title.grid()

button = tk.Button(window, text="Select Image", command=open_photo)
button.grid(column=0, row=1, padx=10, pady=10)

window.mainloop()
