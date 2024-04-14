import cv2  # OpenCV
import matplotlib  # Matplotlib
import PyQt5
from matplotlib import pyplot as plt
import numpy as np  
import torch  # PyTorch
import torch.cuda
from torchvision import transforms
import torchvision  # Torchvision
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchsummary  # Torchsummary
from tensorboardX import SummaryWriter  # Tensorboard (tensorboardX)
from PIL import Image  # Pillow
import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QLabel, QVBoxLayout, QApplication, QWidget
from PyQt5.QtGui import QPixmap
import pandas as pd
import os
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# Create the main application
app = QtWidgets.QApplication(sys.argv) 
global pic1, blue, red, green,filer_type,show_pic
filter_type = 0
# Create the main window
window = QtWidgets.QMainWindow()
window.setWindowTitle("My Main Window")
window.setGeometry(300, 100, 1000, 800)  # (x, y, width, height)

# Create a label
label = QtWidgets.QLabel("Hello, PyQt5!", window)
label.setFixedWidth(200)
label.move(150, 50)

# Create load button
buttonLD1 = QtWidgets.QPushButton("Load Image 1", window)
buttonLD1.move(70, 200)
buttonLD2 = QtWidgets.QPushButton("Load Image 2", window)
buttonLD2.move(70, 400)
# Create image processing button
labelIP= QtWidgets.QLabel("1.Image Processing", window)
labelIP.setFixedWidth(200)
labelIP.move(285, 75)
buttonIP1 = QtWidgets.QPushButton("1.1 Color Separation", window)
buttonIP1.resize(200, 25)
buttonIP1.move(300, 100)
buttonIP2 = QtWidgets.QPushButton("1.2 Color Transformation", window)
buttonIP2.resize(200, 25)
buttonIP2.move(300, 135)
buttonIP3 = QtWidgets.QPushButton("1.3 Color Extraction", window)
buttonIP3.resize(200, 25)
buttonIP3.move(300, 170)
#Create image smoothing button
labelIS= QtWidgets.QLabel("2.Image Smoothing", window)
labelIS.setFixedWidth(200)
labelIS.move(285, 325)
buttonIS1 = QtWidgets.QPushButton("2.1 Guassian blur", window)
buttonIS1.resize(200, 25)
buttonIS1.move(300, 350)
buttonIS2 = QtWidgets.QPushButton("2.2 Bilateral filter", window)
buttonIS2.resize(200, 25)
buttonIS2.move(300, 385)
buttonIS3 = QtWidgets.QPushButton("2.3 Median filter", window)
buttonIS3.resize(200, 25)
buttonIS3.move(300, 420)
#Create Edge Detection Button
labelED= QtWidgets.QLabel("3.Edge Detection", window)
labelED.setFixedWidth(200)
labelED.move(285, 575)
buttonED1 = QtWidgets.QPushButton("3.1 Sobel X", window)
buttonED1.resize(200, 25)
buttonED1.move(300, 600)
buttonED2 = QtWidgets.QPushButton("3.2 Sobel Y", window)
buttonED2.resize(200, 25)
buttonED2.move(300, 635)
buttonED3 = QtWidgets.QPushButton("3.3 Combination and Threshold", window)
buttonED3.resize(200, 25)
buttonED3.move(300, 670)
buttonED4 = QtWidgets.QPushButton("3.4 Gradient Angle", window)
buttonED4.resize(200, 25)
buttonED4.move(300, 705)

#Create transform GUI
labelTF= QtWidgets.QLabel("4.Transformation", window)
labelTF.setFixedWidth(200)
labelTF.move(650, 75)
angle_label = QtWidgets.QLabel("Rotation:", window)
angle_label.setFixedWidth(200)
angle_label.move(650, 100)
deg_label = QtWidgets.QLabel("deg", window)
deg_label.setFixedWidth(200)
deg_label.move(850, 100)
angle_input = QtWidgets.QLineEdit(window)
angle_input.setFixedWidth(100)
angle_input.move(750, 100)
#layout = QtWidgets.QVBoxLayout()
#layout.addWidget(angle_label)
#layout.addWidget(angle_input)
scaling_label = QtWidgets.QLabel("Scaling:", window)
scaling_label.setFixedWidth(200)
scaling_label.move(650, 150)
scaling_input = QtWidgets.QLineEdit(window)
scaling_input.setFixedWidth(100)
scaling_input.move(750, 150)
#layout.addWidget(scaling_label)
#layout.addWidget(scaling_input)
tx_label = QtWidgets.QLabel("Tx:", window)
tx_label.setFixedWidth(200)
tx_label.move(650, 200)
txp_label = QtWidgets.QLabel("pixel", window)
txp_label.setFixedWidth(200)
txp_label.move(850, 200)
tx_input = QtWidgets.QLineEdit(window)
tx_input.setFixedWidth(100)
tx_input.move(750, 200)

ty_label = QtWidgets.QLabel("Ty:", window)
ty_label.setFixedWidth(200)
ty_label.move(650, 250)
typ_label = QtWidgets.QLabel("pixel", window)
typ_label.setFixedWidth(200)
typ_label.move(850, 250)
ty_input = QtWidgets.QLineEdit(window)
ty_input.setFixedWidth(100)
ty_input.move(750, 250)

buttonTF = QtWidgets.QPushButton("4. Transforms", window)
buttonTF.resize(200, 25)
buttonTF.move(650, 300)

#Create VGG19
labelVGG= QtWidgets.QLabel("5.VGG19", window)
labelVGG.setFixedWidth(200)
labelVGG.move(650, 350)
buttonVGG1 = QtWidgets.QPushButton("Load Image", window)
buttonVGG1.resize(200, 25)
buttonVGG1.move(650, 400)
buttonVGG2 = QtWidgets.QPushButton("5.1 Show Agumented Images", window)
buttonVGG2.resize(200, 25)
buttonVGG2.move(650, 440)
buttonVGG3 = QtWidgets.QPushButton("5.2 Show Model Structure", window)
buttonVGG3.resize(200, 25)
buttonVGG3.move(650, 480)
buttonVGG4 = QtWidgets.QPushButton("5.3 Show Acc and Loss", window)
buttonVGG4.resize(200, 25)
buttonVGG4.move(650, 520)
buttonVGG5 = QtWidgets.QPushButton("5.4 Inference", window)
buttonVGG5.resize(200, 25)
buttonVGG5.move(650, 560)
labelVGG_Predict= QtWidgets.QLabel("Predict =", window)
labelVGG_Predict.setFixedWidth(200)
labelVGG_Predict.move(650, 580)




def buttonLD1_clicked():
    global pic1
    pic1 = open_image_using_dialog()
    cv2.imshow("Image", pic1)
    cv2.waitKey(0)
    cv2.destroyWindow("Image")
buttonLD1.clicked.connect(buttonLD1_clicked)    


def open_image_using_dialog():
    options = QFileDialog.Options()
    options |= QFileDialog.ReadOnly

    image_path, _ = QFileDialog.getOpenFileName(None, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tiff);;All Files (*)", options=options)
    print(image_path)
    if image_path:
        image = cv2.imread(image_path)
    else:
        image = None
    return image

def buttonLD2_clicked():
    label.setText("LD2 Clicked!")

buttonLD2.clicked.connect(buttonLD2_clicked)

def buttonIP1_clicked():
    global pic1, blue, green, red
    # Split the image into BGR channels
    blue, green, red = cv2.split(pic1)
    zero_array = np.zeros_like(blue)
    # Merge the channels back into a BGR image
    blue_channel = cv2.merge((blue, zero_array, zero_array))
    green_channel = cv2.merge((zero_array, green, zero_array))
    red_channel = cv2. merge((zero_array, zero_array, red))
    # Display
    cv2.imshow("B channel", blue_channel)
    cv2.imshow("G channel", green_channel)
    cv2.imshow("R channel", red_channel)
    cv2.waitKey(0)
    cv2.destroyWindow("R channel")
    cv2.waitKey(0)
    cv2.destroyWindow("G channel")
    cv2.waitKey(0)
    cv2.destroyWindow("B channel")
buttonIP1.clicked.connect(buttonIP1_clicked)

def buttonIP2_clicked():
    global pic1, blue, red, green
    I1 = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
    average_channel = (blue + red + green)/3                                    #not r,g,b channel plus, is r g b plus
    average_channel = average_channel.astype(np.uint8)                          #make sure value between 0-255
    I2 = average_channel
    cv2.imshow("opencv function", I1)
    cv2.imshow("average weighted", I2)
    cv2.waitKey(0)
    cv2.destroyWindow("average weighted")
    cv2.waitKey(0)
    cv2.destroyWindow("opencv function") 
buttonIP2.clicked.connect(buttonIP2_clicked)

def buttonIP3_clicked():
    global pic1
    hsv_image = cv2.cvtColor(pic1, cv2.COLOR_BGR2HSV)
    lower_mask = np.array([20, 100, 50])  # Lower/Upper bound
    upper_mask = np.array([90, 255, 255]) 
    mask = cv2.inRange(hsv_image, lower_mask, upper_mask)
    un_mask = cv2.bitwise_not(mask)
    remove = cv2.bitwise_and(pic1, pic1, mask=un_mask)

    cv2.imshow("I1", mask)
    cv2.imshow("I2", remove)
    cv2.waitKey(0)
    cv2.destroyWindow("I2") 
    cv2.waitKey(0)
    cv2.destroyWindow("I1") 
buttonIP3.clicked.connect(buttonIP3_clicked)

def update_image(value):
    global pic1, filter_type,show_pic
    m = value
    kernel_size = (2 * m + 1, 2 * m + 1)
    kernel_int = ((2 * m + 1) * (2 * m + 1))
    # Resize the image
    height, width, _ = pic1.shape
    resize_pic = cv2.resize(pic1, (3*width, 3*height))
    if filter_type == 0:            #guassian
        show_pic = cv2.GaussianBlur(resize_pic, kernel_size, 0)
    elif filter_type == 1:          #bilateral
        show_pic = cv2.bilateralFilter(resize_pic, kernel_int, 90, 90)
    else:                           #median blur
        show_pic = cv2.medianBlur(resize_pic, kernel_int)       
    cv2.imshow("Trackbar Example", show_pic)

def buttonIS1_clicked():
    global pic1,filter_type
    filter_type = 0
    cv2.namedWindow("Trackbar Example")
    cv2.resizeWindow("Trackbar Example", 800, 600)
    cv2.createTrackbar("m", "Trackbar Example", 0, 4, update_image)
    update_image(0)
    # Wait for user interaction
    while True:
        key = cv2.waitKey(10)
        if key == 27:  # Exit when the ESC key is pressed
            break
    cv2.destroyWindow("Trackbar Example")
    
buttonIS1.clicked.connect(buttonIS1_clicked)

def buttonIS2_clicked():
    global pic1,filter_type
    filter_type = 1
    cv2.namedWindow("Trackbar Example")
    cv2.resizeWindow("Trackbar Example", 800, 600)
    cv2.createTrackbar("m", "Trackbar Example", 0, 4, update_image)
    update_image(0)
    # Wait for user interaction
    while True:
        key = cv2.waitKey(10)
        if key == 27:  # Exit when the ESC key is pressed
            break
    cv2.destroyWindow("Trackbar Example")
        
buttonIS2.clicked.connect(buttonIS2_clicked)    

def buttonIS3_clicked():
    global pic1,filter_type
    filter_type = 2
    cv2.namedWindow("Trackbar Example")
    cv2.resizeWindow("Trackbar Example", 800, 600)
    cv2.createTrackbar("m", "Trackbar Example", 0, 4, update_image)
    update_image(0)
    # Wait for user interaction
    while True:
        key = cv2.waitKey(10)
        if key == 27:  # Exit when the ESC key is pressed
            break
    cv2.destroyWindow("Trackbar Example")
        
buttonIS3.clicked.connect(buttonIS3_clicked)

def sobel():
    global pic1,sobelx_result, sobely_result, sobelx_origin, sobely_origin
    gray_pic = cv2.cvtColor(pic1, cv2.COLOR_RGB2GRAY)
    gray_pic = cv2.GaussianBlur(gray_pic, (3, 3), 0)
    print(gray_pic.shape)
    #gray pic 6*6, sobel shape 3*3, result shape 4*4
    sobel_x = np.array([[-1, 0, 1], 
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0], 
                        [1, 2, 1]])
    rows, cols = gray_pic.shape
    sobelx_origin = np.zeros((rows-2, cols-2))                          #contain negative
    sobelx_result = np.zeros((rows-2, cols-2), dtype=np.uint8)          # only 0-255
    sobely_origin = np.zeros((rows-2, cols-2))                          #contain negative
    sobely_result = np.zeros((rows-2, cols-2), dtype=np.uint8)          # only 0-255
    # stride
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            neighborhood = gray_pic[i - 1:i + 2, j - 1:j + 2]           #gray_pic[y-1:y+2, x-1:x+2] extract 3*3 neighbor
            x = np.sum(neighborhood * sobel_x)
            sobelx_origin[i-1, j-1] = x
            x_squared = np.uint16(x ** 2)                               #prevent overflow
            sobelx_result[i-1, j-1] = np.sqrt(x_squared)
            #sobelx_result[i-1, j-1] = np.clip(x, 0, 255)  
            y = np.sum(neighborhood * sobel_y)
            sobely_origin[i-1, j-1] = y
            y_squared = np.uint16(y ** 2)                               #prevent overflow
            sobely_result[i-1, j-1] = np.sqrt(y_squared)  
            #sobely_result[i-1, j-1] = np.clip(y, 0, 255)          
   

    #sobelx_result = cv2.normalize(sobelx_result, None, 0, 255,  cv2.NORM_MINMAX, cv2.CV_8U)
    #sobely_result = cv2.normalize(sobely_result, None, 0, 255,  cv2.NORM_MINMAX, cv2.CV_8U)

def buttonED1_clicked():
    global sobelx_result
    sobel()
    cv2.imshow("Sobel_x", sobelx_result)
    cv2.waitKey(0)
    cv2.destroyWindow("Sobel_x")

buttonED1.clicked.connect(buttonED1_clicked)  

def buttonED2_clicked():
    global sobely_result
    sobel()
    cv2.imshow("Sobel_y", sobely_result)
    cv2.waitKey(0)
    cv2.destroyWindow("Sobel_y")

buttonED2.clicked.connect(buttonED2_clicked)

def buttonED3_clicked():
    global sobelx_result, sobely_result, normalized_combination
    sobel_x_y_result = np.sqrt(sobelx_result.astype(np.uint16) ** 2 + sobely_result.astype(np.uint16) ** 2)
    max_val = np.max(sobel_x_y_result)
    normalized_combination = ((sobel_x_y_result / max_val) * 255).astype(np.uint8)
    #normalized_combination = cv2.normalize(sobel_x_y_result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    threshold = 128
    thresholded_result = cv2.threshold(normalized_combination, threshold, 255, cv2.THRESH_BINARY)[1]

    show_2_img = cv2.hconcat([normalized_combination, thresholded_result])
    cv2.imshow("LEFT combination and RIGHT threshold", show_2_img)
    cv2.waitKey(0)
    cv2.destroyWindow("LEFT combination and RIGHT threshold")

buttonED3.clicked.connect(buttonED3_clicked) 

def buttonED4_clicked():
    global sobelx_origin, sobely_origin, normalized_combination
    gradient_angle = np.arctan2(sobely_origin, sobelx_origin) * 180 / np.pi
    gradient_angle[gradient_angle < 0] += 360
    #generate mask
    mask1 = ((gradient_angle >= 120) & (gradient_angle <= 180)).astype(np.uint8) * 255
    mask2 = ((gradient_angle >= 210) & (gradient_angle <= 330)).astype(np.uint8) * 255
    mask1_pic = cv2.bitwise_and(normalized_combination, mask1)
    mask2_pic = cv2.bitwise_and(normalized_combination, mask2)
    show_2_img  =cv2.hconcat([mask1_pic, mask2_pic]) 
    cv2.imshow("LEFT mask1 and RIGHT mask2", show_2_img)
    cv2.waitKey(0)
    cv2.destroyWindow("LEFT mask1 and RIGHT mask2")

buttonED4.clicked.connect(buttonED4_clicked) 

def buttonTF_clicked():
    global angle_input, scaling_input, tx_input, ty_input, pic1
    angle = angle_input.text()
    if not angle:
        angle = 0 
    scale = scaling_input.text()
    if not scale:
        scale = 0   
    tx = tx_input.text()
    if not tx:
        tx = 0
    ty = ty_input.text()    
    if not ty:
        ty = 0
    print(angle, scale, tx, ty)

    # Rotate + Scaling
    height, width = pic1.shape[:2]
    object_center_x = 240
    object_center_y = 200
    object_center = (object_center_x, object_center_y)
    # Define the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(object_center, float(angle), float(scale))
    rotated_image = cv2.warpAffine(pic1, rotation_matrix, (width, height))
    # Move object center
    new_center_x = object_center_x + int(tx)
    new_center_y = object_center_y + int(ty)
    translation_matrix = np.float32([[1, 0, new_center_x], [0, 1, new_center_y]])
    translated_image = cv2.warpAffine(rotated_image, translation_matrix, (width, height))
    print(new_center_x, new_center_y)
    cv2.imshow('Transformed Burger', translated_image)
    cv2.waitKey(0)
    cv2.destroyWindow('Transformed Burger')

buttonTF.clicked.connect(buttonTF_clicked)  

def show_images(images, filenames):
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(images[i])
        ax.axis('off')
        ax.set_title(filenames[i])
    plt.tight_layout()
    plt.show()


def buttonVGG2_clicked():
    folder_path = "./Dataset_OpenCvDl_Hw1/Q5_image/Q5_1"
    loaded_images = []
    loaded_filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            image = Image.open(os.path.join(folder_path, filename))
            loaded_images.append(image)
            root, ext = os.path.splitext(filename)                  # Split the filename into root and extension
            loaded_filenames.append(root)
    #define transform action
    horizontal_flip = transforms.RandomHorizontalFlip()
    vertical_flip = transforms.RandomVerticalFlip()
    rotation = transforms.RandomRotation(30)
    transformed_images = []
    for image in loaded_images:
        image = horizontal_flip(image)
        image = vertical_flip(image)
        image = rotation(image)
        transformed_images.append(image)
    show_images(transformed_images, loaded_filenames)        

buttonVGG2.clicked.connect(buttonVGG2_clicked)

def buttonVGG3_clicked():
    vgg19_bn = models.vgg19_bn(num_classes=10)
    torchsummary.summary(vgg19_bn, (3, 32, 32))
    #print(vgg19_bn)

buttonVGG3.clicked.connect(buttonVGG3_clicked) 



# Show the main window
window.show()

# Run the application
sys.exit(app.exec_())
