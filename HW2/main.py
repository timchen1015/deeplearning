import io
import random
import cv2  # OpenCV
import matplotlib  # Matplotlib
import PyQt5
from matplotlib import pyplot as plt
from matplotlib import image
import numpy as np
from sympy import imageset  
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
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QLabel, QVBoxLayout, QApplication, QWidget, QDialog
from PyQt5.QtGui import QPixmap, QPainter, QPen, QImage,QFont
from PyQt5.QtCore import Qt, QPoint, QBuffer
import pandas as pd
import os
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from resnet_model import resnet_model

#change path to current .py 
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())

# Create the main application
app = QtWidgets.QApplication(sys.argv) 
global pic1, num_of_circle, binary_image, model, transform, image_path
# Create the main window
window = QtWidgets.QMainWindow()
window.setWindowTitle("My Main Window")
window.setGeometry(300, 100, 1000, 800)  # (x, y, width, height)

class GraffitiBoard(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.drawing = False
        self.lastPoint = QPoint()
        self.currentPoint = QPoint()

        self.initUI()

    def initUI(self):
        self.canvas = QPixmap(350, 200)
        self.canvas.fill(Qt.black)
        self.label = QLabel(self)                               # Create a QLabel to display the canvas
        self.label.setGeometry(0, 0, 350, 200)                  # Set the geometry (position and size) of the QLabel within the GraffitiBoard widget
        self.label.setPixmap(self.canvas)

        mainLayout = QVBoxLayout(self)
        mainLayout.addWidget(self.label)

        # Set the initial position of the GraffitiBoard
        self.setGeometry(650, 60, 350, 200)

    def paintEvent(self, event):
        painter = QPainter(self.canvas)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        if self.drawing:
            local_pos = self.label.mapFromGlobal(self.mapToGlobal(self.lastPoint))
            pen = QPen(Qt.white, 5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(local_pos, self.currentPoint)

            self.lastPoint = self.currentPoint

        self.label.setPixmap(self.canvas)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.lastPoint = self.currentPoint = self.label.mapFromGlobal(event.globalPos())
            self.drawing = True

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) & self.drawing:
            self.currentPoint = self.label.mapFromGlobal(event.globalPos())
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def clearCanvas(self):
        self.canvas.fill(Qt.black)
        self.label.setPixmap(self.canvas)

    def saveCanvasToFile(self, filename):
        # Save the contents of the canvas to a file
        self.canvas.save(filename)

graffiti_board = GraffitiBoard(window)



# Create load button
buttonLD1 = QtWidgets.QPushButton("Load Image 1", window)
buttonLD1.move(70, 200)
# Create Hough Circle Transform
labelHCT1= QtWidgets.QLabel("1.Hough Circle Transform", window)
labelHCT1.setFixedWidth(200)
labelHCT1.move(185, 75)
buttonHCT1 = QtWidgets.QPushButton("1.1 Draw Contour", window)
buttonHCT1.resize(200, 25)
buttonHCT1.move(200, 100)
buttonHCT2 = QtWidgets.QPushButton("1.2 Count coins", window)
buttonHCT2.resize(200, 25)
buttonHCT2.move(200, 135)
labelHCT2= QtWidgets.QLabel("There are _ coins in the image", window)
labelHCT2.setFixedWidth(200)
labelHCT2.move(200, 170)
#Create Hist Equalization
labelHE= QtWidgets.QLabel("2.Histrogram Equalization", window)
labelHE.setFixedWidth(200)
labelHE.move(185, 325)
buttonHE = QtWidgets.QPushButton("Histrogram Equalization", window)
buttonHE.resize(200, 25)
buttonHE.move(200, 350)
#Create Morphology Operation
labelMO= QtWidgets.QLabel("3.Morphology Operation", window)
labelMO.setFixedWidth(200)
labelMO.move(185, 575)
buttonMO1 = QtWidgets.QPushButton("3.1 Closing", window)
buttonMO1.resize(200, 25)
buttonMO1.move(200, 600)
buttonMO2 = QtWidgets.QPushButton("3.2 Opening", window)
buttonMO2.resize(200, 25)
buttonMO2.move(200, 635)

#Create MNIST Classifier using VGG19
labelMNIST= QtWidgets.QLabel("4. MNIST Classifier using VGG19", window)
labelMNIST.setFixedWidth(200)
labelMNIST.move(450, 75)
buttonMNIST1 = QtWidgets.QPushButton("1.Show Model Structure", window)
buttonMNIST1.setFixedWidth(200)
buttonMNIST1.move(450, 105)
buttonMNIST2 = QtWidgets.QPushButton("2.Show Accuracy an Loss", window)
buttonMNIST2.setFixedWidth(200)
buttonMNIST2.move(450, 135)
buttonMNIST3 = QtWidgets.QPushButton("3.Predict", window)
buttonMNIST3.setFixedWidth(200)
buttonMNIST3.move(450, 165)
buttonMNIST4 = QtWidgets.QPushButton("4.Reset", window)
buttonMNIST4.setFixedWidth(200)
buttonMNIST4.move(450, 195)



#Create ResNet50
labelRN= QtWidgets.QLabel("5.ResNet50", window)
labelRN.setFixedWidth(200)
labelRN.move(450, 350)
buttonRN1 = QtWidgets.QPushButton("Load Image", window)
buttonRN1.resize(200, 25)
buttonRN1.move(450, 400)
buttonRN2 = QtWidgets.QPushButton("5.1 Show Image", window)
buttonRN2.resize(200, 25)
buttonRN2.move(450, 440)
buttonRN3 = QtWidgets.QPushButton("5.2 Show Model Structure", window)
buttonRN3.resize(200, 25)
buttonRN3.move(450, 480)
buttonRN4 = QtWidgets.QPushButton("5.3 Comparison", window)
buttonRN4.resize(200, 25)
buttonRN4.move(450, 520)
buttonRN5 = QtWidgets.QPushButton("5.4 Inference", window)
buttonRN5.resize(200, 25)
buttonRN5.move(450, 560)
labelPREDICT= QtWidgets.QLabel("", window)
labelPREDICT.setFixedWidth(200)
labelPREDICT.move(800, 650)
font = QFont()
font.setPointSize(20)
labelPREDICT.setFont(font)




def buttonLD1_clicked():
    global pic1
    pic1 = open_image_using_dialog()
    cv2.imshow("Image", pic1)
    cv2.waitKey(0)
    cv2.destroyWindow("Image")
buttonLD1.clicked.connect(buttonLD1_clicked)    


def open_image_using_dialog():
    global image_path
    options = QFileDialog.Options()
    options |= QFileDialog.ReadOnly

    image_path, _ = QFileDialog.getOpenFileName(None, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tiff);;All Files (*)", options=options)
    print(image_path)
    if image_path:
        image = cv2.imread(image_path)
    else:
        image = None
    return image

def buttonHCT1_clicked():
    global pic1, num_of_circle
    num_of_circle = 0
    # 1.RGB  to  Grayscale
    gray_image = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
    # 2.Remove Noise with Gaussian Blur
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    # Step 3: Circle Detection using HoughCircles
    circles = cv2.HoughCircles(
        blurred_image,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=40,
        param1=100,
        param2=30,
        minRadius=10,
        maxRadius=40
    )
    # Ensure circles were found
    if circles is not None:
        # Convert circle coordinates to integer
        circles = np.uint16(np.around(circles))

        # Step 4: Display Original, Processed, and Circle Center Images
        cv2.imshow("Original Image", pic1)

        # Create an image for circle centers
        circle_edge_image = pic1.copy()
        circle_center_image = np.zeros_like(pic1)
        num_of_circle = circles.shape[1]
        for i in circles[0, :]:
            # Draw the outer circle
            #cv2.circle(img, center, radius, color, thickness)
            cv2.circle(circle_edge_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(circle_center_image, (i[0], i[1]), 2, (255, 255, 255), 3)
        cv2.imshow("Processed Image", circle_edge_image)
        cv2.imshow("Circle Centers", circle_center_image)

        # Wait until a key is pressed
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No circles were detected.")
buttonHCT1.clicked.connect(buttonHCT1_clicked)

def buttonHCT2_clicked():
    global num_of_circle
    labelHCT2.setText(f"There are {num_of_circle} coins in the image")
buttonHCT2.clicked.connect(buttonHCT2_clicked)

def equalize_opencv():
    global pic1
    pic1_to_gray = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
    equalized_image_opencv = cv2.equalizeHist(pic1_to_gray)

    hist, bins = np.histogram(equalized_image_opencv.flatten(), 256, [0, 256])
    plt.bar(bins[:-1], hist, width=1, color='gray', edgecolor='black')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Grayscale Histogram')
    plt.show()    

def buttonHE_clicked():
    global pic1
    #oringinal image
    pic1_to_gray = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
    hist1, bins1 = np.histogram(pic1_to_gray.flatten(), 256, [0, 256])                  #hist:frequency bin:bin edge
    #equalze cv image
    equalized_image_opencv = cv2.equalizeHist(pic1_to_gray)
    hist2, bins2 = np.histogram(equalized_image_opencv.flatten(), 256, [0, 256])
    #equalize_manual
    pdf = hist1 / sum(hist1)
    cdf = np.cumsum(pdf)
    lookup_table = np.uint8(255 * cdf)
    equalized_image_manual = lookup_table[pic1_to_gray]  
    hist3, bins3 = np.histogram(equalized_image_manual.flatten(), 256, [0, 256]) 

    #display
    plt.figure(figsize=(12, 8))

    # 3 image
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(pic1, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(equalized_image_opencv, cmap='gray')
    plt.title('Equalized Image (OpenCV)')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(equalized_image_manual, cmap='gray')
    plt.title('Equalized Image (Manual)')
    plt.axis('off')
    # 3 histrogram
    plt.subplot(2, 3, 4)
    plt.bar(bins1[:-1], hist1, width=1, color='gray', edgecolor='black')
    plt.title('Original Image Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    plt.subplot(2, 3, 5)
    plt.bar(bins2[:-1], hist2, width=1, color='gray', edgecolor='black')
    plt.title('Equalized Image Histogram (OpenCV)')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    plt.subplot(2, 3, 6)
    plt.bar(bins3[:-1], hist3, width=1, color='gray', edgecolor='black')
    plt.title('Equalized Image Histogram (Manual)')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()
buttonHE.clicked.connect(buttonHE_clicked)

def preproceess_image():
    global pic1, binary_image
    pic1_to_gray = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
    # Apply binary thresholding
    _, binary_image = cv2.threshold(pic1_to_gray, 127, 255, cv2.THRESH_BINARY)

def dilation(image):
    kernel_size = 3
    pad_size = kernel_size // 2
    padded_image = np.pad(image, pad_size, mode="constant", constant_values=0)
    dilated_image = np.zeros_like(padded_image)
    for i in range(padded_image.shape[0]):
        for j in range(padded_image.shape[1]):
            dilated_image[i, j] = np.max(padded_image[i : i + kernel_size, j : j + kernel_size])
    return dilated_image

def erosion(image):
    kernel_size = 3
    pad_size = kernel_size // 2
    padded_image = np.pad(image, pad_size, mode="constant", constant_values=0)
    erosion_image = np.zeros_like(padded_image)
    for i in range(padded_image.shape[0]):
        for j in range(padded_image.shape[1]):
            erosion_image[i, j] = np.min(padded_image[i : i + kernel_size, j : j + kernel_size])
    return erosion_image     

#Close
def buttonMO1_clicked():
    global pic1, binary_image
    preproceess_image()
    dilated_image = dilation(binary_image)
    erosion_image = erosion(dilated_image)
    cv2.imshow("Closing", erosion_image)
buttonMO1.clicked.connect(buttonMO1_clicked)  

#Open
def buttonMO2_clicked():
    global pic1, binary_image
    preproceess_image()
    erosion_image = erosion(binary_image)
    dilated_image = dilation(erosion_image)
    cv2.imshow("Openning", dilated_image)
buttonMO2.clicked.connect(buttonMO2_clicked)

#show vgg19 with batch normalization model structure
from torchsummary import summary
def buttonMNIST1_clicked():
    global model
    model = torchvision.models.vgg19_bn(num_classes = 10)
    summary(model, (3, 32, 32))
buttonMNIST1.clicked.connect(buttonMNIST1_clicked)

#show accuracy and loss
from torchvision.datasets import MNIST
def Trainnig_process():
    global model, transform
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
    #init data set
    transform = transforms.Compose([
                                    transforms.Resize((32, 32)),
                                    transforms.Grayscale(num_output_channels=3),
                                    transforms.ToTensor(),
                                    ])
    train_set = MNIST(root='./data', train=True, download=True, transform=transform)
    val_set = MNIST(root='./data', train=False, download=True, transform=transform)

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size = 32, shuffle = True)
    valid_dataloader = torch.utils.data.DataLoader(val_set, batch_size = 32, shuffle = False)

    #init loss function
    loss_fn = nn.CrossEntropyLoss()
    #init tensorboard writer
    best_val_acc, best_epoch = 0, 0

    num_epoch = 40
    avg_train_losses = []
    avg_val_losses = []
    avg_train_accs = []
    avg_val_accs = []

    for epoch in range(num_epoch):
        #Trainning
        model.train()
        train_loss_history = []
        train_acc_history = []
        for input, target in train_dataloader:
            #convert target to one hot 
            target_hot = nn.functional.one_hot(target, num_classes = 10).float()
            target_pred = model(input)                    #forward pass
            #compute loss and do backpropagation
            loss = loss_fn(target_pred, target_hot)       #compute loss
            loss.backward()                               #backward pass
            optimizer.step()                              #update parameters
            optimizer.zero_grad()                         #want to start each new batch a new gradient
            #compute accuracy and record loss and acc
            acc = (target_pred.argmax(dim=1) == target).float().mean()
            train_loss_history.append(loss.item())        #.item(): extract a scalar value
            train_acc_history.append(acc.item())
        #Logging trainning loss and acc
        avg_train_loss = sum(train_loss_history) / len(train_loss_history)
        avg_train_losses.append(avg_train_loss)
        avg_train_acc = sum(train_acc_history) / len(train_acc_history)  #len(train_acc_history): total num of trainning iteration
        avg_train_accs.append(avg_train_acc)

        #Validating    
        model.eval()            #set the model to evaluation mode
        val_loss_history = []
        val_acc_history = []

        for input, target in valid_dataloader:
            target_hot = nn.functional.one_hot(target, num_classes=10).float()
            with torch.no_grad():                               #perform operations without calculating gradients
                target_pred = model(input)
                loss = loss_fn(target_pred, target_hot)
                acc = (target_pred.argmax(dim=1) == target).float().mean()
            val_loss_history.append(loss.item())
            val_acc_history.append(acc.item())
        #Logging validation loss and acc
        avg_val_loss = sum(val_loss_history) / len(train_loss_history)
        avg_val_losses.append(avg_val_loss)
        avg_val_acc = sum(val_acc_history) / len(val_acc_history)  #len(train_acc_history): total num of trainning iteration
        avg_val_accs.append(avg_val_acc)    

        #Save model if validation acc is better
        if avg_val_acc >= best_val_acc:
            print('Best model saved at epoch {}, acc: {:.4f}'.format(epoch, avg_val_acc))
            best_val_acc = avg_val_acc
            best_epoch = epoch
            model_save_path = '.'                                   #save in current directory
            torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pth'))

    #Plot    
    plt.figure(figsize=(20, 10))    
    #plot loss
    ax = plt.subplot(1, 2, 1)
    ax.plot(avg_train_losses, label='train loss')
    ax.plot(avg_val_losses, label='val loss')
    ax.legend()
    ax.set_title('Loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    #plot accuracy
    ax = plt.subplot(1, 2, 2)
    ax.plot(avg_train_accs, label='train acc')
    ax.plot(avg_val_accs, label='val acc')
    ax.legend()
    ax.set_title('Accuracy')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch')
    plt.savefig(os.path.join('.', 'loss_acc.png'))

    print('Best model saved at epoch {}, acc: {:.4f}'.format(best_epoch, best_val_acc))

def buttonMNIST2_clicked():
    img = Image.open('./loss_acc.png')
    plt.figure(figsize=(12, 8)) 
    plt.imshow(img)
    plt.show()
buttonMNIST2.clicked.connect(buttonMNIST2_clicked)

#predict
def buttonMNIST3_clicked():
    global model
    graffiti_board.saveCanvasToFile("handwrite.png")
    pic2 = Image.open('./handwrite.png')
    pic2 = pic2.convert("L")                            #convert to gray scale
    #load model
    ckpt = torch.load('./best_model.pth', map_location=torch.device('cpu'))
    model.load_state_dict(ckpt)
    model.eval()
    #convert img to tensor
    transform = transforms.Compose([
                                    transforms.Resize((32, 32)),
                                    transforms.Grayscale(num_output_channels=3),
                                    transforms.ToTensor(),
                                    ])
    pic2 = transform(pic2)
    pic2 = pic2.unsqueeze(0)      #add batch dimension because the model expected 4d

    with torch.no_grad():
        pred = model(pic2)
        pred = torch.softmax(pred, dim=1) #apply softmax to get probability   
    pred = pred.squeeze().numpy()
    plt.figure(figsize=(10, 10))
    plt.bar(range(10), pred)
    plt.xticks(range(10))
    plt.title('probability of each class')
    plt.xlabel('class name')
    plt.ylabel('probability')
    plt.show()    
buttonMNIST3.clicked.connect(buttonMNIST3_clicked)  

def buttonMNIST4_clicked():
    graffiti_board.clearCanvas()
buttonMNIST4.clicked.connect(buttonMNIST4_clicked)  

def load_img(self):
    image_path, _ = QFileDialog.getOpenFileName(self, 'Open file')
    if image_path == '':
        return
    self.img = Image.open(image_path)
    img = QtGui.QPixmap(image_path)
    #img = img.scaled(img, 224, 224, QtCore.Qt.KeepAspectRatio)
    self.ui.img.setPixmap(img)

label_of_pic = QtWidgets.QLabel(window)             # 放入 QLabel
label_of_pic.setGeometry(700, 400, 224, 224)        # 設定 QLabel 尺寸和位置
qpixmap = QPixmap()                                 # 建立 QPixmap 物件

#load inference image
def buttonRN1_clicked():
    global resized_pixmap
    # Open a file dialog to select a new image
    image_path, _ = QFileDialog.getOpenFileName(window, 'Open file')
    if image_path:
        # Load the new image and set it to the QLabel                
        qpixmap.load(image_path)
        resized_pixmap = qpixmap.scaled(224, 224, Qt.KeepAspectRatio)       
        label_of_pic.setPixmap(resized_pixmap)
buttonRN1.clicked.connect(buttonRN1_clicked)

#load two class image
def buttonRN2_clicked():
    print(os.getcwd())
    cat_image_path = './Dataset_OpenCvDl_Hw2_Q5/dataset/inference_dataset/Cat/8043.jpg'
    cat_image = Image.open(cat_image_path)
    cat_image = cat_image.resize((224, 224), Image.ANTIALIAS)
    dog_image_path = './Dataset_OpenCvDl_Hw2_Q5/dataset/inference_dataset/Dog/12051.jpg'
    dog_image = Image.open(dog_image_path)
    dog_image = dog_image.resize((224, 224), Image.ANTIALIAS)

    plt.figure(figsize=(12, 8))

    # 2 image
    plt.subplot(1, 2, 1)
    plt.title('Cat class')
    plt.imshow(cat_image)
    plt.subplot(1, 2, 2)
    plt.title('Dog class')
    plt.imshow(dog_image)
    plt.show()
buttonRN2.clicked.connect(buttonRN2_clicked)

#show model structure
def buttonRN3_clicked():
    global resnet_model
    torchsummary.summary(resnet_model, (3, 224, 224))

buttonRN3.clicked.connect(buttonRN3_clicked) 

#show comparison image
def buttonRN4_clicked():
    # Path to your image file
    image_path = './comparison_bar_chart.png'
    img = Image.open(image_path)
    plt.figure(figsize=(12, 8)) 
    plt.imshow(img)
    plt.show()
buttonRN4.clicked.connect(buttonRN4_clicked) 

#predict
def buttonRN5_clicked():
    global resnet_model, resized_pixmap
    ckpt = torch.load('./best_model_of_5_1.pth')
    resnet_model.load_state_dict(ckpt)
    resnet_model.eval()
    pic_to_predict = resized_pixmap.toImage()
    pic_to_predict.save('./image_to_predict.png')
    buffer = QBuffer()
    buffer.open(QBuffer.ReadWrite)
    pic_to_predict.save(buffer, "PNG")
    pic_to_predict = Image.open(io.BytesIO(buffer.data()))
    transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    ])
    pic_to_predict = transform(pic_to_predict)
    pic_to_predict = pic_to_predict.unsqueeze(0)
    with torch.no_grad():
        pred = resnet_model(pic_to_predict)
        if pred < 0.5:
            print("Cat")
            labelPREDICT.setText(f"Cat")
        elif pred >= 0.5:    
            print("Dog")
            labelPREDICT.setText(f"Dog")        
buttonRN5.clicked.connect(buttonRN5_clicked) 



# Show the main window
window.show()

# Run the application
sys.exit(app.exec_())
