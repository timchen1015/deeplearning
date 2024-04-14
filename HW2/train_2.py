from matplotlib import pyplot as plt
import numpy as np
import torchsummary
from torchvision.datasets import MNIST
import torch
from torchvision import transforms
import torch.nn as nn
import os
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

resnet_model = models.resnet50(pretrained=True)
# Define loss function and optimizer
loss_fn = nn.BCELoss()        #binary cross entrophy
optimizer = torch.optim.Adam(resnet_model.parameters(), lr=0.001)
# Replace the output layer for binary classification (1 node with Sigmoid activation)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Sequential(
                  nn.Linear(num_ftrs, 1),
                  nn.Sigmoid()
                  )

#torchsummary.summary(resnet_model, (3, 224, 224))
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to fit the ResNet input size
    transforms.ToTensor(),
])

from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
import os
train_dataset = datasets.ImageFolder(root='./HW2/Dataset_OpenCvDl_Hw2_Q5/dataset/training_dataset', transform=transform)
val_dataset = datasets.ImageFolder(root='./HW2/Dataset_OpenCvDl_Hw2_Q5/dataset/validation_dataset', transform=transform)
print(len(train_dataset))
print(len(val_dataset))

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
best_val_acc, best_epoch = 0,0
num_epoch = 5
avg_train_accs = []
avg_val_accs = []

#Train
print("start train")
for epoch in range(num_epoch):
    resnet_model.train()
    train_acc_history = []
    #label cat=0/dog=1
    for images, labels in train_loader:
        # Forward pass
        output = resnet_model(images)
        output = output.squeeze()
        loss = loss_fn(output, labels.float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        #compute acc
        # Apply a threshold to the outputs to convert them to binary predictions (0 or 1)

        predictions = (output > 0.5).float()
        correct_predictions = (predictions == labels).float()
        acc = correct_predictions.mean()
        train_acc_history.append(acc.item())
        print("ing")
    avg_train_acc = sum(train_acc_history) / len(train_acc_history)
    print("avg train acc = {}".format(avg_train_acc))
    avg_train_accs.append(avg_train_acc)

    #Validate
    resnet_model.eval()
    val_acc_history = []

    for images, labels in val_loader:
        with torch.no_grad():
            output = resnet_model(images)
            output = output.squeeze()
            loss = loss_fn(output, labels.float())

            predictions = (output > 0.5).float()
            correct_predictions = (predictions == labels).float()
            acc = correct_predictions.mean()
        val_acc_history.append(acc.item())          
    avg_val_acc = sum(val_acc_history) / len(val_acc_history)
    avg_val_accs.append(avg_val_acc)


    if avg_val_acc >= best_val_acc:
        print('Best model saved at epoch {}, acc: {:.4f}'.format(epoch, avg_val_acc))
        best_val_acc = avg_val_acc
        best_epoch = epoch
        torch.save(resnet_model.state_dict(), os.path.join('./HW2', 'best_model_of_5_1.pth'))
    print('epoch {} is done'.format(epoch))

#WITH ERASE
resnet_model_2 = models.resnet50(pretrained=True)
# Define loss function and optimizer
loss_fn_2 = nn.BCELoss()        #binary cross entrophy
optimizer_2 = torch.optim.Adam(resnet_model_2.parameters(), lr=0.001)
# Replace the output layer for binary classification (1 node with Sigmoid activation)
num_ftrs = resnet_model_2.fc.in_features
resnet_model_2.fc = nn.Sequential(
                  nn.Linear(num_ftrs, 1),
                  nn.Sigmoid()
                  )

#torchsummary.summary(resnet_model, (3, 224, 224))
transform_2 = transforms.Compose([
                                transforms.Resize((224, 224)),  # Resize images to fit the ResNet input size
                                transforms.ToTensor(),
                                transforms.RandomErasing(),
])

train_dataset_2 = datasets.ImageFolder(root='./HW2/Dataset_OpenCvDl_Hw2_Q5/dataset/training_dataset', transform=transform_2)
val_dataset_2 = datasets.ImageFolder(root='./HW2/Dataset_OpenCvDl_Hw2_Q5/dataset/validation_dataset', transform=transform_2)

# Create DataLoaders
train_loader_2 = DataLoader(train_dataset_2, batch_size=batch_size, shuffle=True)
val_loader_2 = DataLoader(val_dataset_2, batch_size=batch_size, shuffle=False)
best_val_acc_2, best_epoch_2 = 0,0
avg_train_accs_2 = []
avg_val_accs_2 = []

#Train
print("start train 2")
for epoch in range(num_epoch):
    resnet_model_2.train()
    train_acc_history_2 = []
    #label cat=0/dog=1
    for images, labels in train_loader_2:
        # Forward pass
        output = resnet_model_2(images)
        output = output.squeeze()
        loss = loss_fn_2(output, labels.float())
        loss.backward()
        optimizer_2.step()
        optimizer_2.zero_grad()

        #compute acc
        # Apply a threshold to the outputs to convert them to binary predictions (0 or 1)

        predictions = (output > 0.5).float()
        correct_predictions = (predictions == labels).float()
        acc_2 = correct_predictions.mean()
        train_acc_history_2.append(acc_2.item())
        print("ing")
    avg_train_acc_2 = sum(train_acc_history_2) / len(train_acc_history_2)
    print("avg train acc 2 = :{}".format(avg_train_acc_2))
    avg_train_accs_2.append(avg_train_acc_2)

    #Validate
    resnet_model_2.eval()
    val_acc_history_2 = []

    for images, labels in val_loader_2:
        with torch.no_grad():
            output = resnet_model_2(images)
            output = output.squeeze()
            loss = loss_fn_2(output, labels.float())

            predictions = (output > 0.5).float()
            correct_predictions = (predictions == labels).float()
            acc_2 = correct_predictions.mean()
        val_acc_history_2.append(acc_2.item())          
    avg_val_acc_2 = sum(val_acc_history_2) / len(val_acc_history_2)
    avg_val_accs_2.append(avg_val_acc_2)


    if avg_val_acc_2 >= best_val_acc_2:
        print('Best model saved at epoch {}, acc: {:.4f}'.format(epoch, avg_val_acc_2))
        best_val_acc_2 = avg_val_acc_2
        best_epoch_2 = epoch
        torch.save(resnet_model_2.state_dict(), os.path.join('./HW2', 'best_model_of_5_2.pth'))
    print('epoch {} is done'.format(epoch))

# Example data
classes = ['without random erasing', 'with random erasinig']
values = [best_val_acc, best_val_acc_2]  # Replace with your actual values

# Create the bar chart
bars = plt.bar(classes, values, color='lightblue')
# Create the bar chart
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 4), ha='center', va='bottom')

# Add labels and title
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison')
plt.savefig('./HW2/comparison_bar_chart.png')


