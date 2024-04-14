import torch.nn as nn
import torchsummary
import torchvision.models as models 
resnet_model = models.resnet50(pretrained=True)
# Replace the output layer for binary classification (1 node with Sigmoid activation)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Sequential(
                                nn.Linear(num_ftrs, 1),
                                nn.Sigmoid()
                                )

#torchsummary.summary(resnet_model, (3, 224, 224))