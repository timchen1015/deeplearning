from torchsummary import summary
from torchvision.datasets import MNIST
import torch
from torchvision import transforms
import torch.nn as nn
import os

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
