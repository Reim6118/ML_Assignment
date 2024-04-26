import torch
import torch.nn as nn
from torchvision import datasets, utils
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import math
import wandb
from sklearn.metrics import f1_score, accuracy_score




def train(model,transform,config,device):
    
    #import training data
    Screw_train_path = config['Screw_Img_path']+"/archive/train"
    Screw_train = datasets.ImageFolder(Screw_train_path,transform)
    dataset_size = len(Screw_train)
    #split into training and validation
    train_size = int(config['train_ratio'] * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(Screw_train, [train_size, val_size]) 
    #load data
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

    #just some checking, nothing important
    # for X, y in train_loader:
    #     print("Shape of X:  ", X.shape)
    #     print("Shape of y: ", y.shape, y.dtype)
      
    #     img = utils.make_grid(X, nrow=16)
    #     plt.imshow(img.numpy().transpose((1, 2, 0)))
    #     plt.show()

    #     break

    #Using binary cross entropy for a binary classification model
    loss_fn = nn.BCELoss()
    #Tried couple learning rate, got best result with 1e-4
    learning_rate = config['learning_rate']
    #go with the most basic adam optimizer 
    optimizer = torch.optim.Adam(model.parameters(),learning_rate)
    if not os.path.isdir('./models'):
        os.mkdir('./models') 
    # initialize some parameters
    n_epochs, best_loss, step, early_stop_count = config['epochs'], math.inf, 0, 0

    for epoch in range(n_epochs):
        model.train() 
        loss_record = []

        # use  tqdm to visualize training progress in ide
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            optimizer.zero_grad()               
            x, y = x.to(device), y.to(device)   
            pred = model(x)
            loss = loss_fn(pred,y.float())
            loss.backward()                     
            optimizer.step()                    
            step += 1
            loss_record.append(loss.detach().item())
            # record the traning/val/test loss/f1sore/accuracy with wandb
            wandb.log({"train_loss": loss})

            # Display current epoch number and loss on tqdm progress bar
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record)/len(loss_record)

        # start validation with validation set
        model.eval() 
        true_labels = []
        predicted_labels = []
        loss_record = []
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = loss_fn(pred,y.float())
                wandb.log({"val_loss": loss})
            loss_record.append(loss.item())

            y_numpy = y.cpu().numpy()
            # tensor >=0.5 will become 1, else 0
            pred_numpy = (pred.cpu().numpy() >= 0.5).astype(int) 
            true_labels.extend(y_numpy)
            predicted_labels.extend(pred_numpy)
        #calculate f1 and accuracy with sklearn.metrics
        f1 = f1_score(true_labels, predicted_labels)
        accuracy = accuracy_score(true_labels, predicted_labels)
        # record into wandb
        wandb.log({"Val_F1 Score": f1})
        wandb.log({"Val_Accuracy": accuracy})

        mean_valid_loss = sum(loss_record)/len(loss_record)
        print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')

        #saving best model with lowest validation mean loss
        if mean_valid_loss < best_loss :
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), './models/model_best.ckpt') # Save your best model
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1
        # also save the last model
        if epoch >= config['epochs']-1 :           
            torch.save(model.state_dict(), './models/model_last.ckpt') 
            print("Saving last model....")
        # early stop when the loss is not getting lower
        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, halting the training session...')
            torch.save(model.state_dict(), './models/model_last.ckpt')
            print("Saving last model....")
            return
    
        

