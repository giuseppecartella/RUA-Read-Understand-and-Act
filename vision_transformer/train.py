import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os
import matplotlib.pyplot as plt
import numpy as np
import random


class Trainer():
    def __init__(self):
        pass

    def load_dataset(self, config):
        root =  config['dataset']
        root_test = config['dataset']

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=.5, hue=.3),
            transforms.ToTensor()
            # you can add other transformations in this list
        ])

        dataset = datasets.ImageFolder(os.path.join(os.path.dirname(__file__), root), transform=transform)
        dataset_test = datasets.ImageFolder(os.path.join(os.path.dirname(__file__), root_test), transform=transform)

        batch_size_train = config['batch_size']
        batch_size_test = config['batch_size']
        train_percentage = 0.7
        test_percentage = 1 - train_percentage
        train_size = int(train_percentage * len(dataset))
        test_size = len(dataset) - train_size

        SEED = 1234
        random.seed(SEED)
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        indices = torch.randperm(len(dataset)).tolist()
        dataset = torch.utils.data.Subset(dataset, indices[0:train_size])
        dataset_test = torch.utils.data.Subset(dataset_test, indices[-test_size:])

        train_loader = DataLoader(dataset, batch_size = batch_size_train, shuffle=True)
        test_loader = DataLoader(dataset_test, batch_size = batch_size_test, shuffle=True)
        return train_loader, test_loader


    def train_model(self, model, train_loader ,test_loader, config):
        PATH = config['model_name']
        epochs = config['epochs']
        lr = config['lr']

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr = lr)
        
        # scheduler
        gamma = 0.7
        scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

        #To Load a model
        if config['first_train']:
            old_epoch = 0
        else:
            checkpoint = torch.load(os.path.join('logs_'+config['job_name'], PATH))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            old_epoch = checkpoint['epoch']


        loss_training = []
        loss_validation = []
        accuracy_training = []
        accuracy_validation = []
        y_pred = []
        y_true = []

        for epoch in range(old_epoch, epochs):
            epoch_loss = 0
            epoch_accuracy = 0

            model.train() #training mode

            for data, label in (train_loader):
                data = data.to(device)
                label = label.to(device)

                output = model(data)
                loss = criterion(output, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc = (output.argmax(dim=1) == label).float().mean()
                epoch_accuracy += acc / len(train_loader)
                epoch_loss += loss / len(train_loader)

            with torch.no_grad():
                epoch_val_accuracy = 0
                epoch_val_loss = 0

                model.eval() #validation mode

                for data, label in test_loader:
                    data = data.to(device)
                    label = label.to(device)
                    
                    val_output = model(data)
                    val_loss = criterion(val_output, label)

                    acc = (val_output.argmax(dim=1) == label).float().mean()
                    epoch_val_accuracy += acc / len(test_loader)
                    epoch_val_loss += val_loss / len(test_loader)
                    
                    output = val_output.argmax(dim=1)
                    y_pred.extend(output) # Save Prediction

                    y_true.extend(label) # Save Truth 
            
            print(
                f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
            )

            loss_training.append(epoch_loss)
            loss_validation.append(epoch_val_loss)
            accuracy_training.append(epoch_accuracy)
            accuracy_validation.append(epoch_val_accuracy)

            fig, axes = plt.subplots(2,1)
            axes[0].plot(loss_training, label='loss_training')
            axes[0].plot(loss_validation, label='loss_validation')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()

            axes[1].plot(accuracy_training, label='accuracy_training')
            axes[1].plot(accuracy_validation, label='accuracy_validation')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend()
            fig.tight_layout(pad=3.0)

            logging_file = os.path.join('logs_'+config['job_name'], 'training_results.png')
            plt.savefig(logging_file)

            #save model
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join('logs_'+config['job_name'], PATH))