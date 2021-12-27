from signals_dataset import SignalDataset
import torch
from ViT import ViT
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.notebook import tqdm
from torch.optim.lr_scheduler import StepLR
import os
import matplotlib.pyplot as plt
from linformer import Linformer

root = 'dataset'
root_test = 'dataset'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    # you can add other transformations in this list
])

dataset = datasets.ImageFolder(os.path.join(root), transform=transform)
dataset_test = datasets.ImageFolder(os.path.join(root_test), transform=transform)
print(dataset)

# vedere se è come lo vogliamo fare ******
# split the dataset in train and test set
batch_size_train = 16
batch_size_test = 16
train_percentage = 0.7
test_percentage = 1 - train_percentage
train_size = int(train_percentage * len(dataset))
test_size = len(dataset) - train_size

torch.manual_seed(0)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[0:train_size])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-test_size:])

train_loader = DataLoader(dataset, batch_size = batch_size_train, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size = batch_size_test, shuffle=True)

#*************

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

'''for i, l in train_loader:
    print(type(i[0]))
    print(l)
    plt.imshow(i[0].permute(1,2,0))
    plt.show()
    plt.imshow(i[1].permute(1,2,0))
    plt.show()

    break'''



seed = 42

model = ViT(   
    image_size=224,
    patch_size=32,
    num_classes=7, 
    dim=128,
    depth = 6,
    heads=8,
    mlp_dim =256, 
    channels=3,
).to(device)

print("quaaaa")
PATH = 'model_ViT.pt'

epochs = 20
lr = 1e-3

# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
#optimizer = optim.Adam(model.parameters(), lr=lr)
optimizer = optim.AdamW(model.parameters(), lr = lr)
# scheduler
gamma = 0.7
#scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

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
        for data, label in test_loader:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(test_loader)
            epoch_val_loss += val_loss / len(test_loader)

    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )

    #save model
    torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    }, PATH)