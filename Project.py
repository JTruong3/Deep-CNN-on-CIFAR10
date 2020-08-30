# Base Dependencies
import numpy as np
import os
import math

# Torch Dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as model
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Import classes and functions from local files
from utils import DenseNet121
from utils import save, load

# Is there a model to Load?
Load_Model = False # Set to True if there is a model to load

current_time = datetime.now().strftime("%d:%m:%Y-%H:%M:%S")
writer = SummaryWriter('logs/{}'.format(current_time))
checkpoint_path = "checkpoint/model.pt"

# Use GPU if it is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Transforming images before training/testing
np.random.seed(101)
batch_sze = 32
transforms_step = transforms.Compose([
                                    transforms.RandomHorizontalFlip(0.5),
                                    transforms.ToTensor()])

# Download and load dataset (Set download to True if this is the first time downloading)
train = datasets.CIFAR10('./',train = True, transform = transforms_step, download = False)
test_data = datasets.CIFAR10('./', train = False, transform = transforms.ToTensor(), download = False)

indices = np.random.permutation(len(train)) # Shuffling train data
splitter = 0.85 # Train/Validation split amount
split = int(len(train) * splitter) 
train_idx = indices[:split] # Train data index
val_idx = indices[split:] # Validation Data index

# Get Subsets of train and validation data
train_subset = Subset(train, train_idx) 
val_subset = Subset(train, val_idx)

# Use dataloaders to load in the needed data in batches
train_loader = DataLoader(train_subset, batch_size = batch_sze, shuffle = True, num_workers= 4, drop_last= True)
val_loader = DataLoader(val_subset, batch_size = batch_sze, shuffle = True, num_workers = 8)
test_loader = DataLoader(test_data, batch_size = batch_sze, num_workers = 8)
        

# Model 
models = DenseNet121()
print(models)

models = nn.DataParallel(models).to(device) # Run the model and split the data between different GPUs
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(models.parameters())

if Load_Model:
    epoch_num , loss = load(checkpoint_path, models, optimizer)
    print("Epoch: {} | Loss: {}".format(epoch_num,loss))
  


epochs = 10
train_losses = []
test_losses = []
train_correct = []
test_correct = []


for i in range(epochs):
    trn_corr = 0
    tst_corr = 0
    models.train()
    train_progress_bar = tqdm(train_loader)
    # Run the training batches
    for b, (train_imgs, train_lbls) in enumerate(train_progress_bar):
        b+=1
        glob_step = (i * len(train_progress_bar)) + b
        train_imgs, train_lbls = train_imgs.to(device), train_lbls.to(device)

        # Apply the model
        y_pred = models(train_imgs) #[Batch, # Classes]
        loss = criterion(y_pred, train_lbls)
 
        # Tally the number of correct predictions
        predicted = torch.max(y_pred.data, 1)[1] #[Batch_size]
        batch_corr = (predicted == train_lbls).sum() # [# correct]
        trn_corr += batch_corr
        
        # Update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print interim results
        writer.add_embedding(y_pred, metadata = train_lbls, label_img = train_imgs, global_step= glob_step)
        if b%200 == 0:
            print(f'epoch: {i:2}  batch: {b:4} [{b:6}/1329]  loss: {loss.item():10.8f}  \
                    accuracy: {trn_corr.item()*100/(32*b):7.3f}%')
    
    # models.eval() # Set model to evaluate
    # with torch.no_grad():
    #     val_progress_bar = tqdm(val_loader)
    #     for b, (val_imgs, val_lbls) in enumerate(val_progress_bar):
    #         val_imgs, val_lbls = val_imgs.cuda(), val_lbls.cuda()

    #         prediction = models(val_imgs)
    #         loss = criterion(prediction, val_lbls)


    #         pass
        
    train_losses.append(loss)
    train_correct.append(trn_corr)

    
save(checkpoint_path, epochs, models, optimizer, loss)