# train.py
import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import transforms
from tqdm import tqdm

# Import the SiameseNetwork and ContrastiveLoss from model.py
from model import SiameseNetwork, ContrastiveLoss
# Import the FaceRecognitionDataset from dataset.py
from dataset import FaceRecognitionDataset

import numpy as np
import os

# Assuming model.py and dataset.py are in the same directory as train.py
# If not, adjust the import statements accordingly.

def train_epoch(train_dataloader, model, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0

    # Wrap train_dataloader with tqdm for real-time progress updates
    batch_iterator = tqdm(train_dataloader, desc=f'Epoch {epoch:02d}/ Training', leave=False, total=len(train_dataloader))
    
    for img1, img2, label in batch_iterator:
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        
        optimizer.zero_grad()
        
        output1, output2 = model(img1, img2)
        loss = criterion(output1, output2, label)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Optionally, update the tqdm description with the latest loss
        batch_iterator.set_description(f'Epoch {epoch:02d}/ Training. Loss: {loss.item():.4f}')

    avg_loss = total_loss / len(train_dataloader)
    return avg_loss

def validate_epoch(test_dataloader, model, criterion, device, epoch):
    model.eval()
    val_loss = 0
    
    # Wrap test_dataloader with tqdm for real-time progress updates
    batch_iterator = tqdm(test_dataloader, desc=f'Epoch {epoch:02d}/ Validation', leave=False, total=len(test_dataloader))
    
    with torch.no_grad():
        for img1, img2, label in batch_iterator:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)
            
            val_loss += loss.item()
            
            # Optionally, update the tqdm description with the latest validation loss
            batch_iterator.set_description(f'Epoch {epoch:02d}/ Validation. Loss: {loss.item():.4f}')

    avg_val_loss = val_loss / len(test_dataloader)
    return avg_val_loss

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('To Device: ', device)
    # if device == 'cuda':
    #     torch.cuda.empty_cache()
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((160, 160), interpolation=transforms.InterpolationMode.BICUBIC),
        # transforms.ColorJitter(
        #     brightness=0.2,
        #     contrast=0.2,
        #     saturation=0.2),
        transforms.ToTensor()
    ])
    
    train_dataset = FaceRecognitionDataset(root_dir='train_ds', transform=transform)
    test_dataset = FaceRecognitionDataset(root_dir='test_ds', transform=transform)
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    model = SiameseNetwork(pretrained=True).to(device)
    criterion = ContrastiveLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1, last_epoch=-1)
    
    n_epochs = 20
    
    for epoch in range(1, n_epochs + 1):
        train_loss = train_epoch(train_dataloader, model, criterion, optimizer, device, epoch)
        val_loss = validate_epoch(test_dataloader, model, criterion, device, epoch)
        
        scheduler.step()
        
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save the model checkpoint
        torch.save(model.state_dict(), f'resnetCheckpoints/model_epoch_{epoch}.pth')

if __name__ == '__main__':
    main()
