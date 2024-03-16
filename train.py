# train.py
import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import transforms
from tqdm import tqdm
import os

# Update the imports to use the TripletNetwork and TripletLoss
from model import SiameseNetwork, TripletLoss  # Adjust this line based on your actual implementation
from dataset import FaceRecognitionDataset

def train_epoch(train_dataloader, model, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    
    batch_iterator = tqdm(train_dataloader, desc=f'Epoch {epoch:02d}/ Training', leave=False, total=len(train_dataloader))
    
    for (anchor, positive), (_, negative) in batch_iterator:
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        
        optimizer.zero_grad()
        
        anchor_output, positive_output, negative_output = model(anchor, positive, negative)
        loss = criterion(anchor_output, positive_output, negative_output)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batch_iterator.set_description(f'Epoch {epoch:02d}/ Training. Loss: {loss.item():.4f}')

    avg_loss = total_loss / len(train_dataloader)
    return avg_loss

def validate_epoch(test_dataloader, model, criterion, device, epoch):
    model.eval()
    val_loss = 0
    
    batch_iterator = tqdm(test_dataloader, desc=f'Epoch {epoch:02d}/ Validation', leave=False, total=len(test_dataloader))
    
    with torch.no_grad():
        for (anchor, positive), (_, negative) in batch_iterator:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            anchor_output, positive_output, negative_output = model(anchor, positive, negative)
            loss = criterion(anchor_output, positive_output, negative_output)
            
            val_loss += loss.item()
            batch_iterator.set_description(f'Epoch {epoch:02d}/ Validation. Loss: {loss.item():.4f}')

    avg_val_loss = val_loss / len(test_dataloader)
    return avg_val_loss

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((160, 160), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
        transforms.RandomRotation(degrees=(0 , 30)),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.ToTensor(),
    ])
    
    train_dataset = FaceRecognitionDataset(root_dir='train_ds', transform=transform)
    test_dataset = FaceRecognitionDataset(root_dir='test_ds', transform=transform)
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

    model = SiameseNetwork(pretrained=True).to(device)

    if os.path.exists('checkpoints') and len(os.listdir('checkpoints')):
        ckpt_file = os.listdir('checkpoints')[-1]
        start = int(ckpt_file.split('.')[0].split('_')[-1])
        model.load_state_dict(torch.load(os.path.join('checkpoints' , ckpt_file), map_location=torch.device(device)))
        print(f'Found existing checkpoint: checkpoints/{ckpt_file}')
    else:
        start = 0


    criterion = TripletLoss(margin=1.0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    n_epochs = 20
    
    for epoch in range(start , n_epochs):
        train_loss = train_epoch(train_dataloader, model, criterion, optimizer, device, epoch)
        val_loss = validate_epoch(test_dataloader, model, criterion, device, epoch)
        
        scheduler.step()
        
        print(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
        torch.save(model.state_dict(), f'checkpoints/model_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    main()
