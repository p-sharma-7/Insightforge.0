import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import numpy as np
import time

# Define available models
MODEL_DICT = {
    'ResNet18': models.resnet18,
    'EfficientNet_B0': models.efficientnet_b0,
}

# Define available optimizers
OPTIMIZER_DICT = {
    'Adam': optim.Adam,
    'AdamW': optim.AdamW,  # Latest optimizer
    'SGD': optim.SGD
}

# Data augmentation using Albumentations
def get_transforms():
    return A.Compose([
        A.HorizontalFlip(),
        A.RandomResizedCrop(224, 224),
        A.ColorJitter(),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

# Custom dataset class to use Albumentations
class AlbumentationsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, label
    
    def __len__(self):
        return len(self.dataset)

# Function to select model
def select_model(model_name, num_classes):
    model = MODEL_DICT[model_name](pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Function to select optimizer
def select_optimizer(optimizer_name, parameters, lr):
    return OPTIMIZER_DICT[optimizer_name](parameters, lr=lr)

# Training function with Streamlit progress bar
def train_model(model, dataloaders, criterion, optimizer, num_epochs=5, device='cpu'):
    model = model.to(device)
    progress_bar = st.progress(0)  # Streamlit progress bar
    status_text = st.empty()
    
    for epoch in range(num_epochs):
        status_text.text(f'Epoch {epoch+1}/{num_epochs}')
        model.train()
        running_loss = 0.0
        
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        st.write(f'Epoch {epoch+1} Loss: {epoch_loss:.4f}')
        progress_bar.progress((epoch + 1) / num_epochs)

    st.success('Training Completed')
    return model

# Streamlit app setup
def main():
    st.title("Custom Model Training")

    # Sidebar for selecting model, optimizer, and hyperparameters
    st.sidebar.header("Model & Optimizer Selection")
    
    model_name = st.sidebar.selectbox('Select Model Architecture', list(MODEL_DICT.keys()))
    optimizer_name = st.sidebar.selectbox('Select Optimizer', list(OPTIMIZER_DICT.keys()))
    learning_rate = st.sidebar.slider('Select Learning Rate', 1e-6, 1e-2, 1e-3, step=1e-6, format="%.6f")
    num_epochs = st.sidebar.slider('Select Number of Epochs', 1, 50, 10)
    batch_size = st.sidebar.slider('Batch Size', 16, 64, 32)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load CIFAR-10 dataset with transformations
    st.write("Loading CIFAR-10 Dataset...")
    transform = get_transforms()
    train_data = datasets.CIFAR10(root='./data', train=True, download=True)
    train_dataset = AlbumentationsDataset(train_data, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dataloaders = {'train': train_loader}

    # When "Start Training" button is clicked
    if st.button("Start Training"):
        with st.spinner("Training the model..."):
            model = select_model(model_name, num_classes=10)
            optimizer = select_optimizer(optimizer_name, model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            trained_model = train_model(model, dataloaders, criterion, optimizer, num_epochs=num_epochs, device=device)

            st.balloons()  # Fun completion effect
            st.write("Model Training Completed!")
            st.write("You can now save or use the trained model for further evaluation.")

if __name__ == '__main__':
    main()
