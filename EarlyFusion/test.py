from glob import glob
import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms as T
import torchvision.transforms.functional as TF
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

current_working_dir = os.getcwd()
root_dir = os.path.join(current_working_dir, "ucf101")
root_dir = os.path.join(root_dir, "ufc10")

class FrameImageDataset(torch.utils.data.Dataset):
    def __init__(self, split='train', transform=None):
        self.frame_paths = sorted(glob(f'{root_dir}/frames/{split}/*/*/*.jpg'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform
       
    def __len__(self):
        return len(self.frame_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        video_name = frame_path.split('/')[-2]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()
        
        frame = Image.open(frame_path).convert("RGB")

        if self.transform:
            frames = [self.transform(frame) for frame in video_frames]
        else:
            frames = [T.ToTensor()(frame) for frame in video_frames]

        # Sorteia UM único ângulo para todas as frames deste vídeo (apenas no split de treino)
        if self.split == 'train':
            angle = random.uniform(-15, 15)
            frames = [TF.rotate(frame, angle) for frame in frames]

        if self.stack_frames:
            frames = torch.stack(frames).permute(1, 0, 2, 3)

class FrameVideoDataset(torch.utils.data.Dataset):
    def __init__(self, split='train', transform=None, stack_frames=True):
        self.video_paths = sorted(glob(f'{root_dir}/videos/{split}/*/*.avi'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform
        self.stack_frames = stack_frames
        self.n_sampled_frames = 10

    def __len__(self):
        return len(self.video_paths)
    
    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = video_path.split('/')[-1].split('.avi')[0]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()

        video_frames_dir = self.video_paths[idx].split('.avi')[0].replace('videos', 'frames')
        video_frames = self.load_frames(video_frames_dir)

        if self.transform:
            frames = [self.transform(frame) for frame in video_frames]
        else:
            frames = [T.ToTensor()(frame) for frame in video_frames]
            
        if self.split == 'train':
            angle = random.uniform(-15, 15)
            frames = [TF.rotate(frame, angle) for frame in frames]
        
        if self.stack_frames:
            frames = torch.stack(frames).permute(1, 0, 2, 3)

        return frames, label
    
    def load_frames(self, frames_dir):
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            frame_file = os.path.join(frames_dir, f"frame_{i}.jpg")
            frame = Image.open(frame_file).convert("RGB")
            frames.append(frame)

        return frames

class Network(nn.Module):
    def __init__(self, num_classes=10):
            super(Network, self).__init__()
            # Extractor convolucional 2D ampliado y con Batch Normalization
            self.convolutional = nn.Sequential(
                       nn.Conv2d(30, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2), # Fixed: 30 channels
                       nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                       nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
                       nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
                   )
            self.pool = nn.AdaptiveAvgPool2d((2, 2))
            self.fully_connected = nn.Sequential(
                nn.Linear(128 * 2 * 2, 256), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(256, num_classes),
            )
    
    def forward(self, x):
        # O x que chega aqui tem formato: [Batch, 3, 10, 64, 64]
        batch_size = x.size(0)
        height = x.size(3)
        width = x.size(4)
        
        # ACHATAR canais e frames: [Batch, 30, 64, 64]
        x = x.view(batch_size, -1, height, width) 
        
        x = self.convolutional(x)
        x = self.pool(x)          
        x = x.view(batch_size, -1) # Flatten para fully connected

        x = self.fully_connected(x)
        return x

def plot_data(train_accuracies, validation_accuracies, title='Accuracy per epoch, our architecture'):
    #from 0-1 to 0-100
    train_data = [x * 100 for x in train_accuracies]
    test_data = [x * 100 for x in validation_accuracies]
    
    epochs = list(range(1, len(train_data) + 1))
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_data, color='black', linestyle='-', marker='o',
             label='Train', linewidth=2, markersize=7)
    plt.plot(epochs, test_data, color='red', linestyle='--', marker='s',
             label='Test', linewidth=2, markersize=7)
             
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    
    plt.xticks(epochs)
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    plt.grid(True, linestyle=':', color='lightgray', linewidth=1)
    
    plt.legend(fontsize=12, loc='lower right', framealpha=1, edgecolor='lightgray')
    
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        
    plt.title(title, fontsize=16, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.show()

def save_model_weights(model, filepath='./model_weights.pt'):
    '''use this function to save your model weights to a .pt file
    MUST use .pt at the end of the filename'''
    torch.save(model.state_dict(), filepath)

def load_model_weights(filepath='./model_weights.pt'):
    '''returns trained model weights from a .pt file'''
    model = Network(num_classes=10)  
    model.load_state_dict(torch.load(filepath, weights_only=True))
    model.eval()
    return model

def train_model_and_validate(model, optimizer, dataset_train, dataset_val, dataloader_train, dataloader_val, num_epochs=8):
    '''trains model and prints accuracies'''
    import torch.nn.functional as F
    global device
    train_accuracies = list()
    validation_accuracies = list()
    
    for epoch in tqdm(range(num_epochs), unit='epoch'):
        train_correct = 0
        model.train()
        for minibatch_no, (data, target) in tqdm(enumerate(dataloader_train), total=len(dataloader_train), leave=False):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            predicted = output.argmax(1)
            train_correct += (target==predicted).sum().cpu().item()

        # Validation phase
        val_correct = 0
        model.eval()
        for data, target in dataloader_val:
            data, target = data.to(device), target.to(device) 
            with torch.no_grad(): 
                output = model(data)
            predicted = output.argmax(1)
            val_correct += (target==predicted).sum().item()

        train_acc = train_correct / len(dataset_train)
        val_acc = val_correct / len(dataset_val)
        
        train_accuracies.append(train_acc)
        validation_accuracies.append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} Accuracy train: {100*train_acc:.1f}%\t validation: {100*val_acc:.1f}%")
        
    return train_accuracies, validation_accuracies

def test_model(model, dataloader_test, dataset_test):
    '''Computes and prints final test accuracy'''
    global device
    test_correct = 0
    model.eval()
    for data, target in dataloader_test:
        data, target = data.to(device), target.to(device)
        with torch.no_grad(): 
            output = model(data)
        predicted = output.argmax(1)
        test_correct += (target==predicted).sum().item()
    
    test_acc = test_correct / len(dataset_test)
    print("\n" + "="*40)
    print(f"FINAL TEST ACCURACY: {100*test_acc:.1f}%")
    print("="*40 + "\n")
    return test_acc


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import torch.optim as optim

    # 1. Setup Device (Uses your NVIDIA GPU if available)
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Setup Transforms and Datasets
    transform = T.Compose([
        T.Resize((64, 64)), 
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    
    # Instantiate Train, Validation, and Test datasets using Early Fusion (stack_frames=True)
    print("Loading datasets...")
    train_dataset = FrameVideoDataset(split='train', transform=transform, stack_frames=True)
    val_dataset   = FrameVideoDataset(split='val', transform=transform, stack_frames=True)
    test_dataset  = FrameVideoDataset(split='test', transform=transform, stack_frames=True) 

    # 3. Setup DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 4. Initialize Model and Optimizer
    model = Network(num_classes=10).to(device)
    
    # Adam optimizer is usually a good default
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. Train the Model 
    print("\nStarting Training...")
    train_accs, val_accs = train_model_and_validate(
        model=model, 
        optimizer=optimizer, 
        dataset_train=train_dataset, 
        dataset_val=val_dataset, 
        dataloader_train=train_loader, 
        dataloader_val=val_loader, 
        num_epochs=30
    )

    # 6. Plot the Training/Validation Curves
    plot_data(train_accs, val_accs, title='Early Fusion Accuracy')

    # 7. Test the Model 
    print("Evaluating on Test Set...")
    test_model(model, test_loader, test_dataset)
    
    # 8. Save the weights
    save_model_weights(model, filepath='early_fusion_weights.pt')
    print("Model weights saved!")