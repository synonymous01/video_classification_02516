from glob import glob
import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from tqdm import tqdm
import matplotlib.pyplot as plt



class FrameImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
    #root_dir='/work3/ppar/data/ucf101',
    root_dir='/dtu/datasets1/02516/ufc10',
    split='train', 
    transform=None
):
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
            frame = self.transform(frame)
        else:
            frame = T.ToTensor()(frame)

        return frame, label


class FrameVideoDataset(torch.utils.data.Dataset):
    def __init__(self, 
    #root_dir = '/work3/ppar/data/ucf101', 
    root_dir='/dtu/datasets1/02516/ufc10',
    split = 'train', 
    transform = None,
    stack_frames = True
):

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


class CNN3D_Network(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN3D_Network, self).__init__()
        self.convolutional = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)), 
            
            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),

            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
        )
        
        
        self.fully_connected = nn.Sequential(
            nn.Linear(128 * 1 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        x = self.convolutional(x) 
        # x = self.pool(x)          
        # x = x.view(B * T, -1)    
        x = x.view(x.size(0), -1)
        # x = x.view(B, T, -1)      
        # x = x.mean(dim=1)   

        x = self.fully_connected(x) # [B, num_classes]
        return x


def save_plot(train_accuracies, val_accuracies, filename='cnn_results.png'):
    train_data = [x * 100 for x in train_accuracies]
    val_data = [x * 100 for x in val_accuracies]
    epochs = list(range(1, len(train_data) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_data, label='Train', marker='o', color='black')
    plt.plot(epochs, val_data, label='Validation', marker='s', color='red', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('3D CNN Performance')
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def train_model_and_validate(model, optimizer, dataset_train, dataset_val, dataloader_train, dataloader_val, num_epochs=8, device='cuda'):
    train_accuracies = list()
    validation_accuracies = list()
    
    for epoch in range(num_epochs):
        train_correct = 0
        model.train()
        for minibatch_no, (data, target) in enumerate(dataloader_train):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            output = model(data)
            loss = F.cross_entropy(output, target) 
            
            loss.backward()
            optimizer.step()
            
            predicted = output.argmax(1)
            train_correct += (target == predicted).sum().cpu().item()

        val_correct = 0
        model.eval()
        with torch.no_grad():
            for data, target in dataloader_val:
                data, target = data.to(device), target.to(device)
                output = model(data)
                predicted = output.argmax(1).cpu()
                val_correct += (target == predicted.to(device)).sum().item()
        
        train_acc = train_correct / len(dataset_train)
        train_accuracies.append(train_acc)
        val_acc = val_correct / len(dataset_val)
        validation_accuracies.append(val_acc)
        print(f"Epoch {epoch+1}/{num_epochs} - Accuracy train: {100*train_acc:.1f}%\t validation: {100*val_acc:.1f}%")
        
    return train_accuracies, validation_accuracies

def save_model_weights(model, filepath='./model_weights.pt'):
    '''use this function to save your model weights to a .pt file
    MUST use .pt at the end of the filename'''
    torch.save(model.state, filepath)

def load_model_weights(filepath='./model_weights.pt'):
    '''returns trained model weights from a .pt file'''
    model = LateFusionNetwork()
    model.load_state_dict(torch.load('./{}'.format(filepath), weights_only=True))
    model.eval()
    return model

def test_model(model, dataloader_test):
    '''Computes test accuracy of given model'''
    global device
    test_correct = 0
    for data, target in dataloader_test:
        data = data.to(device)
        with torch.nograd():
            output = model(data)
        predicted = output.argmax(1).cpu()
        test_correct += (target==predicted).sum().item()

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    root_dir = '/dtu/datasets1/02516/ufc10'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform = T.Compose([T.Resize((64, 64)), T.RandomHorizontalFlip(p=0.5), T.RandomVerticalFlip(p=0.5), T.RandomRotation(270), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    dataset_train = FrameVideoDataset(root_dir=root_dir, split='train', transform=transform, stack_frames=True)
    dataset_val = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames=True)

    dataloader_train = DataLoader(dataset_train, batch_size=15, shuffle=True, num_workers=2)
    dataloader_val = DataLoader(dataset_val, batch_size=15, shuffle=False, num_workers=2)

    model = CNN3D_Network(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    train_accs, val_accs = train_model_and_validate(
        model, optimizer, dataset_train, dataset_val, dataloader_train, dataloader_val, num_epochs=30
    )

    save_plot(train_accs, val_accs, filename='cnn3d_results_norm.png')
    torch.save(model.state_dict(), './cnn3d_weights_norm.pt')