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
    root_dir='/dtu/datasets1/02516/ucf101_noleakage/frames',
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
    root_dir='/dtu/datasets1/02516/ucf101_noleakage/flows',
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

class spatial_stream(nn.Module):
    def __init__(self, num_classes=10, num_frames):
        super().__init__()
        in_channels = 3 * num_frames
        self.convolutionals = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels, 96, kernel_size=7, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # conv 2
            nn.Conv2d(96, 256, kernel_size=5, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # conv 3
            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            nn.ReLU(),
            # conv 4
            nn.Conv2d(512, 512, kernel_size=3, stride=1),
            nn.ReLU(),
            # conv 5
            nn.Conv2d(512, 512, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fully_connected = nn.Sequential(
            nn.Linear(3*3*512, 4096),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 2048),
            nn.Dropout(p=0.5),
        )

    def forward(self, x):
        x = self.convolutionals(x)
        x = self.fully_connected(x)
        return x

class temporal_stream(nn.Module):
    def __init__(self, num_classes=10, num_frames):
        super().__init__()
        in_channels = 3 * num_frames
        self.convolutionals = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels, 96, kernel_size=7, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # conv 2
            nn.Conv2d(96, 256, kernel_size=5, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # conv 3
            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            nn.ReLU(),
            # conv 4
            nn.Conv2d(512, 512, kernel_size=3, stride=1),
            nn.ReLU(),
            # conv 5
            nn.Conv2d(512, 512, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fully_connected = nn.Sequential(
            nn.Linear(3*3*512, 4096),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 2048),
            nn.Dropout(p=0.5),
        )

    def forward(self, x):
        x = self.convolutionals(x)
        x = self.fully_connected(x)
        return x


class FrameFlowDataset(torch.utils.data.Dataset):
    def __init__(self, 
    #root_dir = '/work3/ppar/data/ucf101', 
    root_dir='/dtu/datasets1/02516/ucf101_noleakage',
    split = 'train', 
    transform = None,
    stack_frames = True
):

        self.video_paths = sorted(glob(f'{root_dir}/flows/{split}/*/*/*.avi'))
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
        # video_name = video_path.split('/')[-1].split('.avi')[0]
        video_name = video_path.split('/')[-1].split('.npy')[0]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()

        # video_frames_dir = self.video_paths[idx].split('.npy')[0].replace('videos', 'frames')
        # video_frames = self.load_frames(video_frames_dir)

        # if self.transform:
        #     frames = [self.transform(frame) for frame in video_frames]
        # else:
        #     frames = [T.ToTensor()(frame) for frame in video_frames]
        
        # if self.stack_frames:
        #     frames = torch.stack(frames).permute(1, 0, 2, 3)

        flow_np = np.load(video_path)

        if flow_np.ndim == 3:
            flow_tensor = torch.from_numpy(flow_np).permute(2, 0, 1).float()
        else:
            flow_tensor = torch.from_numpy(flow_np).unsqueeze(0).float()

        if self.transform:
            flow_tensor = self.transform(flow_tensor)


        return flow_tensor, label
    
    def load_frames(self, frames_dir):
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            frame_file = os.path.join(frames_dir, f"frame_{i}.jpg")
            frame = Image.open(frame_file).convert("RGB")
            frames.append(frame)

        return frames



class two_stream(nn.Module):
    def __init__(self, spatialstream, temporalstream, num_classes, num_frames):
        super().__init__()
        self.ss = spatialstream
        self.ts = temporalstream

        self.linear_layer = nn.Sequential(
            nn.Linear(4096, 512)
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, rgb_frames, flow_stacks):
        fs = to_vector(self.ss(rgb))
        ft = to_vector(self.ts(flow))

        return self.linear_layer(torch.cat([fs, ft], dim=1))

    def freeze_backbones(self, freeze = True):
        for m in (self.ss, self.ts):
            for p in m.parameters():
                p.requires_grad = not freeze

    def train(self, mode=True):
        super().train(mode)

        for m in (self.ss, self.ts):
            if not any(p.requires_grad for p in m.parameters()):
                m.eval()
        return self


def save_plot(train_accuracies, val_accuracies, title):
    train_data = [x * 100 for x in train_accuracies]
    val_data = [x * 100 for x in val_accuracies]
    epochs = list(range(1, len(train_data) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_data, label='Train', marker='o', color='black')
    plt.plot(epochs, val_data, label='Validation', marker='s', color='red', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{title} Performance')
    plt.legend()
    plt.grid(True, linestyle=':')


    filename = f"{title.replace(' ', '_').lower()}_results.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {filename}")


def train_model_and_validate(model, optimizer, dataset_train, dataset_val, dataloader_train, dataloader_val, num_epochs=8, device='cuda',  best_path='./best_model.pt'):
    train_accuracies = list()
    validation_accuracies = list()
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training
        train_correct = 0
        model.train()
        for minibatch_no, (data, target) in enumerate(dataloader_train):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            output = model(data)
            loss = F.cross_entropy(output, target) # Uso correcto de CrossEntropy con logits
            
            loss.backward()
            optimizer.step()
            
            predicted = output.argmax(1)
            train_correct += (target == predicted).sum().cpu().item()

        # Validation
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
        
        # Keep the checkpoint with the best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"  -> new best model saved ({100*val_acc:.1f}%)")

    return train_accuracies, validation_accuracies

def save_model_weights(model, filepath='./model_weights.pt'):
    '''use this function to save your model weights to a .pt file
    MUST use .pt at the end of the filename'''
    torch.save(model.state_dict(), filepath)

def load_model_weights(model, filepath='./model_weights.pt'):
    '''loads trained weights from a .pt file into the given model'''
    model.load_state_dict(torch.load(filepath, weights_only=True))
    model.eval()
    return model

def test_model(model, dataloader_test, device):
    '''Computes test accuracy of given model'''
    model.eval()
    test_correct, n = 0, 0
    with torch.no_grad():
        for data, target in dataloader_test:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_correct += (output.argmax(1) == target).sum().item()
            n += len(target)
    test_acc = test_correct / n
    print(f"Test Accuracy: {100*test_acc:.1f}%")
    return test_acc

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    
    root_dir = '/dtu/datasets1/02516/ufc101_noleakage'
    USE_NORMALIZATION = True
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform_steps = [T.Resize((64, 64)), T.ToTensor()]
    if USE_NORMALIZATION:
        transform_steps.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    transform = T.Compose(transform_steps)
    
    dataset_train = FrameVideoDataset(root_dir=root_dir, split='train', transform=transform, stack_frames=True)
    dataset_val = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames=True)
    dataset_test = FrameVideoDataset(root_dir=root_dir, split='test', transform=transform, stack_frames=True)

    dataloader_train = DataLoader(dataset_train, batch_size=8, shuffle=True, num_workers=2)
    dataloader_val = DataLoader(dataset_val, batch_size=8, shuffle=False, num_workers=2)
    dataloader_test = DataLoader(dataset_test, batch_size=8, shuffle=False, num_workers=2)

    # ---- Change only these lines ----
    name = 'Late Fusion'                                 # used for the checkpoint and plot filenames
    #model = LateFusionNetwork(num_classes=10).to(device) # <- your model here
    # -----------------------------------------------

    file_prefix = name.replace(' ', '_').lower()
    best_path = f'./{file_prefix}_best.pt'

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    train_accs, val_accs = train_model_and_validate(
        model, optimizer, dataset_train, dataset_val, dataloader_train, dataloader_val, num_epochs=30, device=device, best_path=best_path
    )

    save_plot(train_accs, val_accs, title=name)


    # Test ONCE, with the best checkpoint (chosen on validation)
    #model = load_model_weights(model, best_path)
    #test_acc = test_model(model, dataloader_test, device)
    #print(f"Best val acc: {100*max(val_accs):.1f}% | Test acc: {100*test_acc:.1f}%")
    print("End of training and testing. Uncomment the test section to evaluate the model on the test set.")