from glob import glob
import os
import random

# import cv2  # pip install opencv-python-headless
import re
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# Your original datasets (unchanged)
# ----------------------------------------------------------------------------
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
                 split='train',
                 transform=None,
                 stack_frames=True
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


# ----------------------------------------------------------------------------
# NEW: dataset for the two-stream setup (RGB frame and/or optical flow stack)
# ----------------------------------------------------------------------------
class TwoStreamDataset(torch.utils.data.Dataset):
    """
    mode='spatial'  -> (rgb[3,H,W], label)
    mode='temporal' -> (flow[2*(n_frames-1),H,W], label)
    mode='both'     -> ((rgb, flow), label)

    Optical flow is computed once with OpenCV (Farneback) from your 10 frames
    and cached on disk in `flow_cache`, so only the first epoch is slow.
    Flow channel order is [x1, y1, x2, y2, ...].
    """

    def __init__(self, root_dir='/dtu/datasets1/02516/ucf101_noleakage', split='train', mode='both',
                 img_size=128, n_frames=10, flow_bound=10.0):
        assert mode in ('spatial', 'temporal', 'both')
        self.video_dirs = sorted(d for d in glob(f'{root_dir}/frames/{split}/*/*') if os.path.isdir(d))
        df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.labels = dict(zip(df['video_name'], df['label']))
        self.mode, self.size, self.n_frames = mode, img_size, n_frames
        self.train = (split == 'train')
        self.flow_root = os.path.join(root_dir, 'flows', split)
        self.flow_bound = flow_bound
        self.norm = T.Compose([T.ToTensor(),
                               T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.video_dirs)

    def _open(self, vdir, i):
        img = Image.open(f'{vdir}/frame_{i + 1}.jpg').convert('RGB')
        return img.resize((self.size, self.size))

    def _flow(self, vdir):
        name = os.path.basename(vdir)
        cls = os.path.basename(os.path.dirname(vdir))
        natural = lambda s: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]
        files = sorted(glob(os.path.join(self.flow_root, cls, name, '*.npy')), key=natural)

        flows = []
        for f in files:
            x = torch.from_numpy(np.load(f)).float()
            if x.shape[-1] == 2:                       # (H, W, 2) -> (2, H, W)
                x = x.permute(2, 0, 1)
            H, W = x.shape[1:]
            if (H, W) != (self.size, self.size):       # resize and rescale the flow magnitudes
                x = F.interpolate(x[None], size=(self.size, self.size), mode='bilinear', align_corners=False)[0]
                x = x * torch.tensor([self.size / W, self.size / H]).view(2, 1, 1)
            flows.append(x.clamp(-self.flow_bound, self.flow_bound) / self.flow_bound)
        return torch.cat(flows, 0)                     # (2*n_flows, H, W), order x1, y1, x2, y2, ...

    def __getitem__(self, idx):
        vdir = self.video_dirs[idx]
        label = int(self.labels[os.path.basename(vdir)])
        flip = self.train and random.random() < 0.5   # same flip for rgb and flow

        rgb = flow = None
        if self.mode in ('spatial', 'both'):
            i = random.randrange(self.n_frames) if self.train else self.n_frames // 2
            rgb = self.norm(self._open(vdir, i))
            if flip:
                rgb = rgb.flip(-1)
        if self.mode in ('temporal', 'both'):
            flow = self._flow(vdir)
            if flip:
                flow = flow.flip(-1)
                flow[0::2] *= -1          # horizontal flip reverses the sign of x-flow

        if self.mode == 'spatial':
            return rgb, label
        if self.mode == 'temporal':
            return flow, label
        return (rgb, flow), label


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
class _Stream(nn.Module):
    """Shared CNN-M-2048 style body. forward() returns logits, or the 2048-d features."""

    def __init__(self, in_channels, num_classes=10, feat_dim=2048):
        super().__init__()
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
            # conv 3 (padding=1 so small inputs don't collapse to 0x0)
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # conv 4
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # conv 5
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((3, 3)),   # guarantees 3x3x512 for any input size
        )

        self.fully_connected = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 3 * 512, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, feat_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.classifier = nn.Linear(feat_dim, num_classes)   # used when training the stream alone

    def forward(self, x, return_features=False):
        feats = self.fully_connected(self.convolutionals(x))
        return feats if return_features else self.classifier(feats)


class spatial_stream(_Stream):
    """Input: RGB frame(s). num_frames=1 -> 3 input channels (standard spatial stream)."""

    def __init__(self, num_classes=10, num_frames=1):
        super().__init__(3 * num_frames, num_classes)


class temporal_stream(_Stream):
    """Input: stacked optical flow. num_frames RGB frames -> num_frames-1 flows -> 2*(num_frames-1) channels."""

    def __init__(self, num_classes=10, num_frames=10):
        super().__init__(2 * (num_frames - 1), num_classes)


class two_stream(nn.Module):
    def __init__(self, spatialstream, temporalstream, num_classes=10, feat_dim=2048):
        super().__init__()
        self.ss = spatialstream
        self.ts = temporalstream

        self.linear_layer = nn.Sequential(
            nn.Linear(2 * feat_dim, 512),   # was 2096 (typo): 2048 + 2048 = 4096
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, rgb_frames, flow_stacks):
        fs = self.ss(rgb_frames, return_features=True)
        ft = self.ts(flow_stacks, return_features=True)
        return self.linear_layer(torch.cat([fs, ft], dim=1))

    def freeze_backbones(self, freeze=True):
        for m in (self.ss, self.ts):
            for p in m.parameters():
                p.requires_grad = not freeze

    def train(self, mode=True):
        super().train(mode)
        # frozen streams stay in eval mode (BatchNorm stats fixed, dropout off)
        for m in (self.ss, self.ts):
            if not any(p.requires_grad for p in m.parameters()):
                m.eval()
        return self


# ----------------------------------------------------------------------------
# Plotting / training / evaluation
# ----------------------------------------------------------------------------
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


def _to_device(inputs, device):
    """A batch is either one tensor (single stream) or [rgb, flow] (two-stream)."""
    if isinstance(inputs, (list, tuple)):
        return [x.to(device) for x in inputs]
    return [inputs.to(device)]


def train_model_and_validate(model, optimizer, dataset_train, dataset_val, dataloader_train, dataloader_val,
                             num_epochs=8, device='cuda', best_path='./best_model.pt'):
    train_accuracies = list()
    validation_accuracies = list()
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Training
        train_correct = 0
        model.train()
        for data, target in dataloader_train:
            inputs, target = _to_device(data, device), target.to(device)
            optimizer.zero_grad()

            output = model(*inputs)
            loss = F.cross_entropy(output, target)

            loss.backward()
            optimizer.step()

            train_correct += (output.argmax(1) == target).sum().item()

        # Validation
        val_correct = 0
        model.eval()
        with torch.no_grad():
            for data, target in dataloader_val:
                inputs, target = _to_device(data, device), target.to(device)
                output = model(*inputs)
                val_correct += (output.argmax(1) == target).sum().item()

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
    model.load_state_dict(torch.load(filepath, map_location='cpu', weights_only=True))
    model.eval()
    return model


def test_model(model, dataloader_test, device):
    '''Computes test accuracy of given model'''
    model.eval()
    test_correct, n = 0, 0
    with torch.no_grad():
        for data, target in dataloader_test:
            inputs, target = _to_device(data, device), target.to(device)
            output = model(*inputs)
            test_correct += (output.argmax(1) == target).sum().item()
            n += len(target)
    test_acc = test_correct / n
    print(f"Test Accuracy: {100*test_acc:.1f}%")
    return test_acc


# ----------------------------------------------------------------------------
# Staged training
# ----------------------------------------------------------------------------
def make_loaders(root_dir, mode, img_size, batch_size=8):
    sets = {s: TwoStreamDataset(root_dir, s, mode, img_size) for s in ('train', 'val', 'test')}
    loaders = {s: DataLoader(d, batch_size=batch_size, shuffle=(s == 'train'), num_workers=2)
               for s, d in sets.items()}
    return sets, loaders


def run_stage(name, model, optimizer, root_dir, mode, img_size, epochs, device):
    """Train one stage, plot it, then reload the best-on-validation weights into `model`."""
    print(f"\n===== {name} =====")
    sets, dl = make_loaders(root_dir, mode, img_size)
    best_path = f"./{name.replace(' ', '_').lower()}_best.pt"
    train_accs, val_accs = train_model_and_validate(
        model, optimizer, sets['train'], sets['val'], dl['train'], dl['val'],
        num_epochs=epochs, device=device, best_path=best_path)
    save_plot(train_accs, val_accs, title=name)
    load_model_weights(model, best_path)
    print(f"{name}: best val acc {100*max(val_accs):.1f}%")
    return model, dl


if __name__ == '__main__':
    root_dir = '/dtu/datasets1/02516/ufc101_noleakage'
    IMG_SIZE = 128        # 128 -> exactly 3x3x512 after the conv stack (64 also works thanks to padding)
    NUM_FRAMES = 10
    NUM_CLASSES = 10
    RUN_TEST = False      # flip to True once, at the very end

    torch.manual_seed(42)
    random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ---- Stage 1: spatial stream alone (single RGB frame) ----
    ss = spatial_stream(num_classes=NUM_CLASSES).to(device)
    opt = torch.optim.Adam(ss.parameters(), lr=1e-4)
    ss, _ = run_stage('Spatial Stream', ss, opt, root_dir, 'spatial', IMG_SIZE, epochs=30, device=device)

    # ---- Stage 2: temporal stream alone (stacked optical flow) ----
    ts = temporal_stream(num_classes=NUM_CLASSES, num_frames=NUM_FRAMES).to(device)
    opt = torch.optim.Adam(ts.parameters(), lr=1e-4)
    ts, _ = run_stage('Temporal Stream', ts, opt, root_dir, 'temporal', IMG_SIZE, epochs=30, device=device)

    # ---- Stage 3a: fuse, backbones frozen, train only the fusion head ----
    model = two_stream(ss, ts, num_classes=NUM_CLASSES).to(device)
    model.freeze_backbones(True)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    model, dl = run_stage('Two Stream Head', model, opt, root_dir, 'both', IMG_SIZE, epochs=10, device=device)

    # ---- Stage 3b (optional): unfreeze, fine-tune everything with a small backbone LR ----
    model.freeze_backbones(False)
    opt = torch.optim.Adam([
        {'params': model.linear_layer.parameters(), 'lr': 1e-4},
        {'params': list(model.ss.parameters()) + list(model.ts.parameters()), 'lr': 1e-5},
    ])
    model, dl = run_stage('Two Stream Finetune', model, opt, root_dir, 'both', IMG_SIZE, epochs=10, device=device)

    # Test ONCE, with the best checkpoint (chosen on validation)
    if RUN_TEST:
        test_acc = test_model(model, dl['test'], device)
        print(f"Test acc: {100*test_acc:.1f}%")
    print("End of training. Set RUN_TEST = True to evaluate on the test set.")