import torch
import torch.nn as nn

class PerFrameNetwork(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.convolutional = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),  # [B, 32, H/2, W/2]
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),  # [B, 64, H/4, W/4]
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),  # [B, 128, H/8, W/8]
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),  # [B, 128, H/16, W/16]
        )
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.fully_connected = nn.Sequential(
            nn.Linear(128 * 2 * 2, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def classify_frames(self, x):            # x: [N, 3, H, W]
        x = self.pool(self.convolutional(x)).flatten(1)
        return self.fully_connected(x)       # [N, num_classes] logits

    def forward(self, x):
        # Training: single frames [B, 3, H, W] -> logits per frame
        if x.dim() == 4:
            return self.classify_frames(x)

        # Validation/test: whole videos [B, 3, T, H, W]
        B, C, T, H, W = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # every frame separately
        probs = self.classify_frames(frames).softmax(dim=1)         # [B*T, classes]
        return probs.view(B, T, -1).mean(dim=1)                     # average over the T frames


class LateFusionNetwork(nn.Module):
    def __init__(self, num_classes=10):
        super(LateFusionNetwork, self).__init__()
        # Extractor convolucional 2D ampliado y con Batch Normalization
        self.convolutional = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)), 
            
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)) 
        )
        
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        
        self.fully_connected = nn.Sequential(
            nn.Linear(128 * 2 * 2, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Batch, Channels, Time, Height, Width -> [B, 3, 10, 64, 64]
        B, C, T, H, W = x.shape
        
        # Every frame on parallel
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        
        x = self.convolutional(x) 
        x = self.pool(x)          
        x = x.view(B * T, -1)    
            
        x = x.view(B, T, -1)      
        x = x.mean(dim=1)   

        x = self.fully_connected(x) # [B, num_classes]
        return x

class EarlyFusionNetwork(nn.Module):
    def __init__(self, num_classes=10, num_frames=10):
        super().__init__()
        in_channels = 3 * num_frames

        self.convolutional = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.fully_connected = nn.Sequential(
            nn.Linear(128 * 2 * 2, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: [B, 3, T, H, W]
        B, C, T, H, W = x.shape
        x = x.reshape(B, C * T, H, W)    # [B, 3*T, H, W]
        x = self.convolutional(x)
        x = self.pool(x)                 # [B, 128, 2, 2]
        x = torch.flatten(x, 1)          # [B, 512]
        return self.fully_connected(x)   # [B, num_classes]

class CNN3D_Network(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.convolutional = nn.Sequential(
            nn.Conv3d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d((2, 2, 2)),
            nn.ReLU(),
            nn.MaxPool3d(pool),

            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d((2, 2, 2)),
            nn.ReLU(),
            nn.MaxPool3d(pool),

            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d((1, 2, 2)),
            nn.ReLU(),
            nn.MaxPool3d(pool),

            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d((1, 2, 2)),
            nn.ReLU(),
            nn.MaxPool3d(pool),
        )

        self.pool = nn.AdaptiveAvgPool3d((1, 2, 2))   

        self.fully_connected = nn.Sequential(          
            nn.Linear(128 * 2 * 2, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: [B, 3, T, H, W]
        x = self.convolutional(x)
        x = self.pool(x)                 # [B, 128, 1, 2, 2]
        x = torch.flatten(x, 1)          # [B, 512]
        return self.fully_connected(x)   # [B, num_classes]