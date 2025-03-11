import torch
import torch.nn as nn



def gaussian_kernel(kernel_size, sigma):
    """Create a 2D Gaussian kernel."""
    # Create a grid of (x, y) coordinates
    x = torch.arange(kernel_size) - (kernel_size - 1) / 2
    y = torch.arange(kernel_size) - (kernel_size - 1) / 2
    x_grid, y_grid = torch.meshgrid(x, y, indexing="ij")
    # Calculate the Gaussian function
    kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()  # Normalize
    return kernel

class Gray2RGBadaptor(nn.Module):
    def __init__(self, kernel_size=7, sigmas=[0.5,1.0,2.0], stride=2):
        super().__init__()
        self.kernels = [gaussian_kernel(kernel_size, sigma) for sigma in sigmas]
        self.conv = nn.Conv2d(in_channels=1, out_channels=3, 
                              kernel_size=kernel_size, 
                              bias=False,
                              stride=stride,
                              padding=kernel_size // 2)
        for i, kernel in enumerate(self.kernels):
            self.conv.weight.data[i, 0, :, :] = kernel

        

    def forward(self, x):
        return self.conv(x)
  

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()
        
    
    def forward(self, x):        
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = nn.ReLU()(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        
        y = x * y  # Channel-wise weighting  
        return y
    
class CBAMImagePooling(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=3):
        super().__init__()
        self.channel_attention = SEBlock(in_channels, reduction)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, padding_mode='reflect'),
            nn.Sigmoid()
        )

    def forward(self, x):
        
        
        x = self.channel_attention(x) #NCHW
        # Spatial Attention
        spatial = torch.cat([torch.max(x, dim=1, keepdim=True)[0], 
                            torch.mean(x, dim=1, keepdim=True)], dim=1)
        
        attention_weights =  self.spatial_attention(spatial)
        
    
        # Weighted Global Pooling
        weighted_sum = torch.sum(x * attention_weights, dim=(2, 3))  # Weighted sum across spatial dimensions
        normalization_factor = torch.sum(attention_weights, dim=(2, 3)) + 1e-8  # Avoid division by zero
        aggregated_features = weighted_sum / normalization_factor

        return aggregated_features

    
    
class WholeImageCBAMAttention(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cbam = CBAMImagePooling(in_channels)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # Apply CBAM for spatial and channel attention
        x = x.permute(0, 3, 1, 2) # NHWC -> NCHW
        
        x = self.cbam(x)  # [batch_size, 1024, 4, 5]
        x = x.view(x.size(0), -1)  # [batch_size, 1024]
        x = self.fc1(x) # [batch_size, num_classes]
        return x