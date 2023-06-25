import torch
import torch.nn as nn
from torchvision.transforms import transforms

class UnetContractingBlock(nn.Module):

    def __init__(self, in_channels):
        super(UnetContractingBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels * 2, kernel_size=(3, 3), padding=0)
        self.conv2 = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=(3, 3), padding=0)
        self.activation = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))  # Reduces the size of image by half

    def forward(self, x):
        x1 = self.activation(self.conv1(x))
        x2 = self.activation(self.conv2(x1))
        x3 = self.maxpool1(x2)

        return x3


class UnetExpandingBlock(nn.Module):

    def __init__(self, in_channels):
        super(UnetExpandingBlock, self).__init__()

        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Doubles the size of the image
        self.trans_conv1 = nn.ConvTranspose2d(in_channels, int(in_channels / 2), kernel_size=(2, 2), stride=(2, 2),
                                              padding=0)
        self.conv1 = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=(3, 3), padding=0)
        self.conv2 = nn.Conv2d(int(in_channels / 2), int(in_channels / 2), kernel_size=(3, 3), padding=0)
        self.relu = nn.ReLU()

    def forward(self, x, skip_x_uncropped):
        x1 = self.trans_conv1(x)
        skip_x = transforms.CenterCrop((x1.shape[-2], x1.shape[-1]))(skip_x_uncropped)
        x2 = torch.cat([x1, skip_x], dim=-3)  # -3 is the channel dimension
        x3 = self.conv1(x2)
        x4 = self.relu(x3)
        x5 = self.conv2(x4)
        x6 = self.relu(x5)

        return x6


class UnetModel(nn.Module):

    # According to the paper
    # in_channels = 1 (greyscale image)
    # out_channels = 2 (Number of segmentation maps / classes)
    # hidden_channels = 64
    # Modify the default values for your specific need
    def __init__(self, in_channels=1, out_channels=2, hidden_channels=64):
        super(UnetModel, self).__init__()

        self.UNET_DEBUG = False

        self.increase_feature_map_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)

        self.contracting_block1 = UnetContractingBlock(hidden_channels)
        self.contracting_block2 = UnetContractingBlock(hidden_channels * 2)
        self.contracting_block3 = UnetContractingBlock(hidden_channels * 4)
        self.contracting_block4 = UnetContractingBlock(hidden_channels * 8)

        self.expanding_block1 = UnetExpandingBlock(hidden_channels * 16)
        self.expanding_block2 = UnetExpandingBlock(hidden_channels * 8)
        self.expanding_block3 = UnetExpandingBlock(hidden_channels * 4)
        self.expanding_block4 = UnetExpandingBlock(hidden_channels * 2)

        self.decrease_feature_map_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.increase_feature_map_conv(x)
        if self.UNET_DEBUG: print('x1', x1.shape)
        x2 = self.contracting_block1(x1)
        if self.UNET_DEBUG: print('x2', x2.shape)
        x3 = self.contracting_block2(x2)
        if self.UNET_DEBUG: print('x3', x3.shape)
        x4 = self.contracting_block3(x3)
        if self.UNET_DEBUG: print('x4', x4.shape)
        x5 = self.contracting_block4(x4)
        if self.UNET_DEBUG: print('x5', x5.shape)

        x6 = self.expanding_block1(x5, x4)
        if self.UNET_DEBUG: print('x6', x6.shape)
        x7 = self.expanding_block2(x6, x3)
        if self.UNET_DEBUG: print('x7', x7.shape)
        x8 = self.expanding_block3(x7, x2)
        if self.UNET_DEBUG: print('x8', x8.shape)
        x9 = self.expanding_block4(x8, x1)
        if self.UNET_DEBUG: print('x9', x9.shape)

        x10 = self.decrease_feature_map_conv(x9)
        if self.UNET_DEBUG: print('x10', x10.shape)

        return x10


if __name__ == '__main__':
    print('Unet testing')

    model = UnetModel(in_channels=1, out_channels=2, hidden_channels=64)

    model.UNET_DEBUG = True
    input_tensor = torch.ones((1, 1, 572, 572), dtype=torch.float32)

    model(input_tensor)

    '''
    Output:
    x1 torch.Size([1, 64, 572, 572])
    x2 torch.Size([1, 128, 284, 284])
    x3 torch.Size([1, 256, 140, 140])
    x4 torch.Size([1, 512, 68, 68])
    x5 torch.Size([1, 1024, 32, 32])
    x6 torch.Size([1, 512, 60, 60])
    x7 torch.Size([1, 256, 116, 116])
    x8 torch.Size([1, 128, 228, 228])
    x9 torch.Size([1, 64, 452, 452])
    x10 torch.Size([1, 3, 452, 452])
    '''
