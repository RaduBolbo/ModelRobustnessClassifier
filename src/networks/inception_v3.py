import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, branch1x1, branch5x5_reduce, branch5x5, branch3x3dbl_reduce, branch3x3dbl, branch_pool):
        super(InceptionBlock, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, branch1x1, kernel_size=1)

        self.branch5x5_1 = nn.Conv2d(in_channels, branch5x5_reduce, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(branch5x5_reduce, branch5x5, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = nn.Conv2d(in_channels, branch3x3dbl_reduce, kernel_size=1)
        self.branch3x3dbl_2 = nn.Conv2d(branch3x3dbl_reduce, branch3x3dbl, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = nn.Conv2d(branch3x3dbl, branch3x3dbl, kernel_size=3, padding=1)

        self.branch_pool = nn.Conv2d(in_channels, branch_pool, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = torch.cat([branch1x1, branch5x5, branch3x3dbl, branch_pool], 1)
        return outputs


class InceptionV3(nn.Module):
    def __init__(self, num_classes=10):
        super(InceptionV3, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2)  
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)          
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)     

        self.inception_block1 = InceptionBlock(64, 32, 32, 64, 32, 64, 32) 
        self.inception_block2 = InceptionBlock(192, 64, 48, 64, 64, 96, 64)

        self.aux_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(288, num_classes)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(288, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = self.inception_block1(x)
        x = self.inception_block2(x)

        aux = self.aux_classifier(x)

        # Fully connected layer
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, aux 


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InceptionV3(num_classes=10).to(device)
    print(model)
