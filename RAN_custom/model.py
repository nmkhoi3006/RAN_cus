import torch
from torch import nn
from layer import Conv, ResidualBlock, TrunkBranch, Conv1x1, AttentionModule

class ResidualAttentionModel(nn.Module):
    def __init__(self, in_channels, num_classes, p=1, t=2, r=1):
        super().__init__()
        self.in_conv = Conv(c1=in_channels, c2=32, k=5, s=1, act=True) #32x32

        # self.mpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) #16x16
        self.residual_1 = ResidualBlock(c1=32, c2=128, k=3, s=1, act=True) #16x16
        self.stage_1 = AttentionModule(c=128, k=3, p=p, t=t, r=r) #16x16

        self.residual_2 = ResidualBlock(c1=128, c2=256, k=3, s=2, act=True) #8x8
        self.stage_2 = AttentionModule(c=256, k=3, p=p, t=t, r=r) #8x8

        self.residual_3 = ResidualBlock(c1=256, c2=512, k=3, s=2, act=True) #4x4
        self.stage_3 = AttentionModule(c=512, k=3, p=p, t=t, r=r)

        self.residual_block = nn.Sequential(
            ResidualBlock(c1=512, c2=1024, k=3, s=1, act=True),
            ResidualBlock(c1=1024, c2=1024, k=3, s=1, act=True),
            ResidualBlock(c1=1024, c2=1024, k=3, s=1, act=True)
        )

        self.avgpool = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8)
        )
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.in_conv(x)

        x = self.residual_1(x)
        x = self.stage_1(x)

        x = self.residual_2(x)
        x = self.stage_2(x)

        x = self.residual_3(x)
        x = self.stage_3(x)

        x = self.residual_block(x)
        x = self.avgpool(x)
        x = self.fc(x.view(x.size(0), -1))
        return x


if __name__ == "__main__":
    data = torch.randn(1, 3, 32, 32)
    layer = ResidualAttentionModel(in_channels=3, num_classes=4)
    out = layer(data)
    print(out.shape)