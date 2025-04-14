import torch
from torch import nn

def autopad(k, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size

    p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    def __init__(self, c1, c2, k, s=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k), bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, c1, c2, k, s=1, act=True):
        super().__init__()
        self.conv1 = Conv(c1, c2//4, 1, 1, act)
        self.conv2 = Conv(c2//4, c2//4, k, s, act)
        self.conv3 = Conv(c2//4, c2, 1, 1, act)

        self.add = Conv(c1, c2, 1, s, act=False) if c1 != c2 or s!=1 else nn.Identity()

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x += self.add(residual)
        return x
    
class TrunkBranch(nn.Module):
    def __init__(self, c, k, t=2, s=1, act=True):
        super().__init__()
        self.block = nn.ModuleList([Conv(c, c, k, s, act) for _ in range(t)])

    def forward(self, x):
        for layer in self.block:
            x = layer(x)
        return x
    
class Conv1x1(nn.Module):
    def __init__(self, c, k=1, s=1, act=True):
        super().__init__()
        self.conv1 = Conv(c, c, k, s, act)
        self.conv2 = Conv(c, c, k, s, act)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x
    
class MaskBranch(nn.Module):
    def __init__(self, c, k, r=1, s=1, act=True):
        super().__init__()
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_block1 = nn.ModuleList([ResidualBlock(c, c, k, s, act) for _ in range(r)])

        self.skip_connection = ResidualBlock(c, c, k, s, act)

        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.mid_residual_block = nn.ModuleList([ResidualBlock(c, c, k, s, act) for _ in range(2*r)])
        self.interpolation1 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.residual_block2 = nn.ModuleList([ResidualBlock(c, c, k, s, act) for _ in range(r)])
        self.interpolation2 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv_1x1 = Conv1x1(c, k, s, act)
    
    def forward(self, x):
        x = self.mpool1(x)
        for layer in self.residual_block1:
            x = layer(x)

        skip_connection = self.skip_connection(x)

        x = self.mpool2(x)
        for layer in self.mid_residual_block:
            x = layer(x)
        x = self.interpolation1(x)

        x += skip_connection

        for layer in self.residual_block2:
            x = layer(x)

        x = self.interpolation2(x)
        x = self.conv_1x1(x)
        return x
    
class AttentionModule(nn.Module):
    def __init__(self, c, k, p=1, t=2, r=1, s=1, act=True):
        super().__init__()
        self.pre_residual_block = nn.ModuleList([ResidualBlock(c, c, k, s, act) for _ in range(p)])

        self.trunk_branch = TrunkBranch(c, k, t, s, act)
        self.mask_branch = MaskBranch(c, k, r, s, act)

        self.post_residual_block = nn.ModuleList([ResidualBlock(c, c, k, s, act) for _ in range(p)])
    
    def forward(self, x):
        for layer in self.pre_residual_block:
            x = layer(x)

        trunk_out = self.trunk_branch(x)
        mask_out = self.mask_branch(x)

        x = (1 + mask_out) * trunk_out
        x += trunk_out
        for layer in self.post_residual_block:
            x = layer(x)
        return x

if __name__ == "__main__":
    data = torch.randn(1, 512, 32, 32)

    layer = AttentionModule(c=512, k=3)
    out = layer(data)
    print(out.shape)  # Expected output shape: (1, 2, 32, 32)