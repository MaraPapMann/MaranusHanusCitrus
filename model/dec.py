import torch as T
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channel_num, dilation=1, group=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm1 = nn.InstanceNorm2d(channel_num, affine=True)
        self.conv2 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm2 = nn.InstanceNorm2d(channel_num, affine=True)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.relu(x+y)


class HidingRes(nn.Module):
    def __init__(self, in_c=4, out_c=3, only_residual=False, requires_grad=True):
        super(HidingRes, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 128, 3, 1, 1, bias=False)
        self.norm1 = nn.InstanceNorm2d(128, affine=True)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.norm2 = nn.InstanceNorm2d(128, affine=True)
        self.conv3 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)
        self.norm3 = nn.InstanceNorm2d(128, affine=True)

        self.res1 = ResidualBlock(128, dilation=2)
        self.res2 = ResidualBlock(128, dilation=2)
        self.res3 = ResidualBlock(128, dilation=2)
        self.res4 = ResidualBlock(128, dilation=2)
        self.res5 = ResidualBlock(128, dilation=4)
        self.res6 = ResidualBlock(128, dilation=4)
        self.res7 = ResidualBlock(128, dilation=4)
        self.res8 = ResidualBlock(128, dilation=4)
        self.res9 = ResidualBlock(128, dilation=1)

        self.deconv3 = nn.ConvTranspose2d(128, 128, 4, 2, 1)
        self.norm4 = nn.InstanceNorm2d(128, affine=True)
        self.deconv2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.norm5 = nn.InstanceNorm2d(128, affine=True)
        self.deconv1 = nn.Conv2d(128, out_c, 1)
        self.only_residual = only_residual
        self.sigmoid = nn.Sigmoid()


        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False   


    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        y = F.relu(self.norm2(self.conv2(y)))
        y = F.relu(self.norm3(self.conv3(y)))

        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.res6(y)
        y = self.res7(y)
        y = self.res8(y)
        y = self.res9(y)


        y = F.relu(self.norm4(self.deconv3(y)))
        y = F.relu(self.norm5(self.deconv2(y)))
        if self.only_residual:
            y = self.deconv1(y)
        else:
            y = self.sigmoid(self.deconv1(y))

        return y


class RevealNet(nn.Module):
    def __init__(self, input_nc:int, output_nc:int, nhf:int=64, norm_layer:nn.Module=None, 
        output_function:nn.Module=nn.Sigmoid) -> None:
        super(RevealNet, self).__init__()
        # input is (3) x 256 x 256

        self.conv1 = nn.Conv2d(input_nc, nhf, 3, 1, 1)        
        self.conv2 = nn.Conv2d(nhf, nhf * 2, 3, 1, 1)
        self.conv3 = nn.Conv2d(nhf * 2, nhf * 4, 3, 1, 1)
        self.conv4 = nn.Conv2d(nhf * 4, nhf * 2, 3, 1, 1)
        self.conv5 = nn.Conv2d(nhf * 2, nhf, 3, 1, 1)
        self.conv6 = nn.Conv2d(nhf, output_nc, 3, 1, 1)
        self.output=output_function()
        self.relu = nn.ReLU(True)

        self.norm_layer = norm_layer
        if norm_layer != None:
            self.norm1 = norm_layer(nhf)
            self.norm2 = norm_layer(nhf*2)
            self.norm3 = norm_layer(nhf*4)
            self.norm4 = norm_layer(nhf*2)
            self.norm5 = norm_layer(nhf)

    def forward(self, input:T.Tensor) -> T.Tensor:
        if self.norm_layer != None:
            x=self.relu(self.norm1(self.conv1(input)))
            x=self.relu(self.norm2(self.conv2(x)))
            x=self.relu(self.norm3(self.conv3(x)))
            x=self.relu(self.norm4(self.conv4(x)))
            x=self.relu(self.norm5(self.conv5(x)))
            x=self.output(self.conv6(x))
        else:
            x=self.relu(self.conv1(input))
            x=self.relu(self.conv2(x))
            x=self.relu(self.conv3(x))
            x=self.relu(self.conv4(x))
            x=self.relu(self.conv5(x))
            x=self.output(self.conv6(x))

        return x