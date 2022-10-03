import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Dense_rain(nn.Module):
    def __init__(self):
        super(Dense_rain, self).__init__()

        # self.conv_refin=nn.Conv2d(9,20,3,1,1)
        self.conv_refin=nn.Conv2d(47,47,3,1,1)

        self.tanh=nn.Tanh()


        self.conv1010 = nn.Conv2d(47, 2, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(47, 2, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(47, 2, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(47, 2, kernel_size=1,stride=1,padding=0)  # 1mm

        self.refine3= nn.Conv2d(47+8, 3, kernel_size=3,stride=1,padding=1)

        self.refineclean1= nn.Conv2d(3, 8, kernel_size=7,stride=1,padding=3)
        self.refineclean2= nn.Conv2d(8, 3, kernel_size=3,stride=1,padding=1)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)

        self.sigmoid=nn.Sigmoid()


        self.batchnorm20=nn.BatchNorm2d(20)
        self.batchnorm1=nn.BatchNorm2d(1)

        self.dense0=Dense_base_down0()
        self.dense1=Dense_base_down1()
        self.dense2=Dense_base_down2()

    def forward(self, x):
        ## 256x256
        # label_d11 = torch.FloatTensor(1)
        # label_d11 = Variable(label_d11.cuda())
        shape_out = x.data.size()
        sizePatchGAN = shape_out[3]

        # label_result=1
        # label_result=float(label_d.data.cpu().float().numpy())

        # label_d11.data.resize_((1, 1, sizePatchGAN, sizePatchGAN)).fill_(label_result)
        label_d11 = Variable(torch.ones((shape_out[0], 1 ,sizePatchGAN, sizePatchGAN)).cuda())

        x1=torch.cat([x,label_d11],1)

        x3=self.dense2(x)
        x2=self.dense1(x)
        x1=self.dense0(x)


        label_d11 = torch.FloatTensor(1)
        label_d11 = Variable(label_d11.cuda())
        shape_out = x3.data.size()
        sizePatchGAN = shape_out[3]

        # label_result=1
        # label_result=float(label_d.data.cpu().float().numpy())

        # label_d11.data.resize_((1, 8, sizePatchGAN, sizePatchGAN)).fill_(label_result)
        label_d11 = Variable(torch.ones((shape_out[0], 8, sizePatchGAN, sizePatchGAN)).cuda())

        x8=torch.cat([x1,x,x2,x3,label_d11],1)
        # x8=torch.cat([x1,x,x2,x3],1)
        # print(x8.size())

        x9=self.relu((self.conv_refin(x8)))

        shape_out = x9.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(x9, 32)
        x102 = F.avg_pool2d(x9, 16)
        x103 = F.avg_pool2d(x9, 8)
        x104 = F.avg_pool2d(x9, 4)

        x1010 = self.upsample(self.relu((self.conv1010(x101))), size=shape_out)
        x1020 = self.upsample(self.relu((self.conv1020(x102))), size=shape_out)
        x1030 = self.upsample(self.relu((self.conv1030(x103))), size=shape_out)
        x1040 = self.upsample(self.relu((self.conv1040(x104))), size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
        residual = self.tanh(self.refine3(dehaze))
        clean = x-residual
        # clean1=self.relu(self.refineclean1(clean))
        # clean2=self.tanh(self.refineclean2(clean1))


        return clean
    

class Dense_base_down0(nn.Module):
    def __init__(self):
        super(Dense_base_down0, self).__init__()

        self.dense_block1=BottleneckBlock(3,5)
        self.trans_block1=TransitionBlock1(8,4)

        ############# Block2-down 32-32  ##############
        self.dense_block2=BottleneckBlock(4,8)
        self.trans_block2=TransitionBlock3(12,12)

        ############# Block3-down  16-16 ##############
        self.dense_block3=BottleneckBlock(12,4)
        self.trans_block3=TransitionBlock3(16,12)

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock(12,4)
        self.trans_block4=TransitionBlock3(16,12)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock(24,8)
        self.trans_block5=TransitionBlock3(32,4)

        self.dense_block6=BottleneckBlock(8,8)
        self.trans_block6=TransitionBlock(16,4)

        self.conv11 = nn.Conv2d(4, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv21 = nn.Conv2d(12, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv31 = nn.Conv2d(12, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv41 = nn.Conv2d(24, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv51 = nn.Conv2d(8, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv61 = nn.Conv2d(8, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.upsample = F.upsample_nearest
        self.relu=nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        ## 256x256
        x1=self.dense_block1(x)
        x1=self.trans_block1(x1)

        ###  32x32
        x2=(self.dense_block2(x1))
        x2=self.trans_block2(x2)

        # print x2.size()
        ### 16 X 16
        x3=(self.dense_block3(x2))
        x3=self.trans_block3(x3)

        ## Classifier  ##

        x4=(self.dense_block4(x3))
        x4=self.trans_block4(x4)

        x4=torch.cat([x4,x2],1)

        x5=(self.dense_block5(x4))
        x5=self.trans_block5(x5)

        # x5=x5+x1
        x5=torch.cat([x5,x1],1)

        x6=(self.dense_block6(x5))
        x6=(self.trans_block6(x6))
        shape_out = x6.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]
        x11 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)
        x21 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)
        x31 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)
        x41 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)
        x51 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)



        x6=torch.cat([x6,x51,x41,x31,x21,x11,x],1)


        return x6


class Dense_base_down1(nn.Module):
    def __init__(self):
        super(Dense_base_down1, self).__init__()

        self.dense_block1=BottleneckBlock1(3,13)
        self.trans_block1=TransitionBlock1(16,8)

        ############# Block2-down 32-32  ##############
        self.dense_block2=BottleneckBlock1(8,16)
        self.trans_block2=TransitionBlock1(24,16)

        ############# Block3-down  16-16 ##############
        self.dense_block3=BottleneckBlock1(16,16)
        self.trans_block3=TransitionBlock3(32,16)

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock1(16,16)
        self.trans_block4=TransitionBlock3(32,16)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock1(32,8)
        self.trans_block5=TransitionBlock(40,8)

        self.dense_block6=BottleneckBlock1(16,8)
        self.trans_block6=TransitionBlock(24,4)


        self.conv_refin=nn.Conv2d(11,20,3,1,1)
        self.tanh=nn.Tanh()


        self.conv11 = nn.Conv2d(8, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv21 = nn.Conv2d(16, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv31 = nn.Conv2d(16, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv41 = nn.Conv2d(32, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv51 = nn.Conv2d(16, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv61 = nn.Conv2d(8, 1, kernel_size=3,stride=1,padding=1)  # 1mm

        self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)
        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)


        self.batchnorm20=nn.BatchNorm2d(20)
        self.batchnorm1=nn.BatchNorm2d(1)



    def forward(self, x):
        ## 256x256
        x1=self.dense_block1(x)
        x1=self.trans_block1(x1)

        ###  32x32
        x2=(self.dense_block2(x1))
        x2=self.trans_block2(x2)

        # print x2.size()
        ### 16 X 16
        x3=(self.dense_block3(x2))
        x3=self.trans_block3(x3)

        ## Classifier  ##

        x4=(self.dense_block4(x3))
        x4=self.trans_block4(x4)

        x4=torch.cat([x4,x2],1)

        x5=(self.dense_block5(x4))
        x5=self.trans_block5(x5)

        # x5=x5+x1
        x5=torch.cat([x5,x1],1)

        x6=(self.dense_block6(x5))
        x6=(self.trans_block6(x6))

        shape_out = x6.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]
        x11 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)
        x21 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)
        x31 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)
        x41 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)
        x51 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)



        x6=torch.cat([x6,x51,x41,x31,x21,x11,x],1)

        return x6


class Dense_base_down2(nn.Module):
    def __init__(self):
        super(Dense_base_down2, self).__init__()

        self.dense_block1=BottleneckBlock2(3,13)
        self.trans_block1=TransitionBlock1(16,8)

        ############# Block2-down 32-32  ##############
        self.dense_block2=BottleneckBlock2(8,16)
        self.trans_block2=TransitionBlock1(24,16)

        ############# Block3-down  16-16 ##############
        self.dense_block3=BottleneckBlock2(16,16)
        self.trans_block3=TransitionBlock1(32,16)

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlock2(16,16)
        self.trans_block4=TransitionBlock(32,16)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlock2(32,8)
        self.trans_block5=TransitionBlock(40,8)

        self.dense_block6=BottleneckBlock2(16,8)
        self.trans_block6=TransitionBlock(24,4)


        self.conv_refin=nn.Conv2d(11,20,3,1,1)
        self.tanh=nn.Tanh()


        self.conv11 = nn.Conv2d(8, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv21 = nn.Conv2d(16, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv31 = nn.Conv2d(16, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv41 = nn.Conv2d(32, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv51 = nn.Conv2d(16, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv61 = nn.Conv2d(8, 1, kernel_size=3,stride=1,padding=1)  # 1mm

        self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)
        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)


        self.batchnorm20=nn.BatchNorm2d(20)
        self.batchnorm1=nn.BatchNorm2d(1)



    def forward(self, x):
        ## 256x256
        x1=self.dense_block1(x)
        x1=self.trans_block1(x1)

        ###  32x32
        x2=(self.dense_block2(x1))
        x2=self.trans_block2(x2)

        # print x2.size()
        ### 16 X 16
        x3=(self.dense_block3(x2))
        x3=self.trans_block3(x3)

        ## Classifier  ##

        x4=(self.dense_block4(x3))
        x4=self.trans_block4(x4)

        # x4=x4+x2
        x4=torch.cat([x4,x2],1)

        x5=(self.dense_block5(x4))
        x5=self.trans_block5(x5)

        # x5=x5+x1
        x5=torch.cat([x5,x1],1)

        x6=(self.dense_block6(x5))
        x6=(self.trans_block6(x6))

        shape_out = x6.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]


        x11 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)
        x21 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)
        x31 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)
        x41 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)
        x51 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)



        x6=torch.cat([x6,x51,x41,x31,x21,x11,x],1)

        return x6


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)



class BottleneckBlock1(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock1, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=5, stride=1,
                               padding=2, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)



class BottleneckBlock2(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock2, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=7, stride=1,
                               padding=3, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.upsample_nearest(out, scale_factor=2)



class TransitionBlock1(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock1, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)



class TransitionBlock3(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock3, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return out