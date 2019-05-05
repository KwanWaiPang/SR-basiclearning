import math
import torch
import torch.nn as nn
import torchvision
from . import block as B#if you only realize the FSRCNN, may be you don't need to use the B
from . import spectral_norm as SN#fsrcnn not need to use this file
import data.util as util##############use this for bicbulic
import torch.nn.functional as F#Almost all of the active functions
import pywt###############DMT
from pytorch_wavelets import DWTForward, DWTInverse

####################
# Generator
####################
#this .py is the network structure.may be you can call it model, the architecture of the model is defined in here


#######################################################################################################3
class Block_of_DMT1(nn.Module):
    def __init__(self):
        super(Block_of_DMT1,self).__init__()

        #DMT1
        self.conv1_1=nn.Conv2d(in_channels=160,out_channels=160,kernel_size=3,stride=1,padding=1)
        self.bn1_1=nn.BatchNorm2d(160, affine=True)
        self.relu1_1=nn.ReLU()

    def forward(self, x):
        output = self.relu1_1(self.bn1_1(self.conv1_1(x)))
        return output 

class Block_of_DMT2(nn.Module):
    def __init__(self):
        super(Block_of_DMT2,self).__init__()

        #DMT1
        self.conv2_1=nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.bn2_1=nn.BatchNorm2d(256, affine=True)
        self.relu2_1=nn.ReLU()

    def forward(self, x):
        output = self.relu2_1(self.bn2_1(self.conv2_1(x)))
        return output 

class Block_of_DMT3(nn.Module):
    def __init__(self):
        super(Block_of_DMT3,self).__init__()

        #DMT1
        self.conv3_1=nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.bn3_1=nn.BatchNorm2d(256, affine=True)
        self.relu3_1=nn.ReLU()

    def forward(self, x):
        output = self.relu3_1(self.bn3_1(self.conv3_1(x)))
        return output 


#MWCNN
class MWCNN(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, upscale=2, norm_type='batch', act_type='relu', \
            mode='NAC', res_scale=1, upsample_mode='upconv'):##play attention the upscales
        super(MWCNN,self).__init__()
        
        self.DWT= DWTForward(J=1, wave='haar').cuda()
        self.IDWT=DWTInverse(wave='haar').cuda()

        #DMT1 operation
        #DMT1
        self.conv_DMT1=nn.Conv2d(in_channels=3*4,out_channels=160,kernel_size=3,stride=1,padding=1)
        self.bn_DMT1=nn.BatchNorm2d(160, affine=True)
        self.relu_DMT1=nn.ReLU()
        #IDMT1
        self.conv_IDMT1=nn.Conv2d(in_channels=160,out_channels=3*4,kernel_size=3,stride=1,padding=1)


        self.blockDMT1=self.make_layer(Block_of_DMT1,3)

        #DMT2 operation
        #DMT2
        self.conv_DMT2=nn.Conv2d(in_channels=640,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.bn_DMT2=nn.BatchNorm2d(256, affine=True)
        self.relu_DMT2=nn.ReLU()
        #IDMT2
        self.conv_IDMT2=nn.Conv2d(in_channels=256,out_channels=640,kernel_size=3,stride=1,padding=1)
        self.bn_IDMT2=nn.BatchNorm2d(640, affine=True)
        self.relu_IDMT2=nn.ReLU()

        self.blockDMT2=self.make_layer(Block_of_DMT2,3)

        #DMT3 operation
        #DMT3
        self.conv_DMT3=nn.Conv2d(in_channels=1024,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.bn_DMT3=nn.BatchNorm2d(256, affine=True)
        self.relu_DMT3=nn.ReLU()
        #IDMT3
        self.conv_IDMT3=nn.Conv2d(in_channels=256,out_channels=1024,kernel_size=3,stride=1,padding=1)
        self.bn_IDMT3=nn.BatchNorm2d(1024, affine=True)
        self.relu_IDMT3=nn.ReLU()

        self.blockDMT3=self.make_layer(Block_of_DMT3,3)



    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def _transformer(self, DMT1_yl, DMT1_yh):
        list_tensor = []
        for i in range(3):
            list_tensor.append(DMT1_yh[0][:,:,i,:,:])
        list_tensor.append(DMT1_yl)
        return torch.cat(list_tensor, 1)


    def _Itransformer(self,out):
        #w = pywt.Wavelet('haar')
        yh = []
        C=out.shape[1]/4
        #sz=2*(len(w.dec_lo) // 2 - 1)
        #if yl.shape[-2] % 2 == 1 and yl.shape[-1] % 2 == 1:
            #yl = F.pad(yl, (sz, sz+1, sz, sz+1), mode='reflect')
        #elif yl.shape[-2] % 2 == 1:
            #yl = F.pad(yl, (sz, sz+1, sz, sz), mode='reflect')
        #elif yl.shape[-1] % 2 == 1:
            #yl = F.pad(yl, (sz, sz, sz, sz+1), mode='reflect')
        #else:
            #yl = F.pad(yl, (sz, sz, sz, sz), mode='reflect')


        y = out.reshape((out.shape[0], C, 4, out.shape[-2], out.shape[-1]))
        yl = y[:,:,0].contiguous()
        yh.append(y[:,:,1:].contiguous())

        return yl, yh

    def forward(self, x):#
         DMT1_p=x
         #DMT1
         DMT1_yl,DMT1_yh = self.DWT(x)
         #################################DMT1_yl,DMT1_yh=pywt.dwt2(x, 'haar')####################
         DMT1 = self._transformer(DMT1_yl, DMT1_yh)
         out=self.relu_DMT1(self.bn_DMT1(self.conv_DMT1(DMT1)))
         out=self.blockDMT1(out)###160
         
         DMT2_p=out
         #DMT2
         DMT2_yl, DMT2_yh=self.DWT(out)
         ###########################DMT2_yl, DMT2_yh=pywt.dwt2(out, 'haar')####################
         DMT2=self._transformer(DMT2_yl, DMT2_yh)
         out=self.relu_DMT2(self.bn_DMT2(self.conv_DMT2(DMT2)))
         out=self.blockDMT2(out)###256

         DMT3_p=out
         #DMT3
         DMT3_yl, DMT3_yh=self.DWT(out)
         #######################################DMT3_yl, DMT3_yh=pywt.dwt2(out, 'haar')####################
         DMT3=self._transformer(DMT3_yl, DMT3_yh)
         out=self.relu_DMT3(self.bn_DMT3(self.conv_DMT3(DMT3)))
         out=self.blockDMT3(out)###256

         #IDMT3
         out=self.blockDMT3(out)#DMT4
         out=self.relu_IDMT3(self.bn_IDMT3(self.conv_IDMT3(out)))
         out=self._Itransformer(out)###########
         IDMT3=self.IDWT(out)
         ############################33IDMT3=pywt.idwt2(out, 'haar')####################
         out=torch.add(IDMT3,DMT3_p)

         #IDMT2
         out=self.blockDMT2(out)
         out=self.relu_IDMT2(self.bn_IDMT2(self.conv_IDMT2(out)))
         out=self._Itransformer(out)##############
         IDMT2=self.IDWT(out)
         ######################################IDMT2=pywt.idwt2(out, 'haar')####################
         out=torch.add(IDMT2,DMT2_p)

         #IDMT1
         out=self.blockDMT1(out)
         out=self.conv_IDMT1(out)
         out=self._Itransformer(out)###############
         IDMT1=self.IDWT(out)
         #######################################IDMT1=pywt.idwt2(out, 'haar')####################
         out=torch.add(IDMT1,DMT1_p)

         return out
 
#######################################################################################################


#######################################################################################################3
#FSRCNN
class FSRCNN(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, upscale=2, norm_type='batch', act_type='relu', \
            mode='NAC', res_scale=1, upsample_mode='upconv'):##play attention the upscales
        super(FSRCNN,self).__init__()
        #Feature extractionn
        self.conv1=nn.Conv2d(in_channels=in_nc,out_channels=nf,kernel_size=5,stride=1,padding=2)#nf=56.add padding ,make the data alignment
        self.prelu1=nn.PReLU()
 
        #Shrinking
        self.conv2=nn.Conv2d(in_channels=nf,out_channels=12,kernel_size=1,stride=1,padding=0)
        self.prelu2 = nn.PReLU()
 
        # Non-linear Mapping
        self.conv3=nn.Conv2d(in_channels=12,out_channels=12,kernel_size=3,stride=1,padding=1)
        self.prelu3 = nn.PReLU()
        self.conv4=nn.Conv2d(in_channels=12,out_channels=12,kernel_size=3,stride=1,padding=1)
        self.prelu4 = nn.PReLU()
        self.conv5=nn.Conv2d(in_channels=12,out_channels=12,kernel_size=3,stride=1,padding=1)
        self.prelu5 = nn.PReLU()
        self.conv6=nn.Conv2d(in_channels=12,out_channels=12,kernel_size=3,stride=1,padding=1)
        self.prelu6 = nn.PReLU()
 
        # Expanding
        self.conv7=nn.Conv2d(in_channels=12,out_channels=nf,kernel_size=1,stride=1,padding=0)
        self.prelu7 = nn.PReLU()
 
        # Deconvolution
        self.last_part= nn.ConvTranspose2d(in_channels=nf,out_channels=in_nc,kernel_size=9,stride=upscale, padding=4, output_padding=1)
 
 
    def forward(self, x):#
         out = self.prelu1(self.conv1(x))
         out = self.prelu2(self.conv2(out))
         out = self.prelu3(self.conv3(out))
         out = self.prelu4(self.conv4(out))
         out = self.prelu5(self.conv5(out))
         out = self.prelu6(self.conv6(out))
         out = self.prelu7(self.conv7(out))
         out = self.last_part(out)
 
         return out
 
#######################################################################################################3
#define the block of the noise estimation network
class Block_of_NENET(nn.Module):
    def __init__(self):
        super(Block_of_NENET,self).__init__()

        self.conv1=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1=nn.BatchNorm2d(64, affine=True)
        self.relu=nn.ReLU()

    def forward(self, x):
        output = self.relu(self.bn1(self.conv1(x)))
        return output 

#noise estimation network
class NENET(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, upscale=2, norm_type='batch', act_type='relu', \
            mode='NAC', res_scale=1, upsample_mode='upconv'):##play attention the upscales
        super(NENET,self).__init__() 

        self.conv1=nn.Conv2d(in_channels=in_nc,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True)
        self.bn=nn.BatchNorm2d(64, affine=True)
        self.relu=nn.ReLU()
        self.block=self.make_layer(Block_of_NENET,15)
        self.last_part= nn.Conv2d(in_channels=64, out_channels=in_nc, kernel_size=3, stride=1, padding=1, bias=True)
 

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)
 

    def forward(self, x):#
         #lr_data = x
         out = self.conv1(x)
         out=self.bn(out)
         out=self.relu(out)
         out = self.block(out)
         out = self.last_part(out)
         #out = torch.add(out,lr_data)
 
         return out



#######################################################################################################3
#EPSCN
class ESPCN(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, upscale=2, norm_type='batch', act_type='relu', \
            mode='NAC', res_scale=1, upsample_mode='upconv'):##play attention the upscales
        super(ESPCN,self).__init__()
        self.conv1=nn.Conv2d(in_channels=in_nc,out_channels=64,kernel_size=5,stride=1,padding=2)
        self.conv2=nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.conv3=nn.Conv2d(in_channels=32,out_channels=in_nc*(upscale ** 2),kernel_size=3,stride=1,padding=1)
        self.pixel_shuffle=nn.PixelShuffle(upscale)

 
 
    def forward(self, x):
        out=F.tanh(self.conv1(x))
        out=F.tanh(self.conv2(out))
        out=F.sigmoid(self.pixel_shuffle(self.conv3(out)))
 
        return out

##########################################################################################################
#SRResNet, the Original version
#define the residual block
class DN_Residual_Block(nn.Module):
    def __init__(self):
        super(DN_Residual_Block,self).__init__()

        self.conv1=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1=nn.BatchNorm2d(64, affine=True)
        self.prelu=nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2=nn.BatchNorm2d(64, affine=True)

    def forward(self, x):
        identity_data1 = x
        output = self.prelu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        output = torch.add(output,identity_data1)
        return output 

##############################################
class DN_ResNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, upscale=2, norm_type='batch', act_type='relu', \
            mode='NAC', res_scale=1, upsample_mode='upconv'):##play attention the upscales
        super(DN_ResNet,self).__init__()

        self.conv_input=nn.Conv2d(in_channels=in_nc,out_channels=nf,kernel_size=9,stride=1,padding=4,bias=False)
        self.prelu=nn.PReLU()

        self.residual=self.make_layer(O_Residual_Block,16)

        self.conv_mid = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid=nn.BatchNorm2d(64, affine=True)


        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )

        self.conv_output = nn.Conv2d(in_channels=nf, out_channels=out_nc, kernel_size=9, stride=1, padding=4, bias=False)


    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)
 
    def forward(self,x):
        out = self.prelu(self.conv_input(x))
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out,residual)
        #out = self.upscale4x(out)
        out = self.conv_output(out)
        return out

 
 ##########################################################################################################################################################


 
##########################################################################################################
#SRResNet, the Original version
#define the residual block
class O_Residual_Block(nn.Module):
    def __init__(self):
        super(O_Residual_Block,self).__init__()

        self.conv1=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1=nn.BatchNorm2d(64, affine=True)
        self.prelu=nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2=nn.BatchNorm2d(64, affine=True)

    def forward(self, x):
        identity_data1 = x
        output = self.prelu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        output = torch.add(output,identity_data1)
        return output 

##############################################
class OSRRESNET(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, upscale=2, norm_type='batch', act_type='relu', \
            mode='NAC', res_scale=1, upsample_mode='upconv'):##play attention the upscales
        super(OSRRESNET,self).__init__()

        self.conv_input=nn.Conv2d(in_channels=in_nc,out_channels=nf,kernel_size=9,stride=1,padding=4,bias=False)
        self.prelu=nn.PReLU()

        self.residual=self.make_layer(O_Residual_Block,16)

        self.conv_mid = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid=nn.BatchNorm2d(64, affine=True)


        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )

        self.conv_output = nn.Conv2d(in_channels=nf, out_channels=out_nc, kernel_size=9, stride=1, padding=4, bias=False)


    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)
 
    def forward(self,x):
        out = self.prelu(self.conv_input(x))
        residual = out
        out = self.residual(out)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out,residual)
        out = self.upscale4x(out)
        out = self.conv_output(out)
        return out

 
 ##########################################################################################################################################################


#####################SRResNet########################################################
class SRResNet(nn.Module):#read my CSDN for the nn.Module
#nn.Module is contain the forward and each layyer
    def __init__(self, in_nc, out_nc, nf, nb, upscale=4, norm_type='batch', act_type='relu', \
            mode='NAC', res_scale=1, upsample_mode='upconv'):#the .jason file decide the mode is "CNA"
        #input channels\output channels\the number of filters in the first layer\thw number of resduial block\upscale\   \relu\Conv -> Norm -> Act\
        super(SRResNet, self).__init__()#for the super(),read my CSDN
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)#read the block.py.
        resnet_blocks = [B.ResNetBlock(nf, nf, nf, norm_type=norm_type, act_type=act_type,\
            mode=mode, res_scale=res_scale) for _ in range(nb)]#'nb' is the number of block, and there is 23 in the .jason
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)#here use the BN

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok##Deconvolution
        elif upsample_mode == 'pixelshuffle':##there are 'pixelshuffle' in the network.py
            upsample_block = B.pixelshuffle_block##the espcn
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*resnet_blocks, LR_conv)),\
            *upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
        x = self.model(x)
        return x



############################################################################################################################
##################this is ESRGAN
class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, upscale=4, norm_type=None, \
            act_type='leakyrelu', mode='CNA', upsample_mode='upconv'):
        super(RRDBNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        rb_blocks = [B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb)]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)),\
            *upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
        x = self.model(x)
        return x


####################
# Discriminator
####################


# VGG style Discriminator with input size 128*128
class Discriminator_VGG_128(nn.Module):
    def __init__(self, in_nc, base_nf, norm_type='batch', act_type='leakyrelu', mode='CNA'):
        super(Discriminator_VGG_128, self).__init__()
        # features
        # hxw, c
        # 128, 64
        conv0 = B.conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, \
            mode=mode)
        conv1 = B.conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 64, 64
        conv2 = B.conv_block(base_nf, base_nf*2, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv3 = B.conv_block(base_nf*2, base_nf*2, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 32, 128
        conv4 = B.conv_block(base_nf*2, base_nf*4, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv5 = B.conv_block(base_nf*4, base_nf*4, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 16, 256
        conv6 = B.conv_block(base_nf*4, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv7 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 8, 512
        conv8 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv9 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 4, 512
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,\
            conv9)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# VGG style Discriminator with input size 128*128, Spectral Normalization
class Discriminator_VGG_128_SN(nn.Module):
    def __init__(self):
        super(Discriminator_VGG_128_SN, self).__init__()
        # features
        # hxw, c
        # 128, 64
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.conv0 = SN.spectral_norm(nn.Conv2d(3, 64, 3, 1, 1))
        self.conv1 = SN.spectral_norm(nn.Conv2d(64, 64, 4, 2, 1))
        # 64, 64
        self.conv2 = SN.spectral_norm(nn.Conv2d(64, 128, 3, 1, 1))
        self.conv3 = SN.spectral_norm(nn.Conv2d(128, 128, 4, 2, 1))
        # 32, 128
        self.conv4 = SN.spectral_norm(nn.Conv2d(128, 256, 3, 1, 1))
        self.conv5 = SN.spectral_norm(nn.Conv2d(256, 256, 4, 2, 1))
        # 16, 256
        self.conv6 = SN.spectral_norm(nn.Conv2d(256, 512, 3, 1, 1))
        self.conv7 = SN.spectral_norm(nn.Conv2d(512, 512, 4, 2, 1))
        # 8, 512
        self.conv8 = SN.spectral_norm(nn.Conv2d(512, 512, 3, 1, 1))
        self.conv9 = SN.spectral_norm(nn.Conv2d(512, 512, 4, 2, 1))
        # 4, 512

        # classifier
        self.linear0 = SN.spectral_norm(nn.Linear(512 * 4 * 4, 100))
        self.linear1 = SN.spectral_norm(nn.Linear(100, 1))

    def forward(self, x):
        x = self.lrelu(self.conv0(x))
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv4(x))
        x = self.lrelu(self.conv5(x))
        x = self.lrelu(self.conv6(x))
        x = self.lrelu(self.conv7(x))
        x = self.lrelu(self.conv8(x))
        x = self.lrelu(self.conv9(x))
        x = x.view(x.size(0), -1)
        x = self.lrelu(self.linear0(x))
        x = self.linear1(x)
        return x


class Discriminator_VGG_96(nn.Module):
    def __init__(self, in_nc, base_nf, norm_type='batch', act_type='leakyrelu', mode='CNA'):
        super(Discriminator_VGG_96, self).__init__()
        # features
        # hxw, c
        # 96, 64
        conv0 = B.conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, \
            mode=mode)
        conv1 = B.conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 48, 64
        conv2 = B.conv_block(base_nf, base_nf*2, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv3 = B.conv_block(base_nf*2, base_nf*2, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 24, 128
        conv4 = B.conv_block(base_nf*2, base_nf*4, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv5 = B.conv_block(base_nf*4, base_nf*4, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 12, 256
        conv6 = B.conv_block(base_nf*4, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv7 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 6, 512
        conv8 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv9 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 3, 512
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,\
            conv9)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Discriminator_VGG_192(nn.Module):
    def __init__(self, in_nc, base_nf, norm_type='batch', act_type='leakyrelu', mode='CNA'):
        super(Discriminator_VGG_192, self).__init__()
        # features
        # hxw, c
        # 192, 64
        conv0 = B.conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, \
            mode=mode)
        conv1 = B.conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 96, 64
        conv2 = B.conv_block(base_nf, base_nf*2, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv3 = B.conv_block(base_nf*2, base_nf*2, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 48, 128
        conv4 = B.conv_block(base_nf*2, base_nf*4, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv5 = B.conv_block(base_nf*4, base_nf*4, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 24, 256
        conv6 = B.conv_block(base_nf*4, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv7 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 12, 512
        conv8 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv9 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 6, 512
        conv10 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv11 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 3, 512
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,\
            conv9, conv10, conv11)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


####################
# Perceptual Network
####################


# Assume input range is [0, 1]
class VGGFeatureExtractor(nn.Module):
    def __init__(self,
                 feature_layer=34,
                 use_bn=False,
                 use_input_norm=True,
                 device=torch.device('cpu')):
        super(VGGFeatureExtractor, self).__init__()
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


# Assume input range is [0, 1]
class ResNet101FeatureExtractor(nn.Module):
    def __init__(self, use_input_norm=True, device=torch.device('cpu')):
        super(ResNet101FeatureExtractor, self).__init__()
        model = torchvision.models.resnet101(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.children())[:8])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


class MINCNet(nn.Module):
    def __init__(self):
        super(MINCNet, self).__init__()
        self.ReLU = nn.ReLU(True)
        self.conv11 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv12 = nn.Conv2d(64, 64, 3, 1, 1)
        self.maxpool1 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv21 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv22 = nn.Conv2d(128, 128, 3, 1, 1)
        self.maxpool2 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv31 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv32 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv33 = nn.Conv2d(256, 256, 3, 1, 1)
        self.maxpool3 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv41 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv42 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv43 = nn.Conv2d(512, 512, 3, 1, 1)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv51 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv52 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv53 = nn.Conv2d(512, 512, 3, 1, 1)

    def forward(self, x):
        out = self.ReLU(self.conv11(x))
        out = self.ReLU(self.conv12(out))
        out = self.maxpool1(out)
        out = self.ReLU(self.conv21(out))
        out = self.ReLU(self.conv22(out))
        out = self.maxpool2(out)
        out = self.ReLU(self.conv31(out))
        out = self.ReLU(self.conv32(out))
        out = self.ReLU(self.conv33(out))
        out = self.maxpool3(out)
        out = self.ReLU(self.conv41(out))
        out = self.ReLU(self.conv42(out))
        out = self.ReLU(self.conv43(out))
        out = self.maxpool4(out)
        out = self.ReLU(self.conv51(out))
        out = self.ReLU(self.conv52(out))
        out = self.conv53(out)
        return out


# Assume input range is [0, 1]
class MINCFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=34, use_bn=False, use_input_norm=True, \
                device=torch.device('cpu')):
        super(MINCFeatureExtractor, self).__init__()

        self.features = MINCNet()
        self.features.load_state_dict(
            torch.load('../experiments/pretrained_models/VGG16minc_53.pth'), strict=True)
        self.features.eval()
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        output = self.features(x)
        return output
