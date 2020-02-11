"""Build Point Cloud Generator Pytorch model"""
import torch
from torch import nn
from unet_models import unet11, UNet11
from torch.nn import functional as F
from torchvision import models
import torchvision
from matplotlib import pyplot as plt
import cv2 

def conv2d_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, stride=2, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
    )

# def deconv2d_block(in_c, out_c):
#     return nn.Sequential(
#         nn.ConvTranspose2d(in_c, out_c, 3, stride=2,
#                            padding=1, output_padding=1, bias=True),
#         nn.BatchNorm2d(out_c),
#         nn.ReLU(),
#     )

def get_unet_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    UNetModel = unet11(pretrained='carvana') #'carvana'
    UNetModel.eval()
    return UNetModel.to(device)
unet11_model = get_unet_model()


def deconv2d_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, stride=1, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
    )

def linear_block(in_c, out_c):
    return nn.Sequential(
        nn.Linear(in_c, out_c),
        nn.BatchNorm1d(out_c),
        nn.ReLU(),
    )

def pixel_bias(outViewN, outW, outH, renderDepth):
    X, Y = torch.meshgrid([torch.arange(outH), torch.arange(outW)])
    X, Y = X.float(), Y.float() # [H,W]
    initTile = torch.cat([
        X.repeat([outViewN, 1, 1]), # [V,H,W]
        Y.repeat([outViewN, 1, 1]), # [V,H,W]
        torch.ones([outViewN, outH, outW]).float() * renderDepth, 
        torch.zeros([outViewN, outH, outW]).float(),
    ], dim=0) # [4V,H,W]

    return initTile.unsqueeze_(dim=0) # [1,4V,H,W]


class Encoder(nn.Module):
    """Encoder of Structure Generator"""
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = conv2d_block(4, 96)
        self.conv1_gray = conv2d_block(1, 96)
        self.conv2 = conv2d_block(96, 128)
        self.conv3 = conv2d_block(128, 192)
        self.conv4 = conv2d_block(192, 256)
        self.fc1 = linear_block(4096, 2048) # After flatten
        self.fc2 = linear_block(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)

    def forward(self, x, gray=False):
        if gray:
            x = self.conv1_gray(x)
        else:
            x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc1(x.view(-1, 4096))
        x = self.fc2(x)
        x = self.fc3(x)

        return x


class Decoder(nn.Module):
    """Build Decoder"""
    def __init__(self, outViewN, outW, outH, renderDepth):
        super(Decoder, self).__init__()
        self.outViewN = outViewN

        self.relu = nn.ReLU()
        self.fc1 = linear_block(512, 1024)
        self.fc2 = linear_block(1024, 2048)
        self.fc3 = linear_block(2048, 4096)
        self.deconv1 = deconv2d_block(256, 192)
        self.deconv2 = deconv2d_block(192, 128)
        self.deconv3 = deconv2d_block(128, 96)
        self.deconv4 = deconv2d_block(96, 64)
        self.deconv5 = deconv2d_block(64, 48)
        self.pixel_conv = nn.Conv2d(48, outViewN*4, 1, stride=1, bias=False)
        self.pixel_bias = pixel_bias(outViewN, outW, outH, renderDepth)

    def forward(self, x):
        x = self.relu(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = x.view([-1, 256, 4, 4])
        x = self.deconv1(F.interpolate(x, scale_factor=2))
        x = self.deconv2(F.interpolate(x, scale_factor=2))
        x = self.deconv3(F.interpolate(x, scale_factor=2))
        x = self.deconv4(F.interpolate(x, scale_factor=2))
        x = self.deconv5(F.interpolate(x, scale_factor=2))
        x = self.pixel_conv(x) + self.pixel_bias.to(x.device)
        XYZ, maskLogit = torch.split(
            x, [self.outViewN * 3, self.outViewN], dim=1)

        return XYZ, maskLogit

class Structure_Generator(nn.Module):
    """Structure generator components in PCG"""

    def __init__(self, encoder=None, decoder=None,
                 outViewN=8, outW=128, outH=128, renderDepth=1.0):
        super(Structure_Generator, self).__init__()

        if encoder: self.encoder = encoder
        else: self.encoder = Encoder()
        ## Uncomment this to use resnet
        # self.encoder = torchvision.models.resnext50_32x4d(pretrained=True)
        if decoder: self.decoder = decoder
        else: self.decoder = Decoder(outViewN, outW, outH, renderDepth)
        # self.unet11 = UNet11(pretrained=True)
    def forward(self, x):
        # print(x.shape)
        with torch.no_grad():
            silhouette = torch.sigmoid(unet11_model(x))
        # plt.imshow(  x.permute(1, 2, 0)  )
        # mask = silhouette.data[0].cpu().numpy()[0]
        # cv2.imshow("Mask", mask)
        # cv2.waitKey(0)
        with torch.no_grad():
            x_cat = torch.cat((x, silhouette), 1)
        # print(x_cat.shape)
        # latent_silhouette = self.encoder(silhouette)
        latent = self.encoder(x_cat)
        # with torch.no_grad():
        #     latent_cat = torch.cat((latent_silhouette, latent), 1)
        XYZ, maskLogit = self.decoder(latent)

        return XYZ, maskLogit


# TESTING
if __name__ == '__main__':
    import options
    cfg = options.get_arguments()
    encoder = Encoder()
    decoder = Decoder(cfg.outViewN, cfg.outW, cfg.outH, cfg.renderDepth)
    model = Structure_Generator()
