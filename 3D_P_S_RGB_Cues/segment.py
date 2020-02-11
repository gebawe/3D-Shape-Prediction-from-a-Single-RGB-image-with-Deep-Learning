
import cv2
from pylab import *

# rcParams['figure.figsize'] = 15, 15

import torch
from torch import nn
from torch import sigmoid
from unet_models import unet11
from pathlib import Path
from torch.nn import functional as F
from torchvision.transforms import ToTensor, Normalize, Compose

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_model():
    model = unet11(pretrained='carvana')
    model.eval()
    return model.to(device)

def mask_overlay(image, mask, color=(0, 255, 0)):
    """
    Helper function to visualize mask on the top of the car
    """
    mask = np.dstack((mask, mask, mask)) * np.array(color)
    mask = mask.astype(np.uint8)
    weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.)
    img = image.copy()
    ind = mask[:, :, 1] > 0    
    img[ind] = weighted_sum[ind]    
    return img

def load_image(img, pad=True):
    """
    Load image from a given path and pad it on the sides, so that eash side is divisible by 32 (newtwork requirement)
    
    if pad = True:
        returns image as numpy.array, tuple with padding in pixels as(x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    else:
        returns image as numpy.array
    """
    # img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if not pad:
        return img
    
    height, width, _ = img.shape
    
    if height % 32 == 0:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = 32 - height % 32
        y_min_pad = int(y_pad / 2)
        y_max_pad = y_pad - y_min_pad
        
    if width % 32 == 0:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = 32 - width % 32
        x_min_pad = int(x_pad / 2)
        x_max_pad = x_pad - x_min_pad
    
    img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_REFLECT_101)

    return img, (x_min_pad, y_min_pad, x_max_pad, y_max_pad)

img_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def crop_image(img, pads):
    """
    img: numpy array of the shape (height, width)
    pads: (x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    
    @return padded image
    """
    (x_min_pad, y_min_pad, x_max_pad, y_max_pad) = pads
    height, width = img.shape[:2] 
    
    return img[y_min_pad:height - y_max_pad, x_min_pad:width - x_max_pad]
    


def segment(img):

    # img = cv2.imread('chair.jpeg')
    print(img.ndim)
    if img.ndim == 4:
        dim = img.shape[0]
        model = get_model()
        mask_list = np.ndarray(shape=(24,64,64))

        for i in range(0,dim):
            img, pads = load_image(img[i], pad=True)

            with torch.no_grad():
                input_img = torch.unsqueeze(img_transform(img).to(device), dim=0)

            with torch.no_grad():
                mask = torch.sigmoid(model(input_img))
            # imshow(img)
            mask_array = mask.data[0].cpu().numpy()[0]
            print(mask_array.shape)
            mask_array = crop_image(mask_array, pads)
            print(mask_array.shape)
            mask_list[i] = mask_array
            # print(mask_list.shape)

        return (mask_list)
    else:
        img, pads = load_image(img, pad=True)
        model = get_model()

        with torch.no_grad():
            input_img = torch.unsqueeze(img_transform(img).to(device), dim=0)

        with torch.no_grad():
            mask = torch.sigmoid(model(input_img))

        # imshow(img)
        mask_array = mask.data[0].cpu().numpy()[0]
        mask_array = crop_image(mask_array, pads)
        return mask_array
# cv2.imshow("Mask", mask_array)
# cv2.imshow("Overlay", mask_overlay(crop_image(img, pads), (mask_array > 0.5).astype(np.uint8)))
# cv2.waitKey(0)

