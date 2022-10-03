import torch
from matplotlib import pyplot as plt
from torchvision import transforms as T
from PIL import Image
from torch import Tensor
import numpy as np
from skimage.util import img_as_ubyte
from numpy import ndarray


def tensor_2_array(x:Tensor) -> ndarray:
    """
    @Desc: Transform the tensor to an array.
    @Args:
        x: the tensor of size (n, c, h, w);
    @Return:
        x: the converted array;
    """
    n, c, h, w = x.shape
    x = x.cpu()
    x = x.permute(0, 2, 3, 1)
    x = x.reshape(n*h, w, c)
    x = x.numpy()
    x = img_as_ubyte(x)

    return x


def array_2_img(x:ndarray, pth_out:str) -> None:
    """
    @Desc: Save the array as an png image.
    @Args:
        x: the converted array;
        pth_out: the path to the output file;
    @Return:
    """
    x = Image.fromarray(x)
    x.save(pth_out)

    return


"""
Debug
"""
if __name__ == '__main__':
    pth_img = 'data/example.jpg'
    pth_out = 'test.png'
    img = Image.open(pth_img)
    z = np.array(img)
    transform = T.ToTensor()
    x = transform(img)
    x = x.unsqueeze(0)

    x = tensor_2_array(x)
    array_2_img(x, pth_out)
