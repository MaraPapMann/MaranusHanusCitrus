from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from numpy import ndarray
from torchvision import transforms as T
from skimage.util import img_as_ubyte
from torch import Tensor
import torch
from typing import List, Any


class LowPassFilter(object):
    def __init__(self) -> None:
        return
    

    def lowpass_arr_img(self, arr_img:ndarray, diameter:int) -> ndarray:
        # Assert data type
        assert isinstance(arr_img, ndarray) and isinstance(diameter, int)

        freq = np.fft.fft2(arr_img, axes=(0, 1))
        freq_shift = np.fft.fftshift(freq)
        mask = self.circular_mask(arr_img, diameter)
        low_freq = freq_shift * mask
        arr_img_low = np.abs(np.fft.ifft2(np.fft.ifftshift(low_freq), axes=(0, 1)))

        return arr_img_low
    

    def lowpass_ts4d(self, ts:Tensor, diameter:int) -> Tensor:
        ts_low = torch.zeros_like(ts)
        num_img = ts.shape[0]
        tf = T.Compose([T.ToTensor()])

        for i in range(num_img):
            cur_ts_img = ts[i]
            cur_ts_img = cur_ts_img.detach().cpu().numpy()
            cur_ts_img = np.transpose(cur_ts_img, (1,2,0))
            cur_ts_img = img_as_ubyte(cur_ts_img)
            cur_ts_img = self.lowpass_arr_img(cur_ts_img, diameter)
            cur_ts_img = np.clip(cur_ts_img, 0, 255)
            cur_ts_img = cur_ts_img.astype('uint8')
            cur_ts_img = tf(cur_ts_img)
        
            ts_low[i] = cur_ts_img

        return ts_low


    def circular_mask(self, arr_img:ndarray, diameter:int) -> ndarray:
        h, w, _ = arr_img.shape
        Y, X = np.ogrid[:h, :w]
        center = (int(w / 2), int(h / 2))
        dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        mask = dist_from_center <= diameter
        return np.dstack([mask, mask, mask])
    

    def arr_2_png(self, arr_img:ndarray, pth_out:str) -> None:
        arr_img = np.clip(arr_img, 0, 255)
        arr_img = arr_img.astype('uint8')
        plt.imsave(pth_out, arr_img)
    

    def ts4d_2_lst_arr_uint8(self, ts:Tensor) -> List[ndarray]:
        lst_arr_uint8 = []

        for i in range(ts.shape[0]):
            cur_ts = ts[i]
            cur_ts = cur_ts.detach().cpu().numpy()
            cur_ts = np.transpose(cur_ts, (1, 2, 0))
            cur_ts = img_as_ubyte(cur_ts)

            lst_arr_uint8.append(cur_ts)
        
        return lst_arr_uint8
        

"""
11111111111111111
111   Debug   111
11111111111111111
"""
if __name__ == '__main__':
    low_pass_filter = LowPassFilter()
    d = 50

    tf = T.Compose([T.ToTensor()])

    fp = 'secret/lena/lena.png'
    arr_img = np.array(Image.open(fp))
    arr_img = tf(arr_img).unsqueeze(0)
    ts4d_low = low_pass_filter.lowpass_ts4d(arr_img, d)
    lst_arr_img_uint8 = low_pass_filter.ts4d_2_lst_arr_uint8(ts4d_low)
    low_pass_filter.arr_2_png(lst_arr_img_uint8[0], 'test.png')