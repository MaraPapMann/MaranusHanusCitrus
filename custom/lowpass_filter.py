from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from numpy import ndarray


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
        

"""
11111111111111111
111   Debug   111
11111111111111111
"""
if __name__ == '__main__':
    low_pass_filter = LowPassFilter()
    d = 40

    fp = 'secret/lena/lena.png'
    arr_img = np.array(Image.open(fp))
    arr_img_low = low_pass_filter.lowpass_arr_img(arr_img, d)
    low_pass_filter.arr_2_png(arr_img_low, 'test.png')