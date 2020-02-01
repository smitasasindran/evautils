import numpy as np
from PIL import Image


"""
# ToDo: Mention source
"""
class Cutout(object):
    def __init__(self, length=10):
        self.length = length

    def __call__(self, p, img):
      cutout_im = np.zeros_like(img)
      for i in range(img.shape[0]):
        p_1 = np.random.rand() 
        im = img[i]
        im = np.array(im)
        if p_1 > p:
          cutout_im[i] = im
        else:
          mask_val = im.mean()

          top = np.random.randint(0, im.shape[0])
          left = np.random.randint(0, im.shape[1])
          bottom = top + self.length
          right = left + self.length

          im[top:bottom, left:right, :] = mask_val

          im = Image.fromarray(im)
          cutout_im[i] = im
      return cutout_im

"""
Code from https://github.com/yu4u/cutout-random-erasing/blob/master/random_eraser.py
"""
def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser


