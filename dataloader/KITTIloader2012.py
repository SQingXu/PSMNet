import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','exr'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):

  left_fold  = 'LeftRGB/'
  right_fold = 'RightRGB/'
  disp_noc   = 'LeftDisp/'

  image = [img for img in os.listdir(filepath+left_fold) ]
  
  train = image[:]
  val   = image[:]

  left_train  = [filepath+left_fold+img for img in train]
  right_train = [filepath+right_fold+img for img in train]

  #disp_image = [img for img in os.listdir(filepath + disp_noc)]
  disp_train = [filepath+disp_noc+img[:-4]+'_10.exr' for img in train]


  left_val  = [filepath+left_fold+img for img in val]
  right_val = [filepath+right_fold+img for img in val]
  disp_val = [filepath+disp_noc+img[:-4]+'_10.exr' for img in image]

  return left_train, right_train, disp_train, left_val, right_val, disp_val
