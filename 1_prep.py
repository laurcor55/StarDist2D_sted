import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist.matching import matching
from stardist.models import Config2D, StarDist2D
import sted_analysis as sted


np.random.seed(42)
lbl_cmap = random_label_cmap()

mask_files = glob('/home/lauren/Documents/research/light-microscopy/raw_images/10-25-21/masks_labkit/ts_42deg_0min_decon_image*.tif')

X, Y = sted.get_file_pairs(mask_files)
X_val, Y_val, X_trn, Y_trn = sted.shuffle_split_data(X, Y)

print('number of images: %3d' % len(X))
print('- training:       %3d' % len(X_trn))
print('- validation:     %3d' % len(X_val))

X_val = list(map(sted.parse_image, X_val))
Y_val = list(map(sted.parse_mask, Y_val))

X_trn = list(map(sted.parse_image, X_trn))
Y_trn = list(map(sted.parse_mask, Y_trn))




for img, lbl in zip(X_trn, Y_trn):
  assert img.ndim in (2,3)
  img = img if (img.ndim==2 or img.shape[-1]==3) else img[...,0]
  sted.plot_img_label(img,lbl, lbl_cmap)
  plt.show()
  break

n_rays = 128
use_gpu = False and gputools_available()
grid = (8, 8)
patch_size = 512
n_channel = 1
conf = Config2D (
  n_rays       = n_rays,
  grid         = grid,
  use_gpu      = use_gpu,
  n_channel_in = n_channel,
  train_patch_size = (patch_size, patch_size),
  backbone='unet'
)

if use_gpu:
  from csbdeep.utils.tf import limit_gpu_memory
  limit_gpu_memory(0.8)


path = '/home/lauren/Documents/research/light-microscopy/auto_segmentation/StarDist2D/models'
model_name = 'sted_13'
model = StarDist2D(None, name=model_name, basedir=path)

median_size = calculate_extents(Y_trn, np.median)
fov = np.array(model._axes_tile_overlap('YX'))
print(f"median object size:      {median_size}")
print(f"network field of view :  {fov}")
if any(median_size > fov):
  print("WARNING: median object size larger than field of view of the neural network.")

for img, lbl in zip(X_trn, Y_trn):
  assert img.ndim in (2,3)
  img_aug, lbl_aug = sted.augmenter(img,lbl)
  sted.plot_img_label(img_aug, lbl_aug, lbl_cmap, img_title="image augmented", lbl_title="label augmented")
  plt.show()
  break

