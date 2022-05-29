from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = None
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available, relabel_image_stardist
from stardist.matching import matching, matching_dataset
from stardist.models import Config2D, StarDist2D, StarDistData2D

from itertools import compress
import os

def parse_image(file_name):
  axis_norm = (0,1)   # normalize channels independently
  image = imread(file_name)
  image = np.sum(image, axis=0)
  image = normalize(image,1,99.8,axis=axis_norm)
  return image 

def parse_mask(file_name):
  mask = imread(file_name)
  mask = mask[4, :, :]
  mask = fill_label_holes(mask)
  return mask

np.random.seed(42)
lbl_cmap = random_label_cmap()

mask_files = glob('/home/lauren/Documents/research/light-microscopy/raw_images/10-25-21/masks_labkit/ts_42deg_0min_decon_image*.tif')

image_files_included = []
mask_files_included = []
for mask_file in mask_files:
  experiment_path = mask_file[:mask_file.find('masks_labkit')]
  base_name = Path(mask_file).name
  image_file = os.path.join(experiment_path, 'images', base_name)
  if os.path.exists(image_file):
    image_files_included.append(image_file)
    mask_files_included.append(mask_file)

X = sorted(image_files_included)
Y = sorted(mask_files_included)
assert all(Path(x).name==Path(y).name for x,y in zip(X,Y))

rng = np.random.RandomState(42)
ind = rng.permutation(len(image_files_included))
n_val = max(1, int(round(0.15 * len(image_files_included))))
ind_train, ind_val = ind[:-n_val], ind[-n_val:]
X_val, Y_val = [X[i] for i in ind_val]  , [Y[i] for i in ind_val]
X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train] 

print('number of images: %3d' % len(X))
print('- training:       %3d' % len(X_trn))
print('- validation:     %3d' % len(X_val))

X_val = list(map(parse_image, X_val))
Y_val = list(map(parse_mask, Y_val))

X_trn = list(map(parse_image, X_trn))
Y_trn = list(map(parse_mask, Y_trn))

def plot_img_label(img, lbl, img_title="image", lbl_title="label", **kwargs):
  fig, (ai,al) = plt.subplots(1,2, figsize=(12,5), gridspec_kw=dict(width_ratios=(1.25,1)))
  im = ai.imshow(img, cmap='gray', clim=(0,1))
  ai.set_title(img_title)    
  fig.colorbar(im, ax=ai)
  al.imshow(lbl, cmap=lbl_cmap)
  al.set_title(lbl_title)
  plt.tight_layout()


for img, lbl in zip(X_trn, Y_trn):
  assert img.ndim in (2,3)
  img = img if (img.ndim==2 or img.shape[-1]==3) else img[...,0]
  plot_img_label(img,lbl)
  plt.show()
  break

# 32 is a good default choice (see 1_data.ipynb)
n_rays = 128

# Use OpenCL-based computations for data generator during training (requires 'gputools')
use_gpu = False and gputools_available()

# Predict on subsampled grid for increased efficiency and larger field of view
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
#print(conf)
#vars(conf)

if use_gpu:
    from csbdeep.utils.tf import limit_gpu_memory
    # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
    limit_gpu_memory(0.8)
    # alternatively, try this:
    # limit_gpu_memory(None, allow_growth=True)

path = '/home/lauren/Documents/research/light-microscopy/auto_segmentation/StarDist2D/models'
#model = StarDist2D(conf, name='sted_13', basedir=path)
#model = StarDist2D.from_pretrained('sted_13')

model_name = 'sted_13'
model = StarDist2D(None, name=model_name, basedir=path)


median_size = calculate_extents(list(Y_trn), np.median)
fov = np.array(model._axes_tile_overlap('YX'))
print(f"median object size:      {median_size}")
print(f"network field of view :  {fov}")
if any(median_size > fov):
  print("WARNING: median object size larger than field of view of the neural network.")

def random_fliprot(img, mask): 
  assert img.ndim >= mask.ndim
  axes = tuple(range(mask.ndim))
  perm = tuple(np.random.permutation(axes))
  img = img.transpose(perm + tuple(range(mask.ndim, img.ndim))) 
  mask = mask.transpose(perm) 
  for ax in axes: 
      if np.random.rand() > 0.5:
          img = np.flip(img, axis=ax)
          mask = np.flip(mask, axis=ax)
  return img, mask 

def random_intensity_change(img):
  img = img*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
  return img


def augmenter(x, y):
  """Augmentation of a single input/label image pair.
  x is an input image
  y is the corresponding ground-truth label image
  """
  x, y = random_fliprot(x, y)
  x = random_intensity_change(x)
  # add some gaussian noise
  sig = 0.02*np.random.uniform(0,1)
  x = x + sig*np.random.normal(0,1,x.shape)
  return x, y

for img, lbl in zip(X_trn, Y_trn):
  assert img.ndim in (2,3)
  img_aug, lbl_aug = augmenter(img,lbl)
  plot_img_label(img_aug, lbl_aug, img_title="image augmented", lbl_title="label augmented")
  plt.show()
  break

#model.train(X_trn, Y_trn, validation_data=(X_val,Y_val), augmenter=augmenter)

model.optimize_thresholds(X_val, Y_val)

Y_val_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0]
              for x in tqdm(X_val)]

image_number = 4
plot_img_label(X_val[image_number],Y_val[image_number], lbl_title="label GT")
plt.show()
plot_img_label(X_val[image_number],Y_val_pred[image_number], lbl_title="label Pred")
plt.show()