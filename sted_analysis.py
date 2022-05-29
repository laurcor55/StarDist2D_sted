import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = None
import matplotlib.pyplot as plt
import os

from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist.matching import matching
from stardist.models import Config2D, StarDist2D

def get_file_pairs(mask_files):
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
  return X, Y


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

def shuffle_split_data(X, Y):
  rng = np.random.RandomState(42)
  ind = rng.permutation(len(X))
  n_val = max(1, int(round(0.15 * len(X))))
  ind_train, ind_val = ind[:-n_val], ind[-n_val:]
  X_val, Y_val = [X[i] for i in ind_val]  , [Y[i] for i in ind_val]
  X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train]
  return X_val, Y_val, X_trn, Y_trn


def plot_img_label(img, lbl, lbl_cmap, img_title="image", lbl_title="label", **kwargs):
  fig, (ai,al) = plt.subplots(1,2, figsize=(12,5), gridspec_kw=dict(width_ratios=(1.25,1)))
  im = ai.imshow(img, cmap='gray', clim=(0,1))
  ai.set_title(img_title)    
  fig.colorbar(im, ax=ai)
  al.imshow(lbl, cmap=lbl_cmap)
  al.set_title(lbl_title)
  plt.tight_layout()

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