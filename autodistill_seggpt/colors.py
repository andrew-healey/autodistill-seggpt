from math import floor
from random import random
import numpy as np

# use color palette
from palettable.wesanderson import Moonrise7_5

moonrise_palette = Moonrise7_5.colors
moonrise_palette = np.asarray([list(color) for color in moonrise_palette])

curr_idx = 0
def moonrise_rgb():
  global curr_idx
  ret = Moonrise7_5.colors[curr_idx]
  curr_idx = (curr_idx + 1) % len(Moonrise7_5.colors)
  return np.asarray(list(ret))

def reset_colors():
  global curr_idx
  curr_idx = 0

# use white
def white_rgb():
    return np.asarray([255,255,255])
white_palette = white_rgb()[None,...]

# choose your preset
next_rgb = white_rgb
palette = white_palette