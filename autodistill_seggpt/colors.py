from math import floor
from random import random
import numpy as np

palette_registry = {}

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

palette_registry['moonrise'] = (moonrise_palette,moonrise_rgb,"instance")

# use white
def white_rgb():
    return np.asarray([255,255,255])
white_palette = white_rgb()[None,...]

palette_registry['white'] = (white_palette,white_rgb,"semantic")

# use r/g/b
rgb_palette = np.asarray([[255,0,0],[0,255,0],[0,0,255]])
def rgb_rgb():
  global curr_idx
  ret = rgb_palette[curr_idx]
  curr_idx = (curr_idx + 1) % len(rgb_palette)
  return np.asarray(list(ret))

palette_registry['rgb'] = (rgb_palette,rgb_rgb,"instance")

# choose your preset
preset = "white"

next_rgb,palette,seg_type = palette_registry[preset]