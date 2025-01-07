import numpy as np


def intersection(boxes0, boxes1):
  """Given two np.arrays of (hyper)-bounding boxes of n_boxes x 2 x n_dimension
  (usually 2), calculates the pairwise intersection area (or volume, etc) of
  each pair between boxes0 and boxes1 and returns as a matrix of
  len(boxes0) x len(boxes1)."""
  right = np.minimum(boxes0[:,None, 1], boxes1[None, :,1])
  left = np.maximum(boxes0[:,None, 0], boxes1[None, :,0])
  intersection_ = right - left
  intersection_[intersection_ < 0] = 0
  return np.prod(intersection_, axis=2)


def area(boxes):
  """Given np.array of (hyper)-bounding boxes of n_boxes x 2 x n_dimension
  (usually 2), calculates the area (or volume, etc) of each box and returns a
  np.array of n_boxes."""
  return np.prod(np.diff(boxes, axis=1),axis=2)[:, 0]


def iou(boxes0, boxes1):
  """Given two np.arrays of (hyper)-bounding boxes of n_boxes x 2 x n_dimension
  (usually 2), calculates the pairwise intersection over union values, returning
  a len(boxes0) x len(boxes1) matrix of values between 0 and 1."""
  intersection_ = intersection(boxes0, boxes1)
  area_ = area(boxes0)[:, None] + area(boxes1)[None]
  return intersection_ / (area_ - intersection_)


def inclusion(boxes0, boxes1):
  """Given two np.arrays of (hyper)-bounding boxes of n_boxes x 2 x n_dimension
  (usually 2), calculates the pairwise percentage of box1 that is included in
  box2, returning a len(boxes0) x len(boxes1) matrix of values between 0 and 1.
  """
  intersection_ = intersection(boxes0, boxes1)
  area_ = area(boxes0)[:, None]
  return intersection_ / (area_)

