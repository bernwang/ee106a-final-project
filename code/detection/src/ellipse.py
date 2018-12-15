import numpy as np
from numpy.linalg import eig, inv
import skimage
from skimage import data, color, img_as_ubyte
from skimage.feature import canny
import matplotlib.pyplot as plt


def filter_canny(mask, canny_edges, thresh=3):
  h, w = mask.shape
  res = np.zeros(canny_edges.shape)
  for i in range(h//2):
    row = np.nonzero(mask[i,:])[0]
    if len(row) > 1:
      l_edge = row[0]
      r_edge = row[-1]
      valid_canny = np.nonzero(canny_edges[i,:])[0]
      valid_canny = valid_canny[np.where(np.abs(valid_canny - l_edge) > thresh)[0]]
      valid_canny = valid_canny[np.where(np.abs(valid_canny - r_edge) > thresh)[0]]
      res[i, valid_canny] = 1
  return res

def filter_mask_edges(mask_edges):
  h, w = mask_edges.shape
  res = np.zeros(mask_edges.shape)
  for i in range(h//2):
    row = np.nonzero(mask_edges[i,:])[0]
    if len(row) > 1:
      res[i, row] = 1
  return res

def get_edges_from_mask(mask):
  col_ind = np.nonzero(np.apply_along_axis(any, 0, mask))[0]
  start_col = col_ind[0]
  end_col = col_ind[-1]
  curr_col = start_col
  xs = []
  ys = []
  while curr_col <= end_col:
    y = np.nonzero(mask[:,curr_col])[0][0]
    xs.append(curr_col)
    ys.append(y)
    curr_col += 1

  return xs, ys

def fit_ellipse(image, mask):
  """
  Given an image and a mask, find the ellipse that best fits the lip of the cup.

  Input:
    image: cropped image of the object
    mask: cropped mask of the object

  Output:
    result: dictionary holding all the information about the fitted ellipse
  """

  xs, ys = get_edges_from_mask(mask)

  top_row = np.nonzero(np.apply_along_axis(any, 1, mask))[0][0]
  bot_row = np.nonzero(np.apply_along_axis(any, 1, mask))[0][-1]
  mid_row = int((bot_row+top_row)/2)

  mask_edges = np.zeros(mask.shape)
  mask_edges[ys, xs] = 1

  canny_edges = canny((image*mask), sigma=2.0,
                low_threshold=0.55, high_threshold=0.8)

  filtered_mask_edges = filter_mask_edges(mask_edges)
  filtered_canny_edges = filter_canny(mask, canny_edges, thresh=5)
  edges = np.maximum(filtered_mask_edges, filtered_canny_edges)
  ys, xs = list(map(lambda arr: arr.tolist(), np.nonzero(edges)))


  a = fitEllipse(np.array(xs), np.array(ys))
  phi = ellipse_angle_of_rotation(a)
  arc = 2
  R = np.arange(0,arc*np.pi, 0.01)
  center = ellipse_center(a)
  a, b = ellipse_axis_length(a)

  xx = center[0] + a*np.cos(R)*np.cos(phi) - b*np.sin(R)*np.sin(phi)
  yy = center[1] + a*np.cos(R)*np.sin(phi) + b*np.sin(R)*np.cos(phi)


  result = dict()
  result['xs'] = np.array(xs)
  result['ys'] = np.array(ys)
  result['center'] = center
  result['xx'] = np.array(xx)
  result['yy'] = np.array(yy)

  return result

def fitEllipse(x,y):
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    return a

def ellipse_center(a):
  b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
  num = b*b-a*c
  x0=(c*d-b*f)/num
  y0=(a*f-b*d)/num
  return np.array([x0,y0])

def ellipse_angle_of_rotation( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    return 0.5*np.arctan(2*b/(a-c))


def ellipse_axis_length( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])

