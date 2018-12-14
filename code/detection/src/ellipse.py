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
  xs, ys = get_edges_from_mask(mask)

  top_row = np.nonzero(np.apply_along_axis(any, 1, mask))[0][0]
  bot_row = np.nonzero(np.apply_along_axis(any, 1, mask))[0][-1]
  mid_row = int((bot_row+top_row)/2)

  mask_edges = np.zeros(mask.shape)
  mask_edges[ys, xs] = 1

  canny_edges = canny((image*mask), sigma=2.0,
                low_threshold=0.55, high_threshold=0.8)

  # plt.imshow(canny_edges)
  # plt.show()

  filtered_mask_edges = filter_mask_edges(mask_edges)
  filtered_canny_edges = filter_canny(mask, canny_edges, thresh=5)
  edges = np.maximum(filtered_mask_edges, filtered_canny_edges)
  ys, xs = list(map(lambda arr: arr.tolist(), np.nonzero(edges)))
  # plt.imshow(edges)
  # plt.show()
  # print(np.max(edges))


  # divide_ind = np.nonzero(mask[:,start_col])[0][0]

  # edges[:divide_ind, :] = np.zeros((divide_ind, edges.shape[1]))

  # print(image.shape)
  # print(top_row, mid_row, bot_row)

  # plt.imshow(image)
  # plt.show()
  # plt.imshow(edges)
  # plt.show()
  # edges[:divide_ind, :] = np.zeros((divide_ind, edges.shape[1]))
  # plt.imshow(edges)
  # plt.show()

  # edges = canny(mask, sigma=2.0,
  #               low_threshold=0.45, high_threshold=0.8)
  # edges[:divide_ind, :] = np.zeros((divide_ind, edges.shape[1]))

  ######## obsolete: adds bottom edge from canny 
  # col_ind = np.nonzero(np.apply_along_axis(any, 0, edges))[0]
  # start_col = col_ind[0]
  # end_col = col_ind[-1]
  # curr_col = start_col
  # while curr_col <= end_col:
  #   col = np.nonzero(edges[:,curr_col])[0]
  #   if len(col) > 0:
  #     y = col[0]
  #     xs.append(curr_col)
  #     ys.append(y)
  #   curr_col += 1


  a = fitEllipse(np.array(xs), np.array(ys))
  # a = fitEllipse(np.nonzero(edges)[0], np.nonzero(edges)[1])
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

  # fig = plt.figure()
  # ax1 = fig.add_subplot(121)
  # ax2 = fig.add_subplot(122)
  # ax1.imshow(mask)


  # ax2.imshow(edges)
  # ax2.scatter(result['xs'], result['ys'], c='b')
  # fig.show()
  # print(mask.shape)
  # print(edges.shape)

  # plt.imshow(mask)
  # plt.show()
  # plt.imshow(edges)
  # plt.scatter(result['xs'], result['ys'], c='b')
  # plt.show()
  # plt.imshow(image)
  # plt.scatter(result['xs'], result['ys'], c='b')
  # plt.scatter(result['xs'], result['ys'], c='b')
  # plt.scatter(result['center'][0], result['center'][1])
  # plt.show()
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


# import matplotlib.pyplot as plt

# import skimage
# import os
# from skimage import data, color, img_as_ubyte
# from skimage.feature import canny
# from skimage.transform import hough_ellipse
# from skimage.draw import ellipse_perimeter

# # Load picture, convert to grayscale and detect edges
# # image_rgb = data.coffee()[0:220, 160:420]
# # image_gray = color.rgb2gray(image_rgb)

# # image_gray = color.rgb2gray(skimage.io.imread("cropped.jpg"))

# # edges = canny(image_gray, sigma=2.0,
# #               low_threshold=0.55, high_threshold=0.8)

# # plt.imshow(edges)
# # plt.show()

  
# image = color.rgb2gray(skimage.io.imread("cropped.jpg"))
# mask = np.load("cropped_mask.npy")
# #f.close()
# #XXXXXXXXXXXXXXXXXXXXXX
# # plt.imshow(mask)
# # plt.show()
# ellipse = fit_ellipse(mask)
# plt.imshow(image)
# plt.scatter(ellipse['xs'], ellipse['ys'], c='b')
# plt.show()

# matrix = np.zeros(mask.shape)
# matrix[ellipse['ys'], ellipse['xs']] = 1

# plt.imshow(matrix)
# plt.show()

# # Perform a Hough Transform
# # The accuracy corresponds to the bin size of a major axis.
# # The value is chosen in order to get a single high accumulator.
# # The threshold eliminates low accumulators
# result = hough_ellipse(matrix, accuracy=20, threshold=250,
#                        min_size=100, max_size=120)
# result.sort(order='accumulator')

# # Estimated parameters for the ellipse
# best = list(result[-1])
# yc, xc, a, b = [int(round(x)) for x in best[1:5]]
# orientation = best[5]

# # Draw the ellipse on the original image
# cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
# image_rgb[cy, cx] = (0, 0, 255)
# # Draw the edge (white) and the resulting ellipse (red)
# edges = color.gray2rgb(img_as_ubyte(matrix))
# edges[cy, cx] = (250, 0, 0)

# fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4),
#                                 sharex=True, sharey=True)

# ax1.set_title('Original picture')
# ax1.imshow(image_rgb)

# ax2.set_title('Edge (white) and result (red)')
# ax2.imshow(edges)

# plt.show()
