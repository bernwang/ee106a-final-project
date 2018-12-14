import os
import sys
import random
import math
import numpy as np
from numpy.linalg import eig, inv
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from ellipse import fit_ellipse
from tempfile import TemporaryFile
from skimage import data, color, img_as_ubyte
from skimage.io import imread
import time
import cv2
import glob

# Root directory of the project
ROOT_DIR = os.path.abspath("./Mask_RCNN")
  
global IMAGE_DIR, OUTPUT_DIR


# Directory of images to run detection on
IMAGE_DIR = "images"

# Directory to save data to
OUTPUT_DIR = os.path.join("output")
if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

model = None

def detection_setup():

  print("starting setup")

  # Import Mask RCNN
  sys.path.append(ROOT_DIR)  # To find local version of the library
  from mrcnn import utils
  import mrcnn.model as modellib
  # Import COCO config
  sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
  import coco

  # Directory to save logs and trained model
  MODEL_DIR = os.path.join(ROOT_DIR, "logs")

  # Local path to trained weights file
  COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
  # Download COCO trained weights from Releases if needed
  if not os.path.exists(COCO_MODEL_PATH):
      utils.download_trained_weights(COCO_MODEL_PATH)

  class InferenceConfig(coco.CocoConfig):
      # Set batch size to 1 since we'll be running inference on
      # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
      GPU_COUNT = 1
      IMAGES_PER_GPU = 1


  config = InferenceConfig()
  config.display()

  # Create model object in inference mode.
  global model
  model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

  # Load weights trained on MS-COCO
  model.load_weights(COCO_MODEL_PATH, by_name=True)

  print("finished setup")


def get_detection_data(image):
  from mrcnn import visualize

  if not model:
    detection_setup()

  # Run detection
  while True:
    try:
      print("detecting")
      results = model.detect([image], verbose=1)
      break
    except Exception:
      print("failed detection, will try again")
      pass
    import gc
    gc.collect()

  r = results[0]

  print("visualizing")
  visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                              class_names, r['scores'])

  return r


def save_cropped_images(filename, class_name, only_one=False):

  image = skimage.io.imread(os.path.join(IMAGE_DIR, filename))
  r = get_detection_data(image)
  
  CLASS_INDEX = class_names.index(class_name)

  cropped_ims = []
  cropped_masks = []
  disps = []

  if CLASS_INDEX in r['class_ids']:
    ii = [i for i, class_id in enumerate(r['class_ids']) if class_id == CLASS_INDEX]

    found = False

    for i in ii:

      y1, x1, y2, x2 = r['rois'][i]
      cropped_im = image[y1-10:y2,x1-10:x2+10]

      plt.imshow(cropped_im)
      plt.show()
      to_keep = input("is this the correct object? press y if True, anything else if False: ")
      if to_keep.lower() != "y":
        continue
      elif only_one:
        found = True

      im_name = filename.split(".")[0]
      mask = r['masks'][:,:,i]
      cropped_mask = mask[y1-10:y2,x1-10:x2+10]
      dis = np.array([y1-10, x1-10])
      cropped_ims.append(cropped_im)
      cropped_masks.append(cropped_mask)
      disps.append(dis)

      if found:
        skimage.io.imsave("{}/cropped_{}.jpg".format(OUTPUT_DIR, im_name), cropped_im)
        np.save("{}/cropped_{}_mask.npy".format(OUTPUT_DIR, im_name), cropped_mask)
        np.save("{}/{}_dis.npy".format(OUTPUT_DIR, im_name), dis)
        print("{}/{}_dis.npy".format(OUTPUT_DIR, im_name))
        break

      else:
        skimage.io.imsave("{}/cropped_{}_{}.jpg".format(OUTPUT_DIR, im_name, i), cropped_im)
        np.save("{}/cropped_{}_mask_{}.npy".format(OUTPUT_DIR, im_name, i), cropped_mask)
        np.save("{}/{}_dis_{}.npy".format(OUTPUT_DIR, im_name, i), dis)

  else:
    print("failed to detect", class_name)
    return None

  if len(cropped_ims) == 0:
    return None

  return cropped_ims, cropped_masks, disps


def find_obj(filename, class_name, only_one=False, get_bottom=False):

  image = skimage.io.imread(os.path.join(IMAGE_DIR, filename))
  im_name = filename.split(".")[0]

  if not os.path.isfile("{}/{}_dis.npy".format(OUTPUT_DIR, im_name)) and not os.path.isfile("{}/{}_dis_0.npy".format(OUTPUT_DIR, im_name)):
    ret = save_cropped_images(filename, class_name, only_one)
    if ret:
      cropped_images, cropped_masks, disps = ret
    else:
      raise Exception

  else:
    if only_one:
      cropped_images = [skimage.io.imread("{}/cropped_{}.jpg".format(OUTPUT_DIR, im_name))]
      cropped_masks = [np.load("{}/cropped_{}_mask.npy".format(OUTPUT_DIR, im_name))]
      disps = [np.load("{}/{}_dis.npy".format(OUTPUT_DIR, im_name))]
    else:
      filenames = glob.glob("{}/{}_dis_*.npy".format(OUTPUT_DIR, im_name))
      filname_ids = [int(name.split(".")[0][-1]) for name in filenames]

      cropped_images = []
      cropped_masks = []
      disps = []
      for id_num in filname_ids:
        cropped_images.append(skimage.io.imread("{}/cropped_{}_{}.jpg".format(OUTPUT_DIR, im_name, id_num)))
        cropped_masks.append(np.load("{}/cropped_{}_mask_{}.npy".format(OUTPUT_DIR, im_name, id_num)))
        disps.append(np.load("{}/{}_dis_{}.npy".format(OUTPUT_DIR, im_name, id_num)))

  if len(cropped_images) == 0:
    raise Exception


  ellipses = []
  for i in range(len(cropped_images)):
    cropped_image = cropped_images[i]
    cropped_mask = cropped_masks[i]
    disp = disps[i]

    cropped_image = color.rgb2gray(cropped_image)
    bottom_row_ind = np.nonzero(np.apply_along_axis(any, 1, cropped_mask))[0][-1]
    if get_bottom:
      col_inds = np.nonzero(cropped_mask[bottom_row_ind,:])[0]
      col_ind = col_inds[int(len(col_inds)/2)]
      return (bottom_row_ind, col_ind)
    col_ind = np.nonzero(np.apply_along_axis(any, 0, cropped_mask))[0]
    middle_x_ind = (col_ind[-1] + col_ind[0]) / 2

    cropped_ellipse = fit_ellipse(cropped_image, cropped_mask)

    ellipse = dict()

    dis_y, dis_x = disp
    ellipse['xs'] = cropped_ellipse['xs'] + dis_x
    ellipse['xx'] = cropped_ellipse['xx'] + dis_x
    ellipse['ys'] = cropped_ellipse['ys'] + dis_y
    ellipse['yy'] = cropped_ellipse['yy'] + dis_y
    ellipse['center'] = (cropped_ellipse['center'][0] + dis_x, cropped_ellipse['center'][1] + dis_y)

    # plt.imshow(cropped_mask)
    # plt.show()
    plt.imshow(image)
    plt.scatter(ellipse['xs'], ellipse['ys'], c='b')
    plt.show()
    plt.imshow(image)
    plt.scatter(ellipse['xx'], ellipse['yy'], c='b')
    plt.scatter(ellipse['center'][0], ellipse['center'][1])
    plt.show()

    ellipses.append(ellipse)

  plt.imshow(image)
  for ellipse in ellipses:
    plt.scatter(ellipse['xx'], ellipse['yy'], c='b')
    plt.scatter(ellipse['center'][0], ellipse['center'][1])
  plt.show()


  return ellipses


def get_center(filename, class_name, only_one=False):
  ellipses = find_obj(filename, class_name, only_one)
  centers = []
  for ellipse in ellipses:
    center = ellipse['center']
    centers.append(center)
  return centers


def get_centers_for_calib():

  print("waiting")
  # get centers of all the balls in the grid
  filenames = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))

  centers_d = {}
  max_i = 0
  for filename in filenames:
    try:
      i = int(filename.split(".")[0][-2:])
    except Exception:
      i = int(filename.split(".")[0][-1])
    filename = "calib_{}.jpg".format(i)
    if i > max_i:
      max_i = i
    print("working on " + filename)
    try:
      # if i in {4, 11, 8, 9}: # for calib3
      # if i in {0, 2, 3, 10, 8}: # for calib4
      if i in {3, 10, 8}: # for calib5 # 7 is great
        raise Exception
      center = get_center(filename, "cup", only_one=True)[0]
    except Exception as e:
      print(e)
      center = (0, 0)
      pass
    centers_d[i] = center

  centers = []
  for i in range(max_i + 1):
    centers.append(centers_d.get(i, (0, 0)))

  # print(centers)
  np.save(os.path.join(OUTPUT_DIR, "centers.npy"), np.array(centers)) 

  return centers



def get_bottom(filename):

  row, col = find_obj(filename, "cup", only_one=True, get_bottom=True)[0]
  return row, col

def get_homography_matrix(uv_pts, xy_pts):

  object_pts = np.array(xy_pts)[:,:2]

  centers = np.array(uv_pts)

  l = len(object_pts)

  A = np.zeros((2*l, 8))
  for i in range(l):
    uv = centers[i]
    xy = object_pts[i]
    A[2*i, :2] = xy
    A[2*i, 2] = 1
    A[2*i, 6:] = -uv[0] * xy
    A[2*i+1, 3:5] = xy
    A[2*i+1, 5] = 1
    A[2*i+1, 6:] = -uv[1] * xy

  b = centers.reshape((-1,1))

  H = np.linalg.lstsq(A, b, rcond=-1)[0]
  H = np.append(H, 1)

  return H.reshape((3,3))

def get_xy(uv, z, H):
  pixels = np.array([uv[0], uv[1], 1])
  A = H[:,[0,1,3]]
  a = np.linalg.inv(A).dot(pixels)
  b = np.linalg.inv(A).dot(H[:,2] * z)
  s = (1 + b[2]) / a[2]
  xy = s*a - b
  return xy


def get_cup_centers(filename, H):
  centers = get_center(filename, "cup", only_one=False)
  cups_in_order = np.array(sorted(centers, key=lambda x: x[1]))
  xy_coords = np.linalg.inv(H).dot(np.vstack((cups_in_order.T, np.ones(2)))).T
  xy_coords[:,0] = xy_coords[:,0] / xy_coords[:,2]
  xy_coords[:,1] = xy_coords[:,1] / xy_coords[:,2]
  xy_coords = xy_coords[:,:2]
  print(xy_coords)
  np.save("/home/cc/ee106a/fa18/class/ee106a-acw/ros_workspaces/lab7/src/planning/src/cup_positions/pourer.npy", xy_coords[0])
  np.save("/home/cc/ee106a/fa18/class/ee106a-acw/ros_workspaces/lab7/src/planning/src/cup_positions/receiver.npy", xy_coords[1])
  return xy_coords


def main():

  calibrating = False
  get_homography = False
  find_center_of_obj = False
  get_cups = True

  global IMAGE_DIR, OUTPUT_DIR
  OUTPUT_DIR = "calib5"
  IMAGE_DIR = os.path.join(OUTPUT_DIR, "calib_images")

  if calibrating:

    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    if not os.path.isdir(IMAGE_DIR):
        os.mkdir(IMAGE_DIR)

    centers = get_centers_for_calib()
    print(centers)
  
  if find_center_of_obj:

    filename = "cups.jpg"
    class_name = "cup"
    # filename = "ball5.jpg"
    # class_name = "sports ball"

    center = get_center(filename, class_name, True)[0]

  if get_cups:
    object_pts = np.load(os.path.join(OUTPUT_DIR, "points.npy"))
    # print(object_pts)

    centers = np.load(os.path.join(OUTPUT_DIR, "centers.npy"))

    object_pts, centers = get_valid_coords(object_pts, centers)
    object_pts_xy = object_pts[:,:2]
    H = get_homography_matrix(centers, object_pts_xy)
    print(H)

    OUTPUT_DIR = "output"
    IMAGE_DIR = "images"
    filename = "cups1.jpg"
    get_cup_centers(filename, H)

  if get_homography:
    object_pts = np.load(os.path.join(OUTPUT_DIR, "points.npy"))
    # print(object_pts)

    centers = np.load(os.path.join(OUTPUT_DIR, "centers.npy"))

    object_pts, centers = get_valid_coords(object_pts, centers)
    object_pts_xy = object_pts[:,:2]
    H = get_homography_matrix(centers, object_pts_xy)
    print(H)

    # print("true uv: ", centers)
    # print("true xyz: ", object_pts)
    # # predicted_xyz = np.vstack([get_xy(centers[i,:], object_pts[i,2], H) for i in range(len(centers))])
    # predicted_xyz = (np.linalg.inv(H).dot(np.vstack((centers.T, np.ones(len(centers)))))).T
    # print(predicted_xyz)
    # predicted_xyz[:,0] = predicted_xyz[:,0] / predicted_xyz[:,2]
    # predicted_xyz[:,1] = predicted_xyz[:,1] / predicted_xyz[:,2]
    # predicted_xyz = predicted_xyz[:,:2]

    # print("predicted xyz", predicted_xyz)
    # # plt.scatter()
    # plt.scatter(predicted_xyz[:,0], predicted_xyz[:,1], c='r')
    # plt.scatter(object_pts[:,0], object_pts[:,1], c='b')
    # plt.show()


    if find_center_of_obj:
      xy = np.linalg.inv(H).dot(np.array([center[0], center[1], 1]))
      xy /= xy[2]
      print(xy)
      save_path = os.path.join("output", "xy.npy")
      np.save(save_path, xy[:2])
    # u suc

def get_valid_coords(object_pts, centers):
  inds = []
  for i in range(len(centers)):
    c = centers[i]
    if not(c[0] == 0 and c[1] == 0):
      inds.append(i)

  object_pts = object_pts[inds]
  centers = centers[inds]

  return object_pts, centers






if __name__ == "__main__":
  main()


