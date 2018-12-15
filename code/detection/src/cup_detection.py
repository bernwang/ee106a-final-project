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

# Directory for Mask RCNN
MASK_RCNN_DIR = os.path.abspath("./Mask_RCNN")
  

global IMAGE_DIR, OUTPUT_DIR
# Directory of images to run detection on
IMAGE_DIR = "../images"

# Directory to save data to
OUTPUT_DIR = "../output"


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
  """
  This function runs the setup needed to run Mask RCNN for image detection.
  Only need to run this once per set of images; it sets the global model 
  variable, which is how it is determined whether or not setup needs to occur.
  """

  print("starting setup")

  # Import Mask RCNN
  sys.path.append(MASK_RCNN_DIR)  # To find local version of the library
  from mrcnn import utils
  import mrcnn.model as modellib
  # Import COCO config
  sys.path.append(os.path.join(MASK_RCNN_DIR, "samples/coco/"))  # To find local version
  import coco

  # Directory to save logs and trained model
  MODEL_DIR = os.path.join(MASK_RCNN_DIR, "logs")

  # Local path to trained weights file
  COCO_MODEL_PATH = os.path.join(MASK_RCNN_DIR, "mask_rcnn_coco.h5")
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
  """
  Runs Mask RCNN on the given image and visualizes the results. If this is the first
  time running Mask RCNN on this batch, this function will setup the required
  packages and model.
  This function will continuously try to detect objects from the image until it 
  finds something.

  Input:
    image: image to run image detection on

  Output:
    r: dictionary of the results containing information about each of the detected
      objects
  """

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

  r = results[0]

  print("visualizing")
  visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                              class_names, r['scores'])

  return r


def save_cropped_images(filename, class_name, only_one=False):
  """
  Given the filename of an image we want to run image detection on, this will find
  all instances of class_name. If the only_one flag is set, then it will make sure 
  that there is only one instance of it. Each of the results are shown to the user
  for confirmation that they are indeed the instances of objects that they want. 
  This is to prevent there from being false positives in the case of bad object
  classification or if there happened to be the objects within the frame of the 
  image but not our desired instances.

  Input:
    filename: string of the name of the file of the image with respect to its 
      location in the IMAGE_DIR directory
    class_name: string of the class name of the desired object (must be chosen
      from the class_names list above)
    only_one: boolean that determines if the results are limited to just one 
      instance

    Output:
      cropped_ims: list of the cropped images
      cropped_masks: list of the cropped masks
      disps: list of the displacements for each of the cropped images/masks 
        with respect to the original image
  """

  image = skimage.io.imread(os.path.join(IMAGE_DIR, filename))

  # get the results from Mask RCNN
  r = get_detection_data(image)
  
  CLASS_INDEX = class_names.index(class_name)

  cropped_ims = []
  cropped_masks = []
  disps = []

  if CLASS_INDEX in r['class_ids']:

    # get the results for only the objects in the class that we are interested in
    ii = [i for i, class_id in enumerate(r['class_ids']) if class_id == CLASS_INDEX]

    for i in ii:

      # get the coordinates for the region of interest 
      y1, x1, y2, x2 = r['rois'][i]
      cropped_im = image[y1-10:y2,x1-10:x2+10]

      # have the user confirm that this object is the correct one
      # this is to prevent false positives and to ignore other objects in the field of
      # view from affecting our project
      plt.imshow(cropped_im)
      plt.show()
      to_keep = input("is this the correct object? press y if True, anything else if False: ")
      if to_keep.lower() != "y":
        continue

      im_name = filename.split(".")[0]

      # get the mask from the results 
      mask = r['masks'][:,:,i]
      cropped_mask = mask[y1-10:y2,x1-10:x2+10]

      # get the displacement of the cropped image and mask so that their location in
      # the original image is preserved
      dis = np.array([y1-10, x1-10])

      cropped_ims.append(cropped_im)
      cropped_masks.append(cropped_mask)
      disps.append(dis)

      # dealing with naming conventions for only one instance of the object or many
      if only_one:
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
    # if nothing of class_name was detected, raise an exception
    raise Exception("Failed to detect any objects of the class " + class_name)

  # if the right object of class_name was not detected, raise an exception
  if len(cropped_ims) == 0:
    raise Exception("Failed to detect any objects of the class " + class_name)

  return cropped_ims, cropped_masks, disps


def find_obj(filename, class_name, only_one=False):
  """
  Given the filename of an image we want to run image detection on, this will find
  all instances of class_name. If the only_one flag is set, then it will make sure 
  that there is only one instance of it. For all the objects found, we find the 
  ellipse that best fits the lip of the cup. We then return a list of all the 
  ellipses found for all the objects.

  Input:
    filename: string of the name of the file of the image with respect to its 
      location in the IMAGE_DIR directory
    class_name: string of the class name of the desired object (must be chosen
      from the class_names list above)
    only_one: boolean that determines if the results are limited to just one 
      instance

    Output:
      ellipses: list of the ellipses coordinates with respect to the original image
  """

  image = skimage.io.imread(os.path.join(IMAGE_DIR, filename))
  im_name = filename.split(".")[0]

  # checks to see if the displacement file exists, which means that the objects were
  # already detected, so running Mask RCNN is unnecessary
  # either way, get the lists of cropped images, masks, and displacements
  if not os.path.isfile("{}/{}_dis.npy".format(OUTPUT_DIR, im_name)) and not os.path.isfile("{}/{}_dis_0.npy".format(OUTPUT_DIR, im_name)):
    ret = save_cropped_images(filename, class_name, only_one)
    if ret:
      cropped_images, cropped_masks, disps = ret
    else:
      raise Exception("No objects were found.")

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

  # if no images were found, raise an Exception
  if len(cropped_images) == 0:
    raise Exception("No objects were found.")


  ellipses = []
  for i in range(len(cropped_images)):
    cropped_image = cropped_images[i]
    cropped_mask = cropped_masks[i]
    disp = disps[i]

    cropped_image = color.rgb2gray(cropped_image)

    # find the best fitted ellipse with respect to the cropped image
    cropped_ellipse = fit_ellipse(cropped_image, cropped_mask)

    ellipse = dict()

    # use the displacement information to find the ellipse information for the original image
    dis_y, dis_x = disp
    ellipse['xs'] = cropped_ellipse['xs'] + dis_x
    ellipse['xx'] = cropped_ellipse['xx'] + dis_x
    ellipse['ys'] = cropped_ellipse['ys'] + dis_y
    ellipse['yy'] = cropped_ellipse['yy'] + dis_y
    ellipse['center'] = (cropped_ellipse['center'][0] + dis_x, cropped_ellipse['center'][1] + dis_y)

    # plot the edges found used to find the ellipse
    plt.imshow(image)
    plt.scatter(ellipse['xs'], ellipse['ys'], c='b')
    plt.show()

    # plot the ellipse and its center
    plt.imshow(image)
    plt.scatter(ellipse['xx'], ellipse['yy'], c='b')
    plt.scatter(ellipse['center'][0], ellipse['center'][1])
    plt.show()

    ellipses.append(ellipse)

  # if there is more than one ellipse, plot all of them together on the same image
  if len(ellipses) > 1:
    plt.imshow(image)
    for ellipse in ellipses:
      plt.scatter(ellipse['xx'], ellipse['yy'], c='b')
      plt.scatter(ellipse['center'][0], ellipse['center'][1])
    plt.show()


  return ellipses


def get_center(filename, class_name, only_one=False):
  """
  Find the centers of all the objects that are of class class_name in the 
  image labeled filename.

  Input:
    filename: string of the name of the file of the image with respect to its 
      location in the IMAGE_DIR directory
    class_name: string of the class name of the desired object (must be chosen
      from the class_names list above)
    only_one: boolean that determines if the results are limited to just one 
      instance

    Output:
      centers: list of the centers of the lips of all the cups
  """
  ellipses = find_obj(filename, class_name, only_one)
  centers = []
  for ellipse in ellipses:
    center = ellipse['center']
    centers.append(center)
  return centers


def get_centers_for_calib():
  """
  For calibration purposes, get the centers of the lips of the cups and save them
  in a file.

  Output:
    centers: list of the centers of the lips of all the cups in order
  """

  print("waiting")

  # get centers of all the cups in the grid
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
      center = get_center(filename, "cup", only_one=True)[0]

    except Exception as e:
      print(e)
      # put (0, 0) as the center if it could not be found
      center = (0, 0)
      pass
    centers_d[i] = center

  # put all the centers in order based on their filename 
  # since the above code might not go through the files in order
  # this also fills in the gaps if there are missing pictures
  centers = []
  for i in range(max_i + 1):
    centers.append(centers_d.get(i, (0, 0)))

  # save the centers
  np.save(os.path.join(OUTPUT_DIR, "centers.npy"), np.array(centers)) 

  return centers


def get_homography_matrix(uv_pts, xy_pts):
  """
  Get the homography matrix relating the spatial xy coordinates from Baxter to the 
  uv points on a 2D image.

  Input:
    uv_pts: the centers of the cups used for calibration
    xy_pts: the actual x, y coordinates that the cups were at

  Output:
    H: the homography matrix relating xy to uv
  """

  # make sure that the object points don't include the z coordinate
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


def get_cup_coords(filename, H):
  """
  Get the xy coordinates of the two cups in filename and save the coordinates.

  Input:
    filename: string of the name of the file of the image with respect to its 
      location in the IMAGE_DIR directory
    H: homography matrix relating xy coordinates to uv coordinates
  """

  centers = get_center(filename, "cup", only_one=False)

  # the pouring cup is the one farther to the right from Baxter's POV
  # so cups_in_order is [pouring cup, empty cup]
  cups_in_order = np.array(sorted(centers, key=lambda x: x[1]))
  xy_coords = np.linalg.inv(H).dot(np.vstack((cups_in_order.T, np.ones(2)))).T
  xy_coords[:,0] = xy_coords[:,0] / xy_coords[:,2]
  xy_coords[:,1] = xy_coords[:,1] / xy_coords[:,2]
  xy_coords = xy_coords[:,:2]
  print(xy_coords)
  np.save("/home/cc/ee106a/fa18/class/ee106a-acw/ros_workspaces/lab7/src/planning/src/cup_positions/pourer.npy", xy_coords[0])
  np.save("/home/cc/ee106a/fa18/class/ee106a-acw/ros_workspaces/lab7/src/planning/src/cup_positions/receiver.npy", xy_coords[1])
  return xy_coords


def get_valid_coords(object_pts, centers):
  """
  When finding the centers for calibration, the objects might not be detected or there
  might be some other issue. When there is a problem, the center coordinates is (0, 0).
  We want to remove those coordinates, along with the corresponding coordinates in
  object_pts.

  Input:
    object_pts: the xy coordinates
    centers: the uv coordinates of the centers found
  """
  inds = []
  for i in range(len(centers)):
    c = centers[i]
    if not(c[0] == 0 and c[1] == 0):
      inds.append(i)

  object_pts = object_pts[inds]
  centers = centers[inds]

  return object_pts, centers


def main():

  calibrating = False
  get_homography = False
  find_center_of_obj = False
  get_cups = True

  global IMAGE_DIR, OUTPUT_DIR
  OUTPUT_DIR = "calib"
  IMAGE_DIR = os.path.join(OUTPUT_DIR, "calib_images")

  if calibrating:
    # find the centers of the cups in the images in the calib folder
    # save those centers in a file

    centers = get_centers_for_calib()
    print(centers)
  
  if find_center_of_obj:
    # find the center of the cup in the image listed below

    filename = "cups.jpg"
    class_name = "cup"

    center = get_center(filename, class_name, True)[0]

  if get_cups:
    # get the coordinates of the two cups 

    object_pts = np.load(os.path.join(OUTPUT_DIR, "points.npy"))

    centers = np.load(os.path.join(OUTPUT_DIR, "centers.npy"))

    object_pts, centers = get_valid_coords(object_pts, centers)
    object_pts_xy = object_pts[:,:2]
    H = get_homography_matrix(centers, object_pts_xy)
    print("homography matrix:\n", H)

    OUTPUT_DIR = "output"
    IMAGE_DIR = "images"
    filename = "cups1.jpg"
    get_cup_coords(filename, H)

  if get_homography:
    # get the homography matrix using the saved points and centers files

    object_pts = np.load(os.path.join(OUTPUT_DIR, "points.npy"))

    centers = np.load(os.path.join(OUTPUT_DIR, "centers.npy"))

    object_pts, centers = get_valid_coords(object_pts, centers)
    object_pts_xy = object_pts[:,:2]
    H = get_homography_matrix(centers, object_pts_xy)
    print(H)

    if find_center_of_obj:
      xy = np.linalg.inv(H).dot(np.array([center[0], center[1], 1]))
      xy /= xy[2]
      print(xy)
      save_path = os.path.join("output", "xy.npy")
      np.save(save_path, xy[:2])



if __name__ == "__main__":
  main()


