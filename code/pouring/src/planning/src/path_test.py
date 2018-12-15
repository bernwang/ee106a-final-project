#!/usr/bin/env python
"""
Adapted from the Path Planning Script for Lab 8 by Valmik Prabhu
"""

import os
import sys
import rospy
import numpy as np
import tf
from moveit_msgs.msg import OrientationConstraint
from geometry_msgs.msg import PoseStamped

from path_planner import PathPlanner
from baxter_interface import Limb, CameraController, Gripper
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import matplotlib.pyplot as plt
import skimage.io

camera_image = None
IMAGE_DIR = "/home/cc/ee106a/fa18/class/ee106a-acw/project/calib3/calib_images"
POINTS_DIR = "/home/cc/ee106a/fa18/class/ee106a-acw/project/calib3/points.npy"
OBJECT_HEIGHT = 0.022
GRABBING_OFFSET = 0.05
POURING_CUP_PATH = "/home/cc/ee106a/fa18/class/ee106a-acw/ros_workspaces/lab7/src/planning/src/cup_positions/pourer.npy"
RECEIVING_CUP_PATH = "/home/cc/ee106a/fa18/class/ee106a-acw/ros_workspaces/lab7/src/planning/src/cup_positions/receiver.npy"

"""UTILS"""
def getPouringAxis(fill_cup_pos, pouring_cup_pos):
    #returning normalized pouring axis
    z_axis = np.array([0,0,1])
    pouring_vec = fill_cup_pos - pouring_cup_pos
    w = np.cross(z_axis, pouring_vec)
    return w / np.linalg.norm(w)

def getQuaternion(axis,theta):

    s = np.sin(theta / 2)
    c = np.cos(theta / 2)
    
    return c, axis * s

def quatMult(q, p):
    q0 = q[0]
    q_vec = q[1]
    p0 = p[0]
    p_vec = p[1]

    c = q0* p0 - np.dot(q_vec, p_vec)
    vec = q0 * p_vec + p0 * q_vec + np.cross(q_vec, p_vec)

    return c, vec

# the cup to gripper quaternion is 

def getGripQuat(cupquat):
    cup_to_grip = getQuaternion(np.array([0,1,0]), np.pi / 2)
    #return quatMult(cupquat, cup_to_grip)
    return quatMult(cup_to_grip, cupquat)
########################################################

def main():
    """
    Main Script
    """
    # Setting up head camera
    head_camera = CameraController("left_hand_camera")
    head_camera.resolution = (1280, 800)
    head_camera.open()

    print("setting balance")
    happy = False

    while not happy:

        e = head_camera.exposure
        print("exposue value: " + str(e))

        e_val = int(input("new exposure value"))

        head_camera.exposure = e_val
        satisfaction = str(raw_input("satified? y/n"))
        happy = 'y' == satisfaction


    planner = PathPlanner("right_arm")
    listener = tf.TransformListener()
    grip = Gripper('right')


    grip.calibrate()
    rospy.sleep(3)
    grip.open()


    # creating the table obstacle so that Baxter doesn't hit it
    table_size = np.array([.5, 1, 10])
    table_pose = PoseStamped()
    table_pose.header.frame_id = "base"
    thickness = 1

    table_pose.pose.position.x = .9
    table_pose.pose.position.y = 0.0
    table_pose.pose.position.z = -.112 - thickness / 2
    table_size = np.array([.5, 1, thickness])

    planner.add_box_obstacle(table_size, "table", table_pose)


    raw_input("gripper close")
    grip.close()


    # ###load cup positions found using cup_detection.py ran in virtual environment
    # start_pos_xy = np.load(POURING_CUP_PATH)
    # goal_pos_xy = np.load(RECEIVING_CUP_PATH)

    # start_pos = np.append(start_pos_xy, OBJECT_HEIGHT - GRABBING_OFFSET)
    # goal_pos = np.append(start_pos_xy, OBJECT_HEIGHT - GRABBING_OFFSET)
    # #### END LOADING CUP POSITIONS

    camera_subscriber = rospy.Subscriber("cameras/left_hand_camera/image", Image, get_img)


    Kp = 0.1 * np.array([0.3, 2, 1, 1.5, 2, 2, 3]) # Stolen from 106B Students
    Kd = 0.01 * np.array([2, 1, 2, 0.5, 0.5, 0.5, 0.5]) # Stolen from 106B Students
    Ki = 0.01 * np.array([1, 1, 1, 1, 1, 1, 1]) # Untuned
    Kw = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]) # Untuned
    cam_pos= [0.188, -0.012, 0.750]

    ##
    ## Add the obstacle to the planning scene here
    ##

    # Create a path constraint for the arm
    orien_const = OrientationConstraint()
    orien_const.link_name = "right_gripper";
    orien_const.header.frame_id = "base";
    orien_const.orientation.y = -1.0;
    orien_const.absolute_x_axis_tolerance = 0.1;
    orien_const.absolute_y_axis_tolerance = 0.1;
    orien_const.absolute_z_axis_tolerance = 0.1;
    orien_const.weight = 1.0;
    horizontal = getQuaternion(np.array([0,1,0]), np.pi)

    z_rot_pos = getQuaternion(np.array([0,0,1]), np.pi / 2)

    orig = quatMult(z_rot_pos, horizontal)
    orig = getQuaternion(np.array([0,1,0]), np.pi / 2)

    #IN THE VIEW OF THE CAMERA
    #CORNER1--------->CORNER2
    #   |                |
    #   |                |
    #   |                |
    #CORNER3 ------------|
    width = 0.3
    length = 0.6
    CORNER1 =  np.array([0.799, -0.524, -0.03])
    CORNER2 = CORNER1 + np.array([-width, 0, 0])
    CORNER3 = CORNER1 + np.array([0, length, 0])

    #CREATE THE GRID
    dir1 = CORNER2 - CORNER1
    dir2 = CORNER3 - CORNER1

    grid_vals = []
    ret_vals = []

    for i in range(0, 3):
        for j in range(0, 4):
            grid = CORNER1 + i * dir1 / 2 + j * dir2 / 3
            grid_vals.append(grid) 

            ret_vals.append(np.array([grid[0], grid[1], OBJECT_HEIGHT]))

    i = -1
    while not rospy.is_shutdown():
        for g in grid_vals:
            i += 1
            while not rospy.is_shutdown():
                try:
                    goal_1 = PoseStamped()
                    goal_1.header.frame_id = "base"

                    #x, y, and z position
                    goal_1.pose.position.x = g[0]
                    goal_1.pose.position.y = g[1]
                    goal_1.pose.position.z = g[2]

                    y = - g[1] + cam_pos[1]
                    x = g[0] - cam_pos[0]
                    q = orig

                    #Orientation as a quaternion

                    goal_1.pose.orientation.x = q[1][0]
                    goal_1.pose.orientation.y = q[1][1]
                    goal_1.pose.orientation.z = q[1][2]
                    goal_1.pose.orientation.w = q[0]
                    plan = planner.plan_to_pose(goal_1, [])

                    if planner.execute_plan(plan):
                    # raise Exception("Execution failed")
                        rospy.sleep(1)
                        
                        #grip.open()
                        raw_input("open")
                        rospy.sleep(1)
                        # plt.imshow(camera_image)
                        print("yay: " + str(i))
                       
                        pos = listener.lookupTransform("/reference/right_gripper", "/reference/base", rospy.Time(0))[0]
                        print("move succesfully to " + str(pos))
                        fname = os.path.join(IMAGE_DIR, "calib_{}.jpg".format(i))
                        skimage.io.imsave(fname, camera_image)


                    print(e)
                    print("index: ", i)
                else:
                    break

        print(np.array(ret_vals))
        # save the positions of the gripper so that the homography matrix can be calculated
        np.save(POINTS_DIR, np.array(ret_vals))
        print(np.load(POINTS_DIR))
        break


def get_img(msg):
    global camera_image
    camera_image = msg2cv(msg)

def msg2cv(msg):
    return CvBridge().imgmsg_to_cv2(msg, desired_encoding='rgb8')

if __name__ == '__main__':
    rospy.init_node('moveit_node')
    main()