#!/usr/bin/env python
import os
import sys
import rospy
import numpy as np
import tf
from moveit_msgs.msg import OrientationConstraint
from geometry_msgs.msg import PoseStamped

from path_planner import PathPlanner
from baxter_interface import Limb, CameraController, Gripper
import moveit_commander
# from intera_interface import Limb
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import matplotlib.pyplot as plt
import skimage.io
from sensor_msgs.msg import JointState
POURING_COORD = np.array([0.826, -0.215, 0.089])


thetas = dict()
group_name = "right_arm"
move_group = None
right_arm = None
joint_states_flag = True
initial_thetas = dict()
flag = False
DESIRED_RELATIVE_POSITION = np.array([0, -0.093, 0.1])
OBJECT_HEIGHT = 0.022
GRABBING_OFFSET = 0.05
POURING_CUP_PATH = "/home/cc/ee106a/fa18/class/ee106a-acw/ros_workspaces/lab7/src/planning/src/cup_positions/pourer.npy"
RECEIVING_CUP_PATH = "/home/cc/ee106a/fa18/class/ee106a-acw/ros_workspaces/lab7/src/planning/src/cup_positions/receiver.npy"

###load cup positions found using cup_detection.py ran in virtual environment
# start_pos_xy = np.load(POURING_CUP_PATH)
# goal_pos_xy = np.load(RECEIVING_CUP_PATH)
start_pos_xy = [0.798, -0.224, 0.021]
goal_pos_xy = [0.916, 0.103, -0.008]


#start_pos = np.append(start_pos_xy, OBJECT_HEIGHT - GRABBING_OFFSET)
#goal_pos = np.append(start_pos_xy, OBJECT_HEIGHT - GRABBING_OFFSET)
#### END LOADING CUP POSITIONS
def send_image(path):
    """
    Send the image located at the specified path to the head
    display on Baxter.

    @param path: path to the image file to load and send
    """
    img = cv2.imread(path)
    msg = cv_bridge.CvBridge().cv2_to_imgmsg(img, encoding="bgr8")
    pub = rospy.Publisher('/robot/xdisplay', Image, latch=True)
    pub.publish(msg)
    # Sleep to allow for image to be published.
    rospy.sleep(1)

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
def callback(message):
    global thetas
    global joint_states_flag
    global initial_thetas
    #Print the contents of the message to the console
    # print(rospy.get_name() + ": I heard %s" % message.data)
    names = message.name
    positions = message.position
    # print(names)
    # print(thetas)
    d = {}
    for i in range(len(names)):
        d[names[i]] = positions[i]
    thetas = d
    # if not joint_states_flag:
    #     initial_thetas = right_arm.joint_angles()
    #     joint_states_flag = True
    # thetas = [pos[8], pos[7], pos[6], pos[3], pos[2], pos[5], pos[4]]
    # raw_input("move the cup to desired position: ")

    # cup_position, cup_quat = listener.lookupTransform("right_gripper", "base", rospy.Time(0))
    # print(cup_quat)
    # cup_position = np.array(cup_position)
    # orig_quat = [cup_quat[3], np.array(cup_quat[0:3])]

    # step_size = 0.1
    # cup_quats = [getQuaternion(w, np.pi * step_size * i) for i in range(0, 10)]
    # grip_quats = [quatMult(orig_quat, c) for c in cup_quats]

    # positions = [cup_position for _ in range(0, 10)]
    # print(thetas)
    # if flag:
    #     iters = 1
    #     while initial_thetas['right_w2'] > -np.pi:
    #         print(right_arm.joint_angles())
    #         initial_thetas['right_w2'] -= .4 *10/ (10 + iters)
    #         right_arm.set_joint_positions(initial_thetas)
    #         rospy.sleep(0.1)
    #         iters += 1

    # if "right_w2" in thetas:
    #     print(thetas["right_w2"])
def configureLeftHand():
    planner = PathPlanner("left_arm")

    position = np.array([0.516, 0.485, 0.362])
    orientation = [0.647, 0.655, -0.281, 0.271]

    executePositions(planner, [position], [orientation])


def pour(planner):
    global flag
    global initial_thetas
    initial_thetas = right_arm.joint_angles()
    print(initial_thetas)
    
    # initial_thetas['right_w2'] += -1 #.4 *20/ (20 + iters)
    # print(initial_thetas['right_w2'])
    # while not rospy.is_shutdown():
    #     try:
    #         while True:
    #             plan = planner.plan_to_angle(initial_thetas,[])
    #             if planner.execute_plan(plan):
    #                 print("yay")
    #                 break
    #             else:
    #                 print("failure")
    #                 break
    #     except Exception as e:
    #         print(e)
    #         # break
    #     break
    # return


    iters = 1
    while iters < 10:
        print(initial_thetas)
        initial_thetas['right_w2'] += -0.2 #.4 *20/ (20 + iters)
        while not rospy.is_shutdown():
            try:
                while True:
                    # print(initial_thetas)
                    # print(len(initial_thetas))
                    print(initial_thetas['right_w2'])
                    plan = planner.plan_to_angle(initial_thetas,[])
                    if planner.execute_plan(plan):
                        print("yay")
                        break
                    else:
                        print("failure")
                        break
            except Exception as e:
                print(e)
                break
            break
        rospy.sleep(0.1)
        iters += 1

    flag = True
    # while initial_thetas['right_w2'] > -np.pi:
    #     print(right_arm.joint_angles())
    #     initial_thetas['right_w2'] -= .4#.4 *20/ (20 + iters)
    #     right_arm.set_joint_positions(initial_thetas)
    #     rospy.sleep(0.1)
    #     iters += 1



#Define the method which contains the node's main functionality
def listener():

    #Run this program as a new node in the ROS computation graph
    #called /listener_<id>, where <id> is a randomly generated numeric
    #string. This randomly generated name means we can start multiple
    #copies of this node without having multiple nodes with the same

    #Create a new instance of the rospy.Subscriber object which we can 
    #use to receive messages of type std_msgs/String from the topic /chatter_talk.
    #Whenever a new message is received, the method callback() will be called
    #with the received message as its first argument.
    rospy.Subscriber("robot/joint_states", JointState, callback)


    #Wait for messages to arrive on the subscribed topics, and exit the node
    #when it is killed with Ctrl+C
    # rospy.spin()

#### translation: [0.723, -0.292, -0.174]
#### orientation: [-0.109, 0.994, -0.021, 0.001]
#executes the following set of positions skips failures
def executePositions(planner, positions, orientations, const=[]):
    while not rospy.is_shutdown():
        for i in range(0, len(positions)):
            print(i)
            while not rospy.is_shutdown():
                try:
                    g = positions[i]
                    q = orientations[i]
                    goal_1 = PoseStamped()
                    goal_1.header.frame_id = "base"

                    #x, y, and z position
                    goal_1.pose.position.x = g[0]
                    goal_1.pose.position.y = g[1]
                    goal_1.pose.position.z = g[2]



                    #Orientation as a quaternion
                    goal_1.pose.orientation.x = q[0]
                    goal_1.pose.orientation.y = q[1]
                    goal_1.pose.orientation.z = q[2]
                    goal_1.pose.orientation.w = q[3]

                    while True:
                        plan = planner.plan_to_pose(goal_1, const)

                        if planner.execute_plan(plan):
                        # raise Exception("Execution failed")
                            #raw_input("owo")
                            rospy.sleep(0.2)
                            # plt.imshow(camera_image)
                            break
                        else:
                            print("failure on pose " + str(i))
                except Exception as e:
                    print(e)
                    print("index: ", i)
                    break
                
                break
        print("done")
        return
#assumes the cup is already gripped in a horizontal fashion, then pours to filling using a desired relative
def moveAndPour(planner, fill_pos, const = []):
    orientation = getQuaternion(np.array([0,1,0]), np.pi / 2)
    o = np.array([orientation[1][0],orientation[1][1],orientation[1][2],orientation[0]])
    position = fill_pos + DESIRED_RELATIVE_POSITION

    orien_const = OrientationConstraint()
    orien_const.link_name = "right_gripper";
    orien_const.header.frame_id = "base";
    orien_const.orientation.x = o[0];
    orien_const.orientation.y = o[1];
    orien_const.orientation.z = o[2];
    orien_const.orientation.w = o[3];
    orien_const.absolute_x_axis_tolerance = 0.1;
    orien_const.absolute_y_axis_tolerance = 0.1;
    orien_const.absolute_z_axis_tolerance = 0.1;

    # orien_const.weight = 1.0;
    executePositions(planner, [position], [o], [orien_const])
    #pour(planner)

    orients = []

    for i in range(1, 11):
        theta = -0.2 * i

        rot = getQuaternion(np.array([1,0,0]), theta)

        o_r = quatMult(rot, orientation)
        o_r = o_r[1][0], o_r[1][1], o_r[1][2], o_r[0]
        orients.append(o_r)

    executePositions(planner,[position for _ in range(0, 10)], orients)






def getDefaultQuat():
    y_rot = getQuaternion(np.array([0,1,0]), np.pi * 3 / 4)
    z_rot = getQuaternion(np.array([0,0,1]), np.pi / 2)
    return quatMult(z_rot, y_rot)
    #return y_rot
def putCupObstacle(planner, position, name):
    cup_size = np.array([0.06,0.06, 0.12])
    cup_pose = PoseStamped()
    cup_pose.pose.position.x = position[0] 
    cup_pose.pose.position.y = position[1]
    cup_pose.pose.position.z = position[2] - 0.06

    planner.add_box_obstacle(cup_size, name , cup_pose)
def removeCup(planner, name):
    planner.remove_obstacle("name")

def main():
    global thetas

    configureLeftHand()
    planner = PathPlanner("right_arm")

    grip = Gripper('right')
    grip.calibrate()

    raw_input("gripper calibrated")
    grip.open()

    table_size = np.array([3, 1, 10])
    table_pose = PoseStamped()
    table_pose.header.frame_id = "base"

    table_pose.pose.position.x = .9
    table_pose.pose.position.y = 0.0
    table_pose.pose.position.z = -5 -.112
    #put cup
    cup_pos = start_pos_xy

    end_pos = goal_pos_xy

    putCupObstacle(planner, cup_pos, "cup1")
    putCupObstacle(planner, end_pos, "cup2")



    planner.add_box_obstacle(table_size, "table", table_pose)

    #move gripper to behind cup

    position = cup_pos + np.array([-0.1, 0, -0.02])
    orientation = np.array([0, np.cos(np.pi/4), 0, np.cos(np.pi / 4)])

    executePositions(planner,[position], [orientation])

    raw_input("moved behind cup, press enter")

    #move to cup and remove cup1 as obstacle since picking up
    removeCup(planner, "cup1")


    position = cup_pos + np.array([0,0,-0.05])
    executePositions(planner,[position], [orientation])

    raw_input("moved to cup, press to close")

    grip.close()
    rospy.sleep(1)

    raw_input("gripped")

    moveAndPour(planner, end_pos)

    raw_input("poured")

    executePositions(planner, [position + np.array([0,0, 0.02])], [orientation])
    grip.open()

    removeCup(planner,"cup2")




def testPouring():
    global thetas
    planner = PathPlanner("right_arm")
    pour_here = np.array([1.114, -0.198, 0.135])

    moveAndPour(planner, pour_here)

def printPosition():
    listener = tf.TransformListener()

    while not rospy.is_shutdown():
        l_tip = listener.lookupTransform("r_gripper_l_finger", "base", rospy.Time(0))[0]
        r_tip = listener.lookupTransform("r_gripper_r_finger", "base", rospy.Time(0))[0]

        pos = (np.array(l_tip) + np.array(r_tip) ) / 2

        print(pos)
        ros.sleep(0.1)













if __name__ == '__main__':
    global move_group
    global right_arm
    rospy.init_node('moveit_node3')
    move_group = moveit_commander.MoveGroupCommander(group_name)
    right_arm = Limb("right")
    main()
    #testPouring()
    #printPosition()

