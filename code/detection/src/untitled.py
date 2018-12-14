#!/usr/bin/env python
import rospy
import numpy as np
import sys
from numpy import linalg
from baxter_interface import gripper as robot_gripper
from moveit_msgs.msg import OrientationConstraint
from geometry_msgs.msg import PoseStamped
from path_planner import PathPlanner
from baxter_interface import Limb
from cup_detection import find_cups


def get_xy(u, v, z, K, R, t):
    pixels = np.array([u, v, 1])
    homog = K @ np.hstack((R, t.reshape((-1,1))))
    A = homog[:, [0,1,3]]
    a = np.linalg.inv(A) @ pixels
    b = np.linalg.inv(A) @ homog[:,2] * z
    s = (1 + b[2]) / a[2]
    xy = s*a - b
    return xy


def main():

    planner = PathPlanner("left_arm")

    Kp = 0.1 * np.array([0.3, 2, 1, 1.5, 2, 2, 3]) # Stolen from 106B Students
    Kd = 0.01 * np.array([2, 1, 2, 0.5, 0.5, 0.5, 0.5]) # Stolen from 106B Students
    Ki = 0.01 * np.array([1, 1, 1, 1, 1, 1, 1]) # Untuned
    Kw = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]) # Untuned

    # #Create a path constraint for the arm
    orien_const = OrientationConstraint()
    orien_const.link_name = "left_gripper";
    orien_const.header.frame_id = "base";
    orien_const.orientation.y = -1.0;
    orien_const.absolute_x_axis_tolerance = 0.1;
    orien_const.absolute_y_axis_tolerance = 0.1;
    orien_const.absolute_z_axis_tolerance = 0.1;
    orien_const.weight = 1.0;

    filename = "test_cup.jpg"
    table_height = 0

    # get positions of cups
    cups = find_cups(filename)
    start_cup = cups[0]
    goal_cup = cups[1]
    


    cup_pos = [0.756, 0.300, -0.162] # position of cup
    goal_pos = [0.756, 0.600, -0.162] # position of goal cup
    init_pos = [cup_pos[0] - .2, cup_pos[1], cup_pos[2] + .05] # initial position
    start_pos = [init_pos[0] + .2, init_pos[1], init_pos[2]] # start position, pick up cup
    up_pos = [goal_pos[0], goal_pos[1] - .05, goal_pos[2] + .1] # up position, ready to tilt
    end_pos = start_pos # end position, put down cup
    final_pos = init_pos # final position, away from cup


    # add table obstacle
    table_size = np.array([.4, .8, .1])
    table_pose = PoseStamped()
    table_pose.header.frame_id = "base"

    #x, y, and z position
    table_pose.pose.position.x = 0.5
    table_pose.pose.position.y = 0.0
    table_pose.pose.position.z = 0.0

    planner.add_box_obstacle(table_size, 'table', table_pose)


    # add goal cup obstacle
    goal_cup_size = np.array([.1, .1, .2])
    goal_cup_pose = PoseStamped()
    goal_cup_pose.header.frame_id = "base"

    #x, y, and z position
    goal_cup_pose.pose.position.x = goal_pos[0]
    goal_cup_pose.pose.position.y = goal_pos[1]
    goal_cup_pose.pose.position.z = goal_pos[2]

    planner.add_box_obstacle(goal_cup_size, 'goal cup', goal_cup_pose)


    # add first cup obstacle
    cup_size = np.array([.1, .1, .2])
    cup_pose = PoseStamped()
    cup_pose.header.frame_id = "base"

    #x, y, and z position
    cup_pose.pose.position.x = cup_pos[0]
    cup_pose.pose.position.y = cup_pos[1]
    cup_pose.pose.position.z = cup_pos[2]

    planner.add_box_obstacle(cup_size, 'cup', cup_pose)


    while not rospy.is_shutdown():

        left_gripper = robot_gripper.Gripper('left')

        print('Calibrating...')
        left_gripper.calibrate()

        pose = PoseStamped()
        pose.header.frame_id = "base"

        #Orientation as a quaternion
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = -1.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 0.0

        while not rospy.is_shutdown():
            try:
                #x, y, and z position
                pose.pose.position.x = init_pos[0]
                pose.pose.position.y = init_pos[1]
                pose.pose.position.z = init_pos[2]

                plan = planner.plan_to_pose(pose, [orien_const])

                raw_input("Press <Enter> to move the left arm to initial pose: ")
                if not planner.execute_plan(plan):
                    raise Exception("Execution failed")
            except Exception as e:
                print e
            else:
                break


        # grab cup
        # need to make sure it doesn't knock it over 
        planner.remove_obstacle('cup')
        while not rospy.is_shutdown():
            try:
                #x, y, and z position
                pose.pose.position.x = start_pos[0]
                pose.pose.position.y = start_pos[1]
                pose.pose.position.z = start_pos[2]

                plan = planner.plan_to_pose(pose, [orien_const])

                raw_input("Press <Enter> to move the left arm to grab the cup: ")
                if not planner.execute_plan(plan):
                    raise Exception("Execution failed")

                print('Closing...')
                left_gripper.close()
                rospy.sleep(2)

            except Exception as e:
                print e
            else:
                break


        # position to pour
        while not rospy.is_shutdown():
            try:

                #x, y, and z position
                pose.pose.position.x = up_pos[0]
                pose.pose.position.y = up_pos[1]
                pose.pose.position.z = up_pos[2]

                plan = planner.plan_to_pose(pose, [orien_const])

                raw_input("Press <Enter> to move the left arm to above the goal cup: ")
                if not planner.execute_plan(plan):
                    raise Exception("Execution failed")
            except Exception as e:
                print e
            else:
                break


        # pouring
        raw_input("Press <Enter> to move the left arm to begin pouring: ")
        for degree in range(180):
            while not rospy.is_shutdown():
                try:

                    #Orientation as a quaternion
                    pose.pose.orientation.x = 0.0
                    pose.pose.orientation.y = -1.0
                    pose.pose.orientation.z = 0.0
                    pose.pose.orientation.w = 0.0

                    plan = planner.plan_to_pose(pose, [])

                    if not planner.execute_plan(plan):
                        raise Exception("Execution failed")
                except Exception as e:
                    print e
                else:
                    break


        # move cup away from goal on table
        while not rospy.is_shutdown():
            try:

                #x, y, and z position
                pose.pose.position.x = end_pos[0]
                pose.pose.position.y = end_pos[1]
                pose.pose.position.z = end_pos[2]

                plan = planner.plan_to_pose(pose, [orien_const])

                raw_input("Press <Enter> to move the left arm to away from the goal cup: ")
                if not planner.execute_plan(plan):
                    raise Exception("Execution failed")
            except Exception as e:
                print e
            else:
                break
 

        # let go of cup on table
        # need to make sure to not to hit cup
        while not rospy.is_shutdown():
            try:

                #x, y, and z position
                pose.pose.position.x = final_pos[0]
                pose.pose.position.y = final_pos[1]
                pose.pose.position.z = final_pos[2]

                plan = planner.plan_to_pose(pose, [orien_const])

                raw_input("Press <Enter> to move the left arm to let go of the cup: ")

                print('Opening...')
                left_gripper.open()
                rospy.sleep(1.0)

                if not planner.execute_plan(plan):
                    raise Exception("Execution failed")
            except Exception as e:
                print e
            else:
                break

        # get new cup position
        new_cup_pos = []

        # readd the cup obstacle
        cup_pose.pose.position.x = new_cup_pos[0]
        cup_pose.pose.position.y = new_cup_pos[1]
        cup_pose.pose.position.z = new_cup_pos[2]

        planner.add_box_obstacle(cup_size, 'cup', cup_pose)



if __name__ == '__main__':
    rospy.init_node('moveit_node')
    main()


