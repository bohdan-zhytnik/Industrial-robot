import cv2
import cv2.aruco as aruco
import numpy as np
import csv
import argparse

# Our Basler camera interface
from basler_camera import BaslerCamera
from detect_holes import CameraArUcoNode
from camera_aruco_node import SingleMarkerDetector
import config
import main
KLOB_1_SHIFT_DEG = 5.0
KLOB_4_SHIFT_DEG = 10.0 
max_num_of_poses_4_kloub = 10
#for 10 deg in kloub 4 we should 1 deg kloub 5 and -2 kloub 6
KLOB_5_SHIFT_DEG = 20.0  
KLOB_6_SHIFT_DEG = -30.0 
UP_COEF = 0.05
SIDE_Y_OFFSET = 0.02
SIDE_X_OFFSET = 0.05


if config.ROBOT_NAME == "crs93":
    from ctu_crs import CRS93
    robot = CRS93() 
elif config.ROBOT_NAME == "crs97":
    from ctu_crs import CRS97
    robot = CRS97()

'''
    New get_best_solution function
'''
def get_best_solution_up(solutions: list, robot):
    top_five = []  # List to store 5 best solutions
    min_diff = float('inf')
    
    for sol in solutions:
        # Skip solutions where 4th element is positive or 1st element is negative
        if sol[0] > 0:
            continue
            
        gripper_angle = sol[-1] % (2*np.pi)  # Normalize to [0, 2pi]
        if gripper_angle > np.pi:
            gripper_angle -= 2*np.pi  # Convert to [-pi, pi]
        
        if abs(gripper_angle) <= np.pi/2:
            curr_diff = np.sum(np.abs(sol - robot.get_q())[:4])
            if curr_diff < min_diff:
                min_diff = curr_diff
                if robot.in_limits(sol):
                    # Only add if conditions are met
                    if len(top_five) < 5:
                        top_five.append((sol, curr_diff))
                    else:
                        # Replace worst solution if current is better
                        worst_diff = max(s[1] for s in top_five)
                        if curr_diff < worst_diff:
                            # Find and replace the worst solution
                            worst_idx = max(range(len(top_five)), key=lambda i: top_five[i][1])
                            top_five[worst_idx] = (sol, curr_diff)

        elif not top_five:  # If no solution in [-pi/2, pi/2] found yet
            print("get_best_solution function: elif best_sol is None case (that is for the test purpose)")
            # Try adding/subtracting 2pi to get in range
            adjusted_angle = gripper_angle + 2*np.pi if gripper_angle < -np.pi/2 else gripper_angle - 2*np.pi
            if abs(adjusted_angle) <= np.pi/2:
                sol_adjusted = sol.copy()
                sol_adjusted[-1] = adjusted_angle
                # Only proceed if both conditions are met
                if  sol_adjusted[0] <= 0:
                # if True:
                    curr_diff = np.sum(np.abs(sol_adjusted - robot.get_q())[:4])
                    if curr_diff < min_diff:
                        min_diff = curr_diff
                        if robot.in_limits(sol_adjusted):
                            top_five.append((sol_adjusted, curr_diff))

    # Sort solutions by the 4th element (index 3) of the solution array
    if top_five:
        top_five.sort(key=lambda x: x[0][3])  # Sort by 4th element of solution
        print("\nTop 5 solutions (sorted by 4th element):")
        for i, (sol, diff) in enumerate(top_five, 1):
            print(f"{i}. Solution: {sol}, Difference: {diff}")
        return top_five[0][0]  # Return solution with lowest 4th element
    
    return None


def get_best_solution(solutions: list, robot):
    best_sol = None
    min_diff = float('inf')
    for sol in solutions:
        gripper_angle = sol[-1] % (2*np.pi)  # Normalize to [0, 2pi]
        if gripper_angle > np.pi:
            gripper_angle -= 2*np.pi  # Convert to [-pi, pi]
        
        if abs(gripper_angle) <= np.pi/2:
            curr_diff = np.sum(np.abs(sol - robot.get_q())[:4])
            if curr_diff < min_diff:
                min_diff = curr_diff
                if robot.in_limits(sol): best_sol = sol

        elif best_sol is None:  # If no solution in [-pi/2, pi/2] found yet
            print("get_best_solution function: elif best_sol is None case (that is for the test purpose)")
            # Try adding/subtracting 2pi to get in range
            adjusted_angle = gripper_angle + 2*np.pi if gripper_angle < -np.pi/2 else gripper_angle - 2*np.pi
            if abs(adjusted_angle) <= np.pi/2:
                sol[-1] = adjusted_angle
                curr_diff = np.sum(np.abs(sol - robot.get_q())[:4])
                if curr_diff < min_diff:
                    min_diff = curr_diff
                    if robot.in_limits(sol): best_sol = sol

    return best_sol

# def make_rotation_z_axis(T_current):
#     T_current_transpose = T_current[:3,:3].T
#     T_rotate = np.dot(R,T_current_transpose)
    


def move_to_the_side(T_current, side_offset, aixs):  #side_offset '-' is left, '+' is right ; aixs = 0 is x, aixs = 1 is y, aixs = 2 is z
    # T_new = T_current.copy()
    T_current[aixs,3] -= side_offset
    q_sol_list = robot.ik(T_current)
    q_best_5 = get_best_solution(q_sol_list,robot)
    if q_best_5 is not None:
        robot.move_to_q(q_best_5)
        robot.wait_for_motion_stop()
        return True
    else:
        print("move_to_the_side function: q_best_5 is None")
        return False

    # return T_new

def additional_moves(robot, counter, pose_number,floor_number):
    q = robot.get_q()
    T_current = robot.fk(q)
    q_old = q.copy()
    i = 1
    if counter % 10 == 0 and counter != 0:
        for _ in range(2):
            # q = q_old + np.deg2rad([0.0, 0.0, 0.0, 0.0, 0.0, i*KLOB_6_SHIFT_DEG])
            robot.soft_home()
            robot.wait_for_motion_stop()
            pose_number += 1 
            filename = f"calib_poses/crs93_many_poses/q_{config.ROBOT_NAME}_pos{pose_number}.npy"
            np.save(filename, robot.q_home)
            print(f"pose {pose_number} ArUco marker is visible") 
            if move_to_the_side(T_current, i*SIDE_X_OFFSET, 0):
                if is_aruco_visible():
                    pose_number += 1 
                    save_pose(q, pose_number)
                else:
                    print(f"pose {pose_number} ArUco marker is not visible")
            i = -1 * i
        robot.soft_home()
        robot.wait_for_motion_stop()
        pose_number += 1 
        filename = f"calib_poses/crs93_many_poses/q_{config.ROBOT_NAME}_pos{pose_number}.npy"
        np.save(filename, robot.q_home)
        print(f"pose {pose_number} ArUco marker is visible") 
        robot.move_to_q(q_old)
        robot.wait_for_motion_stop()
            

    if counter % 3 == 0 and counter != 0:
        if floor_number != 0:
            for _ in range(2):
                q = q_old + np.deg2rad([0.0, 0.0, 0.0, 0.0, i*KLOB_5_SHIFT_DEG, 0.0])
                if robot.in_limits(q):
                    robot.move_to_q(q)
                    robot.wait_for_motion_stop()
                    if is_aruco_visible():
                        pose_number += 1 
                        save_pose(q, pose_number)
                    else:
                        print(f"pose {pose_number} ArUco marker is not visible")
                    # list_of_unvisible_marker_pose_numbers.append(q)
                    # unvisible_marker_pose_number += 1
                    robot.move_to_q(q_old)
                    robot.wait_for_motion_stop()
                i = -1 * i

        for _ in range(2):
            q = q_old + np.deg2rad([0.0, 0.0, 0.0, 0.0, 0.0, i*KLOB_6_SHIFT_DEG])
            if robot.in_limits(q):
                robot.move_to_q(q)
                robot.wait_for_motion_stop()
                if is_aruco_visible():
                    pose_number += 1 
                    save_pose(q, pose_number)
                else:
                    print(f"pose {pose_number} ArUco marker is not visible")
                    # list_of_unvisible_marker_pose_numbers.append(q)
                    # unvisible_marker_pose_number += 1
                robot.move_to_q(q_old)
                robot.wait_for_motion_stop()
            i = -1 * i   
        for _ in range(2):
            q = q_old + np.deg2rad([i*5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            if robot.in_limits(q):
                robot.move_to_q(q)
                robot.wait_for_motion_stop()
                if is_aruco_visible():
                    pose_number += 1 
                    save_pose(q, pose_number)
                else:
                    print(f"pose {pose_number} ArUco marker is not visible")
                    # list_of_unvisible_marker_pose_numbers.append(q)
                    # unvisible_marker_pose_number += 1
                robot.move_to_q(q_old)
                robot.wait_for_motion_stop()
            i = -1 * i
            
    return pose_number

def is_aruco_visible():
    camera: BaslerCamera = BaslerCamera()

    camera.connect_by_name(config.CAMERA_NAME)
    camera.open()
    camera.set_parameters()



    camera_aruco_node = SingleMarkerDetector()

    img = camera.grab_image()

    rvec, tvec = camera_aruco_node.get_rvec_tvec_of_first_marker(img)
    if rvec is not None:
        return True
    else:
        return False
    
def save_pose(q_pose, pose_number):
    if is_aruco_visible():
        filename = f"calib_poses/crs93_many_poses/q_{config.ROBOT_NAME}_pos{pose_number}.npy"
        np.save(filename, q_pose)
        print(f"pose {pose_number} ArUco marker is visible") 


def main():
    if config.ROBOT_INITIALIZE:
        robot.initialize()
    else:
        robot.initialize(home=False)
        robot.soft_home()
    np.atan2 = np.arctan2

    q_start = np.load("calib_poses/crs93_many_poses/q_start.npy")
    q_floor_start = q_start.copy()
    T_start = robot.fk(q_start)

    if robot.in_limits(q_start):
        print("q_start is in limits")

        '''
            DELETE LATER line inside 
        '''
        # q_start = q_start + np.deg2rad([-50.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        '''
            DELETE LATER line inside 
        '''

        robot.move_to_q(q_start)
        robot.wait_for_motion_stop()
    else:
        return -1
    
    unvisible_marker_pose_number=0
    list_of_unvisible_marker_pose_numbers = []

    All_aruco_are_visible_1 = True
    pose_number = 0

    floor_number = 0

    q_start1 = robot.get_q()
    side_coef = 1.0

    counter = 0
    while All_aruco_are_visible_1:
        if floor_number != 0:
            
            q = robot.get_q()
            T = robot.fk(q)
            # robot.soft_home()
            if move_to_the_side(T, SIDE_Y_OFFSET, 1):
                if is_aruco_visible():
                    counter += 1
                    pose_number += 1 
                    save_pose(q, pose_number)
                    pose_number = additional_moves(robot, counter, pose_number,floor_number)
                else:
                    floor_number += 1
                    if floor_number == 5:
                        All_aruco_are_visible_1 = False
                        break


                    T_new_start = robot.fk(robot.get_q())
                    robot.soft_home()

                    pose_number += 1 
                    filename = f"calib_poses/crs93_many_poses/q_{config.ROBOT_NAME}_pos{pose_number}.npy"
                    np.save(filename, robot.q_home)
                    print(f"pose {pose_number} ArUco marker is visible") 

                    # T_new_start = robot.fk(robot.get_q())
                    T_floor_up = T_new_start.copy() 
                    T_floor_up[2,3] = T_floor_up[2,3] + floor_number*UP_COEF
                    T_floor_up[2,3] = T_floor_up[0,3] -SIDE_X_OFFSET
                    q_floor_up_list = robot.ik(T_floor_up)
                    best_sol = get_best_solution_up(q_floor_up_list, robot)
                    robot.move_to_q(best_sol)
                    robot.wait_for_motion_stop()
                    q_start1 = best_sol.copy()
                    print(f"q_start1: {q_start1}")
                    
                    q_shit_a_litte = robot.get_q()
                    q_shit_a_litte = q_shit_a_litte + np.deg2rad([60.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    robot.move_to_q(q_shit_a_litte)
                    robot.wait_for_motion_stop()

                    
                    while not is_aruco_visible():
                        q_shit_a_litte = robot.get_q()
                        q_shit_a_litte = q_shit_a_litte + np.deg2rad([side_coef*-3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                        robot.move_to_q(q_shit_a_litte)
                        robot.wait_for_motion_stop()
                        if is_aruco_visible():
                            q_start1 = q_shit_a_litte.copy()
                            print(f"q_start1: {q_start1}")
                # All_aruco_are_visible_1 = False
            else:
                break

            # counter += 1
            

        if floor_number == 0:
            All_aruco_are_visible_2 = True
            q = robot.get_q()
            q_old = q

            # q =q + np.deg2rad([KLOB_1_SHIFT_DEG, 0.0, 0.0, 0.0, 0.0, 0.0])
            q =q + np.deg2rad([0.0, 0.0, 0.0, side_coef*KLOB_4_SHIFT_DEG, side_coef*KLOB_4_SHIFT_DEG/10, -side_coef*KLOB_4_SHIFT_DEG/5])
            
            robot.move_to_q(q)
            robot.wait_for_motion_stop()
            # print("test0")
            if is_aruco_visible():
                # print("test1")
                counter += 1
                # print("test1.1")
                pose_number += 1 
                # print("test1.3")
                save_pose(q, pose_number)
                # print("test1.4")
                pose_number = additional_moves(robot, counter, pose_number,floor_number)

            else:
                print("test2")
                print(f"pose {pose_number} ArUco marker is not visible. First while loop")
                list_of_unvisible_marker_pose_numbers.append(q)
                unvisible_marker_pose_number += 1
                robot.move_to_q(q_start1)
                robot.wait_for_motion_stop()
                q_new_start = q_start1 + np.deg2rad([-side_coef*KLOB_1_SHIFT_DEG, 0.0, 0.0, 0.0, 0.0, 0.0])
                robot.move_to_q(q_new_start)
                robot.wait_for_motion_stop()
                q_start1 = robot.get_q()
                

                if is_aruco_visible():
                    continue
                else:
                    print(f"pose {pose_number} ArUco marker is not visible")
                    list_of_unvisible_marker_pose_numbers.append(q)
                    unvisible_marker_pose_number += 1 
                    All_aruco_are_visible_2 = False
                    floor_number += 1
                    if floor_number == 5:
                        All_aruco_are_visible_1 = False
                        break


                    # TODO: this is not working
                    robot.soft_home()

                    pose_number += 1 
                    filename = f"calib_poses/crs93_many_poses/q_{config.ROBOT_NAME}_pos{pose_number}.npy"
                    np.save(filename, robot.q_home)
                    print(f"pose {pose_number} ArUco marker is visible") 

                    T_new_start = robot.fk(q_new_start)
                    T_floor_up = T_new_start.copy() 
                    T_floor_up[2,3] = T_floor_up[2,3] + floor_number*UP_COEF
                    q_floor_up_list = robot.ik(T_floor_up)
                    best_sol = get_best_solution_up(q_floor_up_list, robot)
                    robot.move_to_q(best_sol)
                    robot.wait_for_motion_stop()
                    q_start1 = best_sol.copy()
                    print(f"q_start1: {q_start1}")
                    
                    q_shit_a_litte = robot.get_q()
                    q_shit_a_litte = q_shit_a_litte + np.deg2rad([60.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    robot.move_to_q(q_shit_a_litte)
                    robot.wait_for_motion_stop()

                    
                    while not is_aruco_visible():
                        q_shit_a_litte = robot.get_q()
                        q_shit_a_litte = q_shit_a_litte + np.deg2rad([side_coef*-3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                        robot.move_to_q(q_shit_a_litte)
                        robot.wait_for_motion_stop()
                        if is_aruco_visible():
                            q_start1 = q_shit_a_litte.copy()
                            print(f"q_start1: {q_start1}")
        # counter += 1
            print("test3")


        
if __name__ == "__main__":
    main()


