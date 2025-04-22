import cv2
import cv2.aruco as aruco
import numpy as np
import csv
import argparse

# Our Basler camera interface
from basler_camera import BaslerCamera
from detect_holes import CameraArUcoNode
import config

if config.ROBOT_NAME == "crs93":
    from ctu_crs import CRS93
    robot = CRS93() 
elif config.ROBOT_NAME == "crs97":
    from ctu_crs import CRS97
    robot = CRS97()

OFFSET_ABOVE_CUBE=0.1
OFFSET_GRAB_CUBE=0.035
# OFFSET_RELEASE_CUBE=0.04
OFFSET_RELEASE_CUBE=0.06
TASK_A_SHIFT_DEG=2.0 

def align_robot_to_marker(T_rh):
    """
    Align the robot's Z-axis to be opposite to the marker's Z-axis.
    """
    R_align = np.array([
        [1,  0,  0],
        [ 0, -1,  0],
        [ 0,  0,  -1]
    ])
    
    T_rh_aligned = T_rh.copy()
    T_rh_aligned[:3, :3] = T_rh[:3, :3] @ R_align

    return T_rh_aligned

def get_holes_dict(aruco_node:CameraArUcoNode, offset_version: bool = False):
    camera = BaslerCamera()
    camera.connect_by_name(config.CAMERA_NAME)
    camera.open()
    camera.set_parameters()

    holes_dict = None
    if offset_version:
        holes_dict = aruco_node.camera_callback(camera, offset_version=True)
    else:
        holes_dict = aruco_node.camera_callback(camera)
    if holes_dict is None:
        print("No holes detected")
        camera.close()
        return None
    camera.close()
    return holes_dict
    
'''
    OLD get_best_solution function
'''
def get_best_solution(solutions: list, robot):
    best_sol = None
    min_diff = float('inf')
    for sol in solutions:
        gripper_angle = sol[-1] % (2*np.pi)  # Normalize to [0, 2pi]
        if gripper_angle > np.pi:
            gripper_angle -= 2*np.pi  # Convert to [-pi, pi]
        
        if abs(gripper_angle) <= np.pi/2:
            curr_diff = np.sum(np.abs(sol - robot.get_q()))
            if curr_diff < min_diff:
                min_diff = curr_diff
                if robot.in_limits(sol): best_sol = sol

        elif best_sol is None:  # If no solution in [-pi/2, pi/2] found yet
            print("get_best_solution function: elif best_sol is None case (that is for the test purpose)")
            # Try adding/subtracting 2pi to get in range
            adjusted_angle = gripper_angle + 2*np.pi if gripper_angle < -np.pi/2 else gripper_angle - 2*np.pi
            if abs(adjusted_angle) <= np.pi/2:
                sol[-1] = adjusted_angle
                curr_diff = np.sum(np.abs(sol - robot.get_q()))
                if curr_diff < min_diff:
                    min_diff = curr_diff
                    if robot.in_limits(sol): best_sol = sol

    return best_sol

def q_rotate_kloub_6(robot, q):
    
    q_left = q.copy()
    q_right = q.copy()
    q_right[5] = q[5] + np.deg2rad(90)
    q_left[5] = q[5] - np.deg2rad(90)
    if robot.in_limits(q_right):
        return q_right
    elif robot.in_limits(q_left):
        return q_left
    else:
        print("Robot is not in limits")
        return q
    
def rotate_kloub_6(robot):
    q = robot.get_q()
    q_rotated = q_rotate_kloub_6(robot, q)
    robot.move_to_q(q_rotated)
    robot.wait_for_motion_stop()

def grab_cube(robot):
    q = robot.get_q()
    q_left = q.copy()
    q_right = q.copy()
    q_right[5] = q[5] + np.deg2rad(90)
    q_left[5] = q[5] - np.deg2rad(90)

    robot.gripper.control_position(-1000)
    robot.gripper.wait_for_motion_stop()

    robot.gripper.control_position(1000)
    robot.gripper.wait_for_motion_stop()
    
    if robot.in_limits(q_right):
        robot.move_to_q(q_right)
        robot.wait_for_motion_stop()
    elif robot.in_limits(q_left):
        robot.move_to_q(q_left)
        robot.wait_for_motion_stop()
    else:
        print("Robot is not in limits")


    robot.gripper.control_position(-1000)
    robot.gripper.wait_for_motion_stop()

        


def main():
    """
    Example usage with optional command line arguments for start board selection.
    Runs in parts, waiting for user input between steps.
    """
    
    aruco_node = CameraArUcoNode(config.START_BOARD_CSV, config.TARGET_BOARD_CSV)
    
    #Part 1: Initialize robot and move away
    print("\nPart 1: Initialize robot and move away")
    
    if config.ROBOT_INITIALIZE:
        robot.initialize()
    else:
        robot.initialize(home=False)
        robot.soft_home()
    np.atan2 = np.arctan2

    robot.set_speed(robot._max_speed_irc256_per_ms)
    
    robot.gripper.control_position(1000)
    robot.gripper.wait_for_motion_stop()

    q_task_start = np.load(config.ROBOT_INITIAL_POSE)
    T_task_start = robot.fk(q_task_start)

    q_home = robot.get_q()
    q_shift_home = q_home + np.deg2rad([45.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    robot.move_to_q(q_shift_home)
    robot.wait_for_motion_stop()
    
    # Part 2: Detect holes
    print("\nPart 2: Detecting holes")
    
    holes_dict = get_holes_dict(aruco_node)
    if holes_dict is None:
        print("Error: holes_dict is None")
        return 0
    

    holes_dict_offset_version = get_holes_dict(aruco_node, offset_version=True)
    if holes_dict_offset_version is None:
        print("Error: holes_dict_offset_version is None")
        return 0
    
    
    # Part 2: Move to cube position with offset

    for i in range(len(holes_dict['start'])):
        
        print("\nPart 2: Moving to cube position")
        print("Press Enter...")

        '''
            Go to the start offset posion 
        '''

        robot.move_to_q(q_task_start)
        robot.wait_for_motion_stop()

        T_ch_start = holes_dict['start'][i]
        T_rh_start = config.T_RC @ T_ch_start
        T_rh_start = align_robot_to_marker(T_rh_start)
        # Get IK solutions and find best one
        # T_offset_start_0 = T_ch_start.copy()
        # T_offset_start_0[2, 3]  = T_ch_start[2, 3] - OFFSET_ABOVE_CUBE
        # T_rh_offset_start = config.T_RC @ T_offset_start_0
        # T_rh_offset_start = align_robot_to_marker(T_rh_offset_start)

        T_offset_start_0 = holes_dict_offset_version['start'][i]
        T_rh_offset_start = config.T_RC @ T_offset_start_0
        T_rh_offset_start = align_robot_to_marker(T_rh_offset_start)

        solutions_offset_start = robot.ik(T_rh_offset_start)
        if not solutions_offset_start:
            print("No IK solutions_offset_start found")
            continue
        
        best_sol_offset_start = get_best_solution(solutions_offset_start, robot)
    
        if best_sol_offset_start is None:
            print("No best_sol_offset_start found within gripper angle constraints")
            continue

        robot.move_to_q(best_sol_offset_start)
        robot.wait_for_motion_stop()

        # rotate_kloub_6(robot)
        
        print("\nReached offset position. Press Enter to move down to cube...")
        
        '''
            Go to the start cube posion 
        '''
        # Slow down and move down to cube
        robot.set_speed(robot._min_speed_irc256_per_ms)  # min speed

        T_cube_start = T_rh_start.copy()
        T_cube_start[2, 3]  = T_rh_start[2, 3] + OFFSET_GRAB_CUBE
        solutions_cube_start = robot.ik(T_cube_start)
        if not solutions_cube_start:
            print("No IK solutions_cube_start found")
            continue
        
        # Find solution with gripper angle between -pi/2 and pi/2
        best_sol_cube_start = get_best_solution(solutions_cube_start, robot)
    
        if best_sol_cube_start is None:
            print("No best_sol_cube_start found within gripper angle constraints")
            continue

        # best_sol_cube_start = q_rotate_kloub_6(robot, best_sol_cube_start)

        robot.move_to_q(best_sol_cube_start)
        robot.wait_for_motion_stop()
        
        print("\nReached cube position. Closing gripper...")
        
        robot.gripper.control_position(-1000)
        robot.gripper.wait_for_motion_stop()
        # grab_cube(robot)

        print("\nGripper closed. Move up...")

        robot.move_to_q(best_sol_offset_start)
        robot.wait_for_motion_stop()
        robot.set_speed(robot._max_speed_irc256_per_ms)

        # Mpve to start position
        robot.move_to_q(q_task_start)
        robot.wait_for_motion_stop()

        if not config.SINGLE_BOARD_MODE:
        
            print("\nPart 3: Moving to the next board")

            '''
                Go to the target offset posion 
            '''
            
            T_ch_target = holes_dict['target'][i]

            # Transform hole pose to robot frame
            T_rh_target = config.T_RC @ T_ch_target
            T_rh_target = align_robot_to_marker(T_rh_target)

            # Get IK solutions and find best one

            T_rh_offset_target = T_rh_target.copy()
            T_rh_offset_target[2, 3]  = T_rh_target[2, 3] + OFFSET_ABOVE_CUBE

            # T_offset_target_0 = holes_dict_offset_version['target'][i]
            # T_rh_offset_target = config.T_RC @ T_offset_target_0
            # T_rh_offset_target = align_robot_to_marker(T_rh_offset_target)

            solutions_offset_target = robot.ik(T_rh_offset_target)
            if not solutions_offset_target:
                print("No IK solutions_offset_target found")
                continue
            
            best_sol_offset_target = get_best_solution(solutions_offset_target, robot)
            
            if best_sol_offset_target is None:
                print("No best_sol_offset_target found within gripper angle constraints")
                continue

            robot.move_to_q(best_sol_offset_target)
            robot.wait_for_motion_stop()
            
            print("\nReached offset position. Moving down to cube...")
            
            '''
                Go to the target cube posion 
            '''

            # Slow down and move down to cube
            robot.set_speed(robot._min_speed_irc256_per_ms)  # min speed


            T_cube_target = T_rh_target.copy()
            T_cube_target[2, 3]  = T_rh_target[2, 3] + OFFSET_RELEASE_CUBE
            solutions_cube_target = robot.ik(T_cube_target)
            if not solutions_cube_target:
                print("No IK solutions_cube_target found")
                continue
            best_sol_cube_target = get_best_solution(solutions_cube_target, robot)
            
            if best_sol_cube_target is None:
                print("No best_sol_cube_target found within gripper angle constraints")
                break

            robot.move_to_q(best_sol_cube_target)
            robot.wait_for_motion_stop()
            
            print("\nReached cube position. Opening gripper...")
            
            robot.gripper.control_position(1000)
            robot.gripper.wait_for_motion_stop()

            print("\nCube is released. Moving up...")

            robot.move_to_q(best_sol_offset_target)
            robot.wait_for_motion_stop()

            robot.set_speed(robot._max_speed_irc256_per_ms)

            print("\nGoing to the home position...")
        
        else:
            print("\nPart 3: Moving to the side")

            '''
                Go to the side of the start board 
            '''

            q_shift = best_sol_offset_start
            q_shift[0]=q_shift_home[0]+i*np.deg2rad(TASK_A_SHIFT_DEG)
            robot.move_to_q(q_shift)
            robot.wait_for_motion_stop()

            print("\n Cube is moved to the side. Press Enter to move down the cube...")
            
            robot.set_speed(robot._min_speed_irc256_per_ms)  # min speed

            T_shift = robot.fk(q_shift)
            T_shift[2, 3]  = T_shift[2, 3] - 0.5 * OFFSET_ABOVE_CUBE
            solutions_shift = robot.ik(T_shift)
            if not solutions_shift:
                print("No IK solutions_shift found")
                continue
            best_sol_shift = get_best_solution(solutions_shift, robot)
            
            if best_sol_shift is None:
                print("No best_sol_shift found within gripper angle constraints")
                break

            robot.move_to_q(best_sol_shift)
            robot.wait_for_motion_stop()

            print("\nReleasing the cube...")

            robot.gripper.control_position(1000)
            robot.gripper.wait_for_motion_stop()

            print("\nMoving up...")

            robot.move_to_q(q_shift)
            robot.wait_for_motion_stop()
            
            robot.set_speed(robot._max_speed_irc256_per_ms)

            print("\nGoing to the home position...")

        robot.move_to_q(q_shift_home)
        robot.wait_for_motion_stop()

        print("Get updated holes_dict")
        holes_dict = get_holes_dict(aruco_node)
        holes_dict_offset_version = get_holes_dict(aruco_node, offset_version=True)
        if holes_dict is None:
            print("Error: holes_dict is None")
            return 0

        # break  # Process only first hole for now

    # Move to safe position
    robot.move_to_q(q_home)
    robot.wait_for_motion_stop()

    print("Task is done")
    robot.reset_motors()

if __name__ == "__main__":
    main()