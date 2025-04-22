import numpy as np
from basler_camera import BaslerCamera
from camera_aruco_node import SingleMarkerDetector
from detect_holes import CameraArUcoNode
import cv2
import argparse

import config


if config.ROBOT_NAME == "crs93":
    from ctu_crs import CRS93
    robot = CRS93() 
elif config.ROBOT_NAME == "crs97":
    from ctu_crs import CRS97
    robot = CRS97()

def invert_4x4(T: np.ndarray) -> np.ndarray:
    """
    Efficiently inverts a 4x4 SE(3) transformation matrix.
    """
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=T.dtype)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv

def get_marker_pose():
    """
    Get the pose of the ArUco marker in the camera frame.
    Returns a 4x4 homogeneous transform T_marker2cam, or None if not detected.
    """
    camera = BaslerCamera()
    camera.connect_by_name(config.CAMERA_NAME)
    camera.open()
    camera.set_parameters()
    
    # Take one image from the camera
    img = camera.grab_image()
    # aruco_detector = SingleMarkerDetector()
    # rvec, tvec = aruco_detector.get_rvec_tvec_of_first_marker(img)
    
    # Part 0: Initialize camera and CameraArUcoNode
    parser = argparse.ArgumentParser(description='ArUco marker detection with start/target board specification')
    parser.add_argument('-start_board', type=int, required=False, 
                       help='ID of the marker on the start board (with cubes). '
                            'If not specified, only one board will be processed.')
    args = parser.parse_args()

    start_board_csv = "csvs/positions_plate_start.csv"
    target_board_csv = "csvs/positions_plate_target.csv"

    aruco_node = CameraArUcoNode(start_board_csv, target_board_csv)
    holes_dict = aruco_node.camera_callback(img)

    camera.close()
    
    # if rvec is None or tvec is None:
    #     return None
    
    # # Convert rvec/tvec to transformation matrix
    # R, _ = cv2.Rodrigues(rvec)
    # T_marker2cam = np.eye(4)
    # T_marker2cam[:3, :3] = R
    # T_marker2cam[:3, 3] = tvec.flatten()

    T_marker2cam = holes_dict['start'][0]
    
    return T_marker2cam

def main():
    # Initialize robot
    if config.ROBOT_INITIALIZE:
        robot.initialize()
    else:
        robot.initialize(home=False)
        robot.soft_home()
    np.atan2 = np.arctan2
    
    q_home = robot.get_q()
    robot.move_to_q(q_home + np.deg2rad([45.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    robot.wait_for_motion_stop()

    if config.ROBOT_NAME == "crs93":
        calibration_name = "crs93"
    elif config.ROBOT_NAME == "crs97":
        calibration_name = "crs97"
    T_gm = np.load(f"robot_matrices/calibration_gm_{calibration_name}.npy")
    T_rc = np.load(f"robot_matrices/calibration_rc_{calibration_name}.npy")
    print("Original T_rc =\n", T_rc)
    # # We have a nominal "start" joint pose
    # q_start = np.load('calib_poses/q_calib_start.npy')
    # robot.move_to_q(q_start)
    # robot.wait_for_motion_stop()

    # Get marker pose in camera frame
    T_cm = get_marker_pose()
    if T_cm is None:
        print("No marker detected!")
        return

    marker_rm = T_rc @ T_cm

    # Transform the marker pose to the robot base frame
    print("T_cm =\n", T_cm)
    print("T_rc =\n", T_rc)
    print("marker_rm =\n", marker_rm)
    for i in range(len(robot.q_min)):
        print(f"Joint {i} limits: {robot.q_min[i]}, {robot.q_max[i]}")
    
    # Adjust the marker frame to align with robot's preferred orientation
    # Create a rotation matrix that aligns Z-axis up
    R_align = np.array([
        [1,  0,  0],
        [ 0, -1,  0],
        [ 0,  0,  -1]
    ])
    
    T_rm_aligned = marker_rm.copy()
    T_rm_aligned[:3, :3] = marker_rm[:3, :3] @ R_align

    # Add offset from aruco marker by 10 cm
    T_rm_aligned[2, 3] = marker_rm[2, 3] + 0.1


    solutions = robot.ik(T_rm_aligned)
    
    if not solutions:
        print("No IK solution found!")
        return
    else:
        print("IK solution found!")
    
    for i in range(len(solutions)):
        print(f"Solution {i}: {solutions[i]}")

    # Find the first solution within joint limits
    valid_solution = None
    for solution in solutions:

        # Works well in some cases, but not always
        # solution[1] += np.pi
        # solution[0] += np.pi
        if robot.in_limits(solution):
            valid_solution = solution
            break
    
    if valid_solution is None:
        print("No valid solution within joint limits!")
        return

    print("Valid solution found!: ", valid_solution)

    print(f"Robot fk(valid_solution) = {robot.fk(valid_solution)}")

    # Move robot to the marker
    print("Moving to marker...")
    robot.move_to_q(valid_solution)
    robot.wait_for_motion_stop()
    print("Reached marker position!")

if __name__ == "__main__":
    main()
    
    # Uncomment to do other tests if needed:
    # robot = CRS97()
    # robot.initialize()
    # np.atan2 = np.arctan2
    # q = np.load('calib_poses/q9.npy')
    # robot.move_to_q(q)
    # robot.wait_for_motion_stop()
    # print("Reached marker position!")
