from ctu_crs import CRS97
import numpy as np
from basler_camera import BaslerCamera
from camera_aruco_node import SingleMarkerDetector
import config
import cv2

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
    aruco_detector = SingleMarkerDetector()
    rvec, tvec = aruco_detector.get_rvec_tvec_of_first_marker(img)
    
    camera.close()
    
    if rvec is None or tvec is None:
        return None
    
    # Convert rvec/tvec to transformation matrix
    R, _ = cv2.Rodrigues(rvec)
    T_marker2cam = np.eye(4)
    T_marker2cam[:3, :3] = R
    T_marker2cam[:3, 3] = tvec.flatten()
    
    return T_marker2cam

def main():
    # Initialize robot
    robot = CRS97()
    # robot.initialize()
    np.atan2 = np.arctan2
    
    # === CHANGED: load the revised calibration transforms
    # According to the updated calibration code, we now have:
    #   "calibration_base2world.npy"  => T_base2world
    #   "calibration_gripper2cam.npy" => T_gripper2cam
    T_base2world = np.load("robot_matrices/calibration_base2world.npy")
    T_gripper2cam = np.load("robot_matrices/calibration_gripper2cam.npy")

    # We have a nominal "start" joint pose
    q_start = np.load('calib_poses/q_calib_start.npy')
    # robot.move_to_q(q_start)
    # robot.wait_for_motion_stop()

    # Get marker pose in camera frame
    T_marker2cam = get_marker_pose()
    if T_marker2cam is None:
        print("No marker detected!")
        return

    # For demonstration, we look at T_marker2base by combining
    #   T_marker2base = T_base2marker^-1
    # but T_base2marker == T_base2world in our notation (marker=world).
    # So:
    T_marker2base = invert_4x4(T_base2world)  # base←marker

    # Print the translation part
    T_marker2cam_t = T_marker2cam[:3, 3].flatten()
    print("Marker to Camera translation part:\n", T_marker2cam_t)

    marker_position = T_marker2base @ np.append(T_marker2cam_t, 1)
    print("Marker position:\n", marker_position)

    # Example usage: we want the robot end-effector to go exactly to the marker’s origin.
    # That means a pose in the robot base frame is T_marker2base.  Then we do IK:
    # solutions = robot.ik(marker_position)

    solutions = None
    
    if not solutions:
        print("No IK solution found!")
        return
    else:
        print("IK solution found!")
    
    # Find the first solution within joint limits
    valid_solution = None
    for solution in solutions:
        if robot.in_limits(solution):
            valid_solution = solution
            break
    
    if valid_solution is None:
        print("No valid solution within joint limits!")
        return
        
    # # Move robot to the marker
    # print("Moving to marker...")
    # robot.move_to_q(valid_solution)
    # robot.wait_for_motion_stop()
    # print("Reached marker position!")

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
