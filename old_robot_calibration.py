import numpy as np
import cv2
import config
from camera_aruco_node import SingleMarkerDetector
from basler_camera import BaslerCamera

if config.ROBOT_NAME == "crs93":
    from ctu_crs import CRS93
    robot = CRS93()  # set argument tty_dev=None if you are not connected to robot,
elif config.ROBOT_NAME == "crs97":
    from ctu_crs import CRS97
    robot = CRS97()  # set argument tty_dev=None if you are not connected to robot,

def matrix_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """
    Forms a 4x4 transformation matrix from rotation vector (rvec) and translation vector (tvec).
    Assumes rvec and tvec are provided in OpenCV format (e.g., from solvePnP).
    """
    R, _ = cv2.Rodrigues(rvec)
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = tvec.flatten()
    return M

def rtvec_from_matrix(M):
    # Extra safety check: M[3, :] should be [0, 0, 0, 1]
    assert np.allclose(M[3, :], [0, 0, 0, 1]), f"Bottom row not [0,0,0,1]:\n{M}"
    R = M[:3, :3]
    t = M[:3, 3]
    rvec, _ = cv2.Rodrigues(R)
    return (rvec, t)


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

def example_calibration(robot, q_poses):
    """
    Gather data for Robot-World/Hand-Eye calibration in a static-camera + marker scenario.

    For each robot pose q_i:
      1) Move to q_i, get T_base2gripper via robot.fk(q_i).
      2) Capture an image, detect the marker, get T_marker2cam (via solvePnP).
    Finally call cv2.calibrateRobotWorldHandEye(...).

    Returns the two 4x4 transforms:
       T_base2world  and  T_gripper2cam
    (OpenCV’s doc calls them R_base2world,t_base2world & R_gripper2cam,t_gripper2cam.)
    """
    
    R_world2cam = []     # Will store rotation from marker ("world") to camera
    t_world2cam = []     # Will store translation from marker ("world") to camera
    R_base2gripper = []  # Will store rotation from robot base to gripper
    t_base2gripper = []  # Will store translation from robot base to gripper

    successful_poses = 0
    i = 0

    for q in q_poses:
        # Move robot and wait - no changes needed
        robot.move_to_q(q)
        robot.wait_for_motion_stop()
        print(f"\nProcessing pose {i+1}/{len(q_poses)}")

        # Get base to gripper transformation - NOTE: No longer need to invert
        T_base2gripper = robot.fk(q)
        
        # Camera and ArUco detection - no changes needed
        camera: BaslerCamera = BaslerCamera()
        camera.connect_by_name(config.CAMERA_NAME)
        camera.open()
        camera.set_parameters()
        img = camera.grab_image()
        aruco_node = SingleMarkerDetector()
        rvec, tvec = aruco_node.get_rvec_tvec_of_first_marker(img)

        if rvec is None or tvec is None:
            print(f"Skipping pose {i+1} - marker not detected")
            continue

        try:
            T_marker2cam = matrix_from_rvec_tvec(rvec, tvec)
            # Invert marker to camera to get world to camera
            # T_world2cam = invert_4x4(T_marker2cam)
            T_world2cam = T_marker2cam
        except Exception as e:
            print(f"Error processing pose {i+1}: {e}")
            continue

        # Extract rotations and translations
        R_b2g = T_base2gripper[:3, :3]
        t_b2g = T_base2gripper[:3, 3]
        
        R_w2c = T_world2cam[:3, :3]
        t_w2c = T_world2cam[:3, 3]
        
        # Store the transformations
        R_world2cam.append(R_w2c)
        t_world2cam.append(t_w2c)
        R_base2gripper.append(R_b2g)
        t_base2gripper.append(t_b2g)
        
        successful_poses += 1
        print(f"Successfully processed pose {i+1}")
        i += 1
    
    camera.close()

    print(f"\nSuccessfully processed {successful_poses} poses out of {len(q_poses)}")
    if successful_poses < 3:
        raise ValueError(f"Not enough valid poses for calibration. Only {successful_poses} poses were successful.")

    try:
        # Convert lists to numpy arrays
        R_world2cam = np.array(R_world2cam)
        t_world2cam = np.array(t_world2cam)
        R_base2gripper = np.array(R_base2gripper)
        t_base2gripper = np.array(t_base2gripper)

        print("\nPerforming robot-world hand-eye calibration...")
        R_cam2gripper, t_cam2gripper, R_base2target, t_base2target = cv2.calibrateRobotWorldHandEye(
            R_world2cam, t_world2cam,
            R_base2gripper, t_base2gripper,
            method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH
        )
        
        # Create transformation matrices
        T_cam2gripper = np.eye(4, dtype=np.float64)
        T_cam2gripper[:3, :3] = R_cam2gripper
        T_cam2gripper[:3, 3] = t_cam2gripper.ravel()

        T_base2target = np.eye(4, dtype=np.float64)
        T_base2target[:3, :3] = R_base2target
        T_base2target[:3, 3] = t_base2target.ravel()
        
        print("\n=== Results of Robot-World Hand-Eye Calibration ===")
        print("T_cam2gripper =\n", T_cam2gripper)
        print("T_base2target =\n", T_base2target)
        
        # Verify calibration for each pose
        for i, q in enumerate(q_poses):
            T_base2gripper = robot.fk(q)
            T_cam2base = T_base2gripper @ T_cam2gripper
            print(f"\nVerification for pose {i+1}:")
            print(f"T_cam2base:\n", T_cam2base)
            print(f"T_cam2gripper:\n", T_cam2gripper)
        
        return T_cam2gripper, T_base2target
        
    except Exception as e:
        print(f"\nError during calibration: {str(e)}")
        print("R_world2cam shape:", R_world2cam.shape)
        print("t_world2cam shape:", t_world2cam.shape)
        print("R_base2gripper shape:", R_base2gripper.shape)
        print("t_base2gripper shape:", t_base2gripper.shape)
        raise

# ... existing code ...

if __name__ == "__main__":
    robot.initialize()
    np.atan2 = np.arctan2 # should be there to resolve numpy dependency mismatch

    # Initialize empty list
    q_poses = []

    # Load files q1.npy through however many you have
    for i in range(1, 11):  # adjust range as needed
        try:
            q = np.load(f'calib_poses/q{i}.npy')
            q_poses.append(q)
            print(f"Loaded q{i}.npy: {q}")
        except FileNotFoundError:
            break  # Stop when we don't find the next file

    print(f"\nLoaded {len(q_poses)} poses")
    
    # Run calibration with 10 poses
    T_cam2gripper, T_base2target = example_calibration(robot, q_poses)
    
    # Save calibration results
    np.save('robot_matrices/calibration_cam2gripper.npy', T_cam2gripper)
    np.save('robot_matrices/calibration_base2target.npy', T_base2target)
    
    print("Calibration completed and results saved successfully!")