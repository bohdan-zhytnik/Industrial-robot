import numpy as np
import cv2
import config
from camera_aruco_node import SingleMarkerDetector
from basler_camera import BaslerCamera

import time

if config.ROBOT_NAME == "crs93":
    from ctu_crs import CRS93
    robot = CRS93()  # set argument tty_dev=None if you are not connected
elif config.ROBOT_NAME == "crs97":
    from ctu_crs import CRS97
    robot = CRS97()  # set argument tty_dev=None if you are not connected

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
    TODO
    """
    A_R_rg = []
    A_t_rg = [] 
    B_R_cm = []
    B_t_cm = [] 

    camera = BaslerCamera()
    camera.connect_by_name(config.CAMERA_NAME)
    camera.open()
    camera.set_parameters()

    successful_poses = 0

    for i, q in enumerate(q_poses):

        if not robot.in_limits(q):
            print("Robot is not in limits")
            continue


        # Move robot and wait
        robot.move_to_q(q)
        robot.wait_for_motion_stop()
        print(f"\nProcessing pose {i+1}/{len(q_poses)}")

        # (1) T_gripper2base
        T_rg = robot.fk(q)
        print("T_rg =\n", T_rg)
        time.sleep(0.5)
        # (2) Detect marker in the camera
        img = camera.grab_image()
        # if img is not None:
        # # Save the image with a similar naming convention as your pose files
        #     image_filename = f"calib_poses/img_{config.ROBOT_NAME}_pos{i}.jpg"
        #     cv2.imwrite(image_filename, img)
        #     print(f"Image saved for pose {i}")
        aruco_node = SingleMarkerDetector()
        rvec, tvec = aruco_node.get_rvec_tvec_of_first_marker(img)

        if rvec is None or tvec is None:
            print(f"Skipping pose {i+1} - marker not detected")
            continue

        # (3) Build the marker->camera transform
        T_cm = matrix_from_rvec_tvec(rvec, tvec)

        R_cm = T_cm[:3, :3]
        t_cm = T_cm[:3, 3]

        print(f"T_cm: {T_cm}")
        
        # Extract the gripper->base rotation&translation
        R_rg = T_rg[:3, :3]
        t_rg = T_rg[:3, 3]

        print("T_rg =\n", T_rg)

        # Store them
        A_R_rg.append(R_rg)
        A_t_rg.append(t_rg)
        B_R_cm.append(R_cm)
        B_t_cm.append(t_cm)

        successful_poses += 1
        print(f"Successfully processed pose {i+1}")

    camera.close()

    print(f"\nSuccessfully processed {successful_poses} poses out of {len(q_poses)}")
    if successful_poses < 3:
        raise ValueError(f"Not enough valid poses for calibration. Only {successful_poses} were good.")

    # Convert to the format calibrateRobotWorldHandEye expects
    A_R_rg = np.array(A_R_rg)
    A_t_rg = np.array(A_t_rg)
    B_R_cm = np.array(B_R_cm)
    B_t_cm = np.array(B_t_cm)

    # === Perform the calibration ===
    try:
        print("\nPerforming Robot-World/Hand-Eye calibration...")
        (R_gm, t_gm,
         R_rc, t_rc) = cv2.calibrateRobotWorldHandEye(
            A_R_rg, A_t_rg,
            B_R_cm, B_t_cm,
            method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH
        )


        print("R_gm =\n", R_gm) 
        print("t_gm =\n", t_gm)
        print("R_rc =\n", R_rc)
        print("t_rc =\n", t_rc)

        # Create transformation matrices directly (R is already a matrix)
        T_gm = np.eye(4, dtype=np.float64)
        T_gm[:3, :3] = R_gm
        T_gm[:3, 3] = t_gm.flatten()

        T_rc = np.eye(4, dtype=np.float64)
        T_rc[:3, :3] = R_rc
        T_rc[:3, 3] = t_rc.flatten()

        print("\n=== Results of Robot-In-Hand Calibration ===")
        print("T_gm =\n", T_gm)
        print("T_rc =\n", T_rc)

        return T_gm, T_rc

    except Exception as e:
        print(f"\nError during calibration: {str(e)}")
        print("R_gm shape:", R_gm.shape)
        print("t_gm shape:", t_gm.shape)
        print("R_rc shape:", R_rc.shape)
        print("t_rc shape:", t_rc.shape)
        raise

if __name__ == "__main__":
    if config.ROBOT_INITIALIZE:
        robot.initialize()
    else:
        robot.initialize(home=False)
        robot.soft_home()
    np.atan2 = np.arctan2  # to resolve numpy dependency mismatch

    # Load q-poses from disk
    q_poses = []
    # for i in range(1, 250):
    # for i in range(16, 155):
        # if i < 60 or i > 70:
    for i in range(1, 13):
        #     continue
        try:
            # q = np.load(f'calib_poses/crs93_many_poses/q_crs97_pos{i}.npy')
            # q = np.load(f'calib_poses/crs93_many_poses_old_better/q_crs97_pos{i}.npy')
            q = np.load(f'calib_poses/only_10_both_robots/q_crs93_pos{i}.npy')
            q_poses.append(q)
            print(f"Loaded q_crs93_pos{i}.npy: {q}")
        except FileNotFoundError:
            break

    print(f"\nLoaded {len(q_poses)} poses")
    T_gm, T_rc = example_calibration(robot, q_poses)

    # Save results
    # np.save(f'robot_matrices/calibration_gm_{config.ROBOT_NAME}_BODYA_today3.npy', T_gm)
    # np.save(f'robot_matrices/calibration_rc_{config.ROBOT_NAME}_BODYA_today3.npy', T_rc)
    np.save(f'robot_matrices/calibration_gm_{config.ROBOT_NAME}_BODYA_old_10_poses_today3.npy', T_gm)
    np.save(f'robot_matrices/calibration_rc_{config.ROBOT_NAME}_BODYA_old_10_poses_today3.npy', T_rc)

    print("Calibration completed and results saved successfully!")
