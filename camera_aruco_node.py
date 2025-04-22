import cv2
import cv2.aruco as aruco
import numpy as np
import csv

# Our Basler camera interface
from basler_camera import BaslerCamera

import config

# === Load camera intrinsics ===
if config.USE_CALIB_MATRICES:
    # Load camera matrix and distortion coefficients
    camera_matrix = np.load(f'camera_matrices/camera_matrix_{config.ROBOT_NAME}_today3.npy')
    dist_coeffs = np.load(f'camera_matrices/dist_coeffs_{config.ROBOT_NAME}_today3.npy')
    # Transformation matrix from camera to robot
    T_robot_camera = config.T_RC
else:
    # Camera matrix and distortion coefficients given by realsense
    camera_matrix = config.CAMERA_MATRIX_HARD_CODED
    dist_coeffs = config.DIST_COEFFS_HARD_CODED


class SingleMarkerDetector:
    def __init__(self):
        # Initialize ArUco detector
        self.detect_params = aruco.DetectorParameters()
        self.refine_params = aruco.RefineParameters()
        
        # Example: marker_size for pose estimation (in meters)
        self.marker_size = config.MARKER_SIZE_TEST
        
        self.detector = aruco.ArucoDetector(
            dictionary=aruco.getPredefinedDictionary(config.ARUCO_DICT_SIZE_TEST),
            detectorParams=self.detect_params,
            refineParams=self.refine_params
        )

    def get_rvec_tvec_of_first_marker(self, image: np.ndarray):
        """
        Detect ArUco markers in 'image' and return (rvec, tvec)
        for the first detected marker. If none found, returns (None, None).
        """
        try:
            corners, ids, _ = self.detector.detectMarkers(image)
        except Exception as e:
            print(f"Error detecting ArUco markers: {e}")
            return None, None
        
        if ids is not None and len(ids) > 0:

            # Draw detected markers for visualization
            image_drawn = aruco.drawDetectedMarkers(image.copy(), corners, ids)

            # We'll take the first detected marker (index 0).
            i = 0
            
            # Object points for a square marker of side self.marker_size
            # Defined in the marker's own coordinate system (Z = 0 plane).
            obj_points = np.array([
                [-self.marker_size / 2,  self.marker_size / 2, 0],
                [ self.marker_size / 2,  self.marker_size / 2, 0],
                [ self.marker_size / 2, -self.marker_size / 2, 0],
                [-self.marker_size / 2, -self.marker_size / 2, 0]
            ], dtype=np.float32)
            try:
                success, rvec, tvec = cv2.solvePnP(
                    obj_points,
                    corners[i],
                    camera_matrix,
                    dist_coeffs
                )
            except Exception as e:
                print(f"Error solving PnP: {e}")
                return None, None
            
            if success:
                # Draw frame axes for visualization
                cv2.drawFrameAxes(
                    image_drawn,
                    camera_matrix,
                    dist_coeffs,
                    rvec,
                    tvec,
                    self.marker_size/2
                )
                
                # Print marker center
                position = tvec.flatten()  # (x, y, z) in meters
                pos_text = f"ID X={position[0]:.3f}m Y={position[1]:.3f}m Z={position[2]:.3f}m"
                cv2.putText(
                    image_drawn,
                    pos_text,
                    (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
                
                # Convert rotation vector to rotation matrix and then to Euler angles
                rot_mat, _ = cv2.Rodrigues(rvec)
                euler_angles = cv2.RQDecomp3x3(rot_mat)[0]  # Returns in degrees
                
                # Add Euler angles text
                euler_text = f"Roll={euler_angles[0]:.1f} Pitch={euler_angles[1]:.1f} Yaw={euler_angles[2]:.1f}"
                cv2.putText(
                    image_drawn,
                    euler_text,
                    (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2
                )
                return rvec, tvec
            else:
                print("PnP failed")
        else:
            print("No ArUco marker detected in this frame\n")
        
        return None, None
