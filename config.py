"""
Configuration file for the project
"""

import cv2.aruco as aruco
import numpy as np
import os

# MAIN SETTINGS
# ----------------------------------------------------------------
# ROBOT and CAMERA
ROBOT_NAME = "crs97" # "crs93" or "crs97"
CAMERA_NAME = "camera-crs97" # "camera-crs93" or "camera-crs97"
ROBOT_INITIALIZE = True # Whether to perform full robot initialization at the start
NUM_PHOTOS = 5 # Number of photos to take during task

# BOARDS
SINGLE_BOARD_MODE = False 
START_BOARD_CSV = "csvs/positions_plate_07-08.csv"
TARGET_BOARD_CSV = "csvs/positions_plate_05-06.csv"

# ROBOT INITIAL POSE
# ROBOT_INITIAL_POSE = "calib_poses/q_task_start_new.npy"
ROBOT_INITIAL_POSE = "calib_poses/q_task_start_2.npy"
ROBOT_ROTATE_Z_LEFT = "robot_matrices/R_z_axis.npy"
# ----------------------------------------------------------------

# ARUCO MARKERS and CALIBRATION
# ----------------------------------------------------------------
# ARUCO markers for task
ARUCO_DICT_SIZE_TEST = aruco.DICT_4X4_100 # size of ArUco dictionary and Aruco markers
MARKER_SIZE_TEST = 0.035 # size of ArUco marker in meters

# Camera calibration
USE_ARUCO_CALIB = True # use ArUco board for calibration instead of chessboard
ARUCO_DICT_SIZE = aruco.DICT_5X5_100 # size of ArUco dictionary and Aruco markers
MARKER_SIZE = 0.022 # size of ArUco marker in meters
# MARKER_SIZE = 0.015 # size of ArUco marker in meters
# MARKER_SIZE = 0.026 # size of ArUco marker in meters
SQUARE_SIZE = 0.03 # size of square in meters
# SQUARE_SIZE = 0.02 # size of square in meters
# SQUARE_SIZE = 0.035 # size of square in meters
NUM_IMAGES = 20 # number of images to take for calibration
# NUM_SQUARES = (9, 7) # number of squares width x height
NUM_SQUARES = (7, 5) # number of squares width x height
# NUM_SQUARES = (10, 14) # number of squares width x height
SLEEP_TIME_CALIB = 4000 # time to wait between photos for calibration in ms

# Robot EYE-TO-HAND calibration
# ARUCO_DICT_SIZE_TEST = aruco.DICT_6X6_100 # size of ArUco dictionary and Aruco markers
# MARKER_SIZE_TEST = 0.038 # size of ArUco marker in meters
# ----------------------------------------------------------------

# CONSTANTS
# ----------------------------------------------------------------
CUBE_HOLE_SIZE = 0.04 # size of cube hole in meters

# corners of the board in meters. Center is at the center of aruco marker 
# with the smallest id on the board
BOARD_CORNERS = np.array([ 
        [-0.024,  0.164, 0], # top-left
        [ 0.204,  0.164, 0], # top-right
        [ 0.204, -0.024, 0], # bottom-right
        [-0.024, -0.024, 0]  # bottom-left
    ], dtype=np.float32)        
# ----------------------------------------------------------------

USE_CALIB_MATRICES = True

# LOAD CAMERA and ROBOT CALIBRATION MATRICES
# ----------------------------------------------------------------
try:
    # camera_matrix_path = f'camera_matrices/camera_matrix_{ROBOT_NAME}.npy'
    # dist_coeffs_path = f'camera_matrices/dist_coeffs_{ROBOT_NAME}.npy'
    # t_rc_path = f"robot_matrices/calibration_rc_{ROBOT_NAME}_BODYA_2.npy"

    camera_matrix_path = f'camera_matrices/camera_matrix_{ROBOT_NAME}_today3.npy'
    dist_coeffs_path = f'camera_matrices/dist_coeffs_{ROBOT_NAME}_today3.npy'
    t_rc_path = f"robot_matrices/calibration_rc_{ROBOT_NAME}_BODYA_old_better_today3.npy"
    # t_rc_path = f"robot_matrices/calibration_rc_{ROBOT_NAME}_BODYA_today3.npy"
    # t_rc_path = f"robot_matrices/calibration_rc_{ROBOT_NAME}_BODYA_old_10_poses_today3.npy"

    if os.path.exists(camera_matrix_path) and os.path.exists(dist_coeffs_path) and os.path.exists(t_rc_path):
        CAMERA_MATRIX = np.load(camera_matrix_path)
        DIST_COEFFS = np.load(dist_coeffs_path)
        T_RC = np.load(t_rc_path)
    else:
        print("Warning: One or more calibration files not found. Setting matrices to None.")
        CAMERA_MATRIX = None
        DIST_COEFFS = None 
        T_RC = None
except Exception as e:
    print(f"Error loading calibration files: {e}")
    CAMERA_MATRIX = None
    DIST_COEFFS = None
    T_RC = None


# DEBUG prints
print(f"Imported camera matrix =\n{CAMERA_MATRIX}")
print(f"Imported distortion coefficients =\n{DIST_COEFFS}")
print(f"Imported original T_rc =\n{T_RC}")
# ----------------------------------------------------------------