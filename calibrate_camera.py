#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import cv2.aruco as aruco
import time
from basler_camera import BaslerCamera

import config

def calibrate_with_chessboard(camera: BaslerCamera, num_images: int = 20):
    """
    Calibrate camera using traditional chessboard pattern
    Args:
        camera: BaslerCamera instance
        num_images: Number of images to capture for calibration
    """
    # Chessboard parameters
    board_size = (9, 6)  # number of internal corners width x height
    square_size = 0.025  # size of square in meters

    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ...
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1,2) * square_size

    # Arrays to store object points and image points
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane
    
    print("Preparing to capture images in 3 seconds...")
    time.sleep(3)
    
    images_captured = 0
    while images_captured < num_images:
        # Capture image from camera
        img = camera.grab_image()
        
        if img is None or img.size == 0:
            print("Failed to capture image")
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(
            image=gray, 
            patternSize=board_size, 
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            objpoints.append(objp)
            
            # Refine corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(
                image=gray,
                corners=corners,
                winSize=(11, 11),
                zeroZone=(-1, -1),
                criteria=criteria
            )
            imgpoints.append(corners2)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, board_size, corners2, ret)
            images_captured += 1
            print(f"Captured image {images_captured}/{num_images}")
            
        # Display image
        cv2.imshow('Calibration', img)
        cv2.waitKey(1)
        
        # Wait 1 second before next capture
        time.sleep(1)

    cv2.destroyAllWindows()

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=objpoints,
        imagePoints=imgpoints,
        imageSize=gray.shape[::-1],
        cameraMatrix=None,
        distCoeffs=None
    )
    
    return mtx, dist

def calibrate_with_aruco(camera: BaslerCamera, num_images: int = 20):
    """
    Calibrate camera using ArUco markers
    Args:
        camera: BaslerCamera instance
        num_images: Number of images to capture for calibration
    """
    # Initialize detector the same way as in the working code
    detect_params = aruco.DetectorParameters()
    ref_param = aruco.RefineParameters()
    
    # Create dictionary first
    aruco_dict = aruco.getPredefinedDictionary(config.ARUCO_DICT_SIZE)
    
    # Create detector with same parameters as working code
    detector = aruco.ArucoDetector(
        dictionary=aruco_dict,
        detectorParams=detect_params,
        refineParams=ref_param
    )
    
    # Create CharUco board
    board = aruco.CharucoBoard(
        size=config.NUM_SQUARES,          # number of squares width x height
        squareLength=config.SQUARE_SIZE,    # size of square in meters
        markerLength=config.MARKER_SIZE,   # size of ArUco marker in meters
        dictionary=aruco_dict  # use dictionary directly, not from detector,
    )
    
    # Create CharUco detector
    charucodetector = cv2.aruco.CharucoDetector(board)
    
    all_corners = []
    all_ids = []
    
    print("Preparing to capture images in 3 seconds...")
    time.sleep(3)
    
    images_captured = 0
    while images_captured < num_images:
        print(f"Capturing image {images_captured + 1}/{num_images}")
        # Capture image from camera
        img = camera.grab_image()
        
        if img is None or img.size == 0:
            print("Failed to capture image")
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers using same method as working code
        corners, ids, rejected = detector.detectMarkers(gray)
        
        if rejected is not None:
            print("No of rejected markers: ", len(rejected))
            print("Rejected markers:")
            # print(rejected)


        print(f"Number of corners: {len(corners)}")

        charuco_corners, charuco_ids, marker_corners, marker_ids = charucodetector.detectBoard(gray)
        # if charuco_corners is not None:
        #     print(f"charuco_corners: {charuco_corners}\n")
        # if charuco_ids is not None:
        #     print(f"charuco_ids: {charuco_ids}\n")
        # if marker_corners is not None:
        #     print(f"marker_corners: {marker_corners}\n")
        # if marker_ids is not None:
        #     print(f"marker_ids: {marker_ids}\n")

        if ids is not None and len(ids) > 0:
            # Draw markers just like in working code
            img = aruco.drawDetectedMarkers(img.copy(), corners, ids)
            print("Markers detected successfully")
            if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 0:
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)
                images_captured += 1
                print(f"Captured image {images_captured}/{num_images}")
        else:
            print("No markers detected")
            
        # Display image
        cv2.imshow('Calibration', img)
        cv2.waitKey(config.SLEEP_TIME_CALIB)
    
    cv2.destroyAllWindows()
    
    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraCharuco(
        charucoCorners=all_corners,
        charucoIds=all_ids,
        board=board,
        imageSize=gray.shape[::-1],
        cameraMatrix=None,
        distCoeffs=None
    )

    print_calibration_error(mtx, dist, all_corners, all_ids, rvecs, tvecs, board)
    
    return ret, mtx, dist

def print_calibration_error(mtx, dist, all_corners, all_ids, rvecs, tvecs, board):
    """
    Calculate reprojection error for CharUco calibration
    Args:
        mtx: Camera matrix
        dist: Distortion coefficients
        all_corners: List of detected CharUco corners from all images
        all_ids: List of detected CharUco corner IDs from all images
        rvecs: Rotation vectors from calibration
        tvecs: Translation vectors from calibration
        board: CharucoBoard instance
    """
    mean_error = 0
    total_points = 0
    
    print("\nCalculating calibration error...")
    print(f"Number of images to process: {len(all_corners)}")
    
    for i, (corners, ids, rvec, tvec) in enumerate(zip(all_corners, all_ids, rvecs, tvecs)):
        try:
            print(f"\nProcessing image {i+1}")
            print(f"Number of corners: {len(corners)}")
            print(f"Original IDs shape and content: {ids.shape}, {ids}")
            
            # Convert ids to the correct format
            ids = ids.astype(np.int32)
            if ids.ndim > 1:
                ids = ids.squeeze()
            
            print(f"Processed IDs shape and content: {ids.shape}, {ids}")
            
            # Get 3D coordinates of the detected corners
            objpoints = board.getChessboardCorners(ids)
            
            if objpoints is None:
                print(f"Warning: Could not get chessboard corners for IDs: {ids}")
                print(f"Board size: {board.getChessboardSize()}")
                print(f"Valid IDs range: 0 to {board.getChessboardSize()[0] * board.getChessboardSize()[1] - 1}")
                continue
            
            print(f"Successfully got {len(objpoints)} 3D points")
            
            # Project points using the calibrated camera
            imgpoints2, _ = cv2.projectPoints(objpoints, rvec, tvec, mtx, dist)
            
            # Calculate error
            error = cv2.norm(corners, imgpoints2.reshape(-1, 2), cv2.NORM_L2)
            mean_error += error
            total_points += len(corners)
            print(f"Added error: {error}, Total points so far: {total_points}")
            
        except Exception as e:
            print(f"Error processing points: {e}")
            print(f"Error type: {type(e)}")
            continue
    
    mean_error = mean_error / total_points if total_points > 0 else 0
    print(f"\nFinal Results:")
    print(f"Total points processed: {total_points}")
    print(f"Total reprojection error: {mean_error} pixels")
    
    return mean_error

def main():
    # Initialize camera
    camera = BaslerCamera()
    
    # Connect to camera (modify as needed for your setup)
    camera.connect_by_name(config.CAMERA_NAME)  # or use connect_by_ip("192.168.137.106")
    
    # Open the communication with the camera
    camera.open()
    
    # Set capturing parameters
    camera.set_parameters()
    
    try:
        if config.USE_ARUCO_CALIB:
            ret, K, dist = calibrate_with_aruco(camera, num_images=config.NUM_IMAGES)
        else:
            ret, K, dist = calibrate_with_chessboard(camera, num_images=config.NUM_IMAGES)
        
        print("\nCamera Matrix K:")
        print(K)
        print("\nDistortion Coefficients:")
        print(dist)
        print(f"Reprojection error:")
        print(ret) 
        
        # Save calibration results
        np.save(f'camera_matrices/camera_matrix_{config.ROBOT_NAME}.npy', K)
        np.save(f'camera_matrices/dist_coeffs_{config.ROBOT_NAME}.npy', dist)
        np.save(f'camera_matrices/reprojection_error_{config.ROBOT_NAME}.npy', ret)
        print(f"\nCalibration parameters saved to camera_matrix_{config.ROBOT_NAME}.npy, dist_coeffs_{config.ROBOT_NAME}.npy and reprojection_error_{config.ROBOT_NAME}.npy")
        
    finally:
        # Always close the camera connection
        camera.close()

if __name__ == '__main__':
    main()