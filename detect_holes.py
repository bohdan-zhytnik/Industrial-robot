import cv2
import cv2.aruco as aruco
import numpy as np
import csv
import argparse
import statistics
from typing import Dict, List, Tuple
import time
# Our Basler camera interface
from basler_camera import BaslerCamera

from scipy.spatial.transform import Rotation as R
import config

class CameraArUcoNode:
    def __init__(self, start_board_csv, target_board_csv = None):
        """
        Initialize the ArUco detector and load hole offsets from CSV files.
        
        Args:
            start_board_csv: Path to CSV file with hole offsets for start board
            target_board_csv: Path to CSV file with hole offsets for target board
        """
        # --- Set up ArUco Detector ---
        self.detect_params = aruco.DetectorParameters()
        self.ref_param = aruco.RefineParameters()
        self.marker_size = config.MARKER_SIZE_TEST
        
        self.detector = aruco.ArucoDetector(
            dictionary=aruco.getPredefinedDictionary(config.ARUCO_DICT_SIZE_TEST),
            detectorParams=self.detect_params,
            refineParams=self.ref_param,
        )
        
        # --- Load offsets from CSV files (distances in mm) ---
        self.hole_offsets_mm = {
            'start': [],
            'target': []
        }
        self._load_hole_offsets(start_board_csv, 'start')

        if target_board_csv is not None:
            self.single_board_mode = False
            self._load_hole_offsets(target_board_csv, 'target')
        else:
            self.single_board_mode = True

        self.board_states = {}  # Will store True for start board, False for target board
        
        if self.single_board_mode:
            print("Single board mode: only one board will be processed.")
        else:
            print(f"Dual board mode: two boards will be processed.")
        
        self.marker_history: Dict[int, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}  # Store (rvec, tvec) history
        
    def _load_hole_offsets(self, csv_file_path, board_type):
        """
        Load hole offsets from a CSV file in millimeters.
        
        Args:
            csv_file_path: Path to CSV file with hole offsets
            board_type: Either 'start' or 'target' to identify which board
        """
        with open(csv_file_path, mode='r', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            lines = list(reader)

            # Store marker IDs separately for each board
            if board_type == 'start':
                self.start_marker_ids  = [int(x) for x in lines[0]]
            else:
                self.target_marker_ids = [int(x) for x in lines[0]]
            
            for row in lines[1:]:
                if len(row) < 2:
                    continue
                x_mm = float(row[0])
                y_mm = float(row[1])
                self.hole_offsets_mm[board_type].append((x_mm, y_mm))
        
        print(f"Loaded {len(self.hole_offsets_mm[board_type])} hole offsets for {board_type} board.")
        print(f"Board type: {board_type}, marker IDs: {self.start_marker_ids if board_type == 'start' else self.target_marker_ids}")
    
    def get_marker_averages(self, marker_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate average pose for a marker from its history."""
        if marker_id not in self.marker_history or not self.marker_history[marker_id]:
            return None, None, None
        
        history = self.marker_history[marker_id]
        
        # Position averaging (tvec) - simple mean is fine
        tvecs = np.array([h[1] for h in history])
        all_corners = np.array([h[2] for h in history])
        """
        Choose mean or median
        """
        # avg_tvec = np.mean(tvecs, axis=0)
        avg_tvec_median = np.median(tvecs, axis=0)  # Compute median along each axis
        """
        """
        avg_corners_median = np.median(all_corners, axis=0)
        # avg_corners_mean = np.mean(all_corners, axis=0)

        # Rotation averaging using rotation matrices
        rvecs_for_quant = []
        rot_matrices = []
        for h in history:
            R, _ = cv2.Rodrigues(h[0])
            rot_matrices.append(R)

            rvecs_for_quant.append(h[0])
        
        # Compute the average rotation matrix
        avg_R = np.median(rot_matrices, axis=0)
        
        # Ensure the averaged rotation matrix is valid using SVD
        U, _, Vt = np.linalg.svd(avg_R)
        avg_R = U @ Vt
        
        # Convert the averaged rotation matrix back to a rotation vector
        avg_rvec, _ = cv2.Rodrigues(avg_R)

        # avg_rvec_quant = self.average_rvec_quaternions(rvecs_for_quant)
        # avg_rvec_median, _ = cv2.Rodrigues(avg_rvec_quant)
        
        # Convert the averaged rotation matrix to Euler angles for display
        euler_angles = cv2.RQDecomp3x3(avg_R)[0]
        # euler_angles_median = cv2.RQDecomp3x3(avg_rvec_median)[0]
        
        # NOTE: DEBUG prints
        # print(f"\nMarker {marker_id} rotations:")
        # print(f"Final average angles: {euler_angles}")
        # print(f"Average Euler angles for marker {marker_id}: Roll={euler_angles[0]:.1f}° Pitch={euler_angles[1]:.1f}° Yaw={euler_angles[2]:.1f}°")
        
        return avg_rvec, avg_tvec_median, euler_angles, avg_corners_median
        # return avg_rvec_median, avg_tvec_median, euler_angles_median
    

    def average_rvec_quaternions(self,rvecs):
        """
        Compute the average rotation vector (rvec) using quaternion averaging.

        Args:
            rvecs (list of np.ndarray): List of 3x1 rotation vectors.

        Returns:
            np.ndarray: Averaged rotation vector.
        """
        # Convert rvecs to rotation matrices
        quaternions = []
        for rvec in rvecs:
            rot_mat, _ = cv2.Rodrigues(rvec)  # Convert to rotation matrix
            quat = R.from_matrix(rot_mat).as_quat()  # Convert to quaternion (x, y, z, w)
            quaternions.append(quat)
        
        # Compute mean quaternion
        quaternions = np.array(quaternions)
        mean_quat = np.mean(quaternions, axis=0)
        mean_quat /= np.linalg.norm(mean_quat)  # Normalize to ensure unit quaternion

        # Convert back to rotation matrix and then to rvec
        avg_rot_mat = R.from_quat(mean_quat).as_matrix()
        avg_rvec, _ = cv2.Rodrigues(avg_rot_mat)


        for i in range(1, len(quaternions)):
            if np.dot(quaternions[0], quaternions[i]) < 0:
                quaternions[i] = -quaternions[i]
            
        # Compute the component-wise median
        median_quat = np.median(quaternions, axis=0)
        median_quat /= np.linalg.norm(median_quat)  # Normalize to ensure unit quaternion

        # Convert back to rotation matrix and then to rvec
        median_rot_mat = R.from_quat(median_quat).as_matrix()
        median_rvec, _ = cv2.Rodrigues(median_rot_mat)
        # return median_rvec
    
        return avg_rvec



    def camera_callback(self, camera, offset_version: bool = False) -> dict:
        """Modified callback to handle multiple photos and averaging."""
        holes_camera_frame = {
            'start': [],
            'target': []
        }        
        original_parameters = {}
        # Initialize marker history at the start
        self.marker_history.clear()  # Clear any previous history
        print(f"\nTaking {config.NUM_PHOTOS} photos...")
        # Take multiple photos and collect data
        for i in range(config.NUM_PHOTOS):
            print(f"\nCapturing photo {i+1}/{config.NUM_PHOTOS}")
            img = camera.grab_image()
            if img is not None and img.size > 0:
                """
                    Sleep
                """
                time.sleep(0.05)  # Short delay between photos
                marker_poses, cv_image, corners, ids = self.detect_and_process_markers(img, photo_number=(i+1))

                # Initialize history for newly detected markers
                if ids is not None:
                    print(f"Detected {len(ids)} markers in photo {i+1}")
                    for marker_id in ids.flatten():
                        if marker_id not in self.marker_history:
                            self.marker_history[marker_id] = []

                # Store poses in history
                for marker_id in marker_poses.keys():
                    rvec, tvec, corners = marker_poses[marker_id]
                    self.marker_history[marker_id].append((rvec, tvec, corners))
            else:
                print("Image was not captured.")

        for marker_id in self.marker_history.keys():
            avg_rvec, avg_tvec, avg_euler, avg_corners = self.get_marker_averages(marker_id)
            original_parameters[marker_id] = (avg_rvec, avg_tvec)
            # avg_rvec, avg_tvec, avg_euler = (rvec, tvec, None)
            if avg_rvec is not None and avg_tvec is not None and avg_corners is not None:
                marker_poses[marker_id] = (avg_rvec, avg_tvec, avg_corners)
                if avg_euler is None:
                    print(f"No Euler angles available for marker {marker_id}")
                # Visualize marker information
                self._visualize_marker_info(
                    cv_image,
                    marker_id,
                    avg_corners,
                    avg_rvec,
                    avg_tvec,
                    avg_euler,
                    photo_number=None
                )

        # Process markers and holes using averaged poses
        if marker_poses:
            # Update board states based on detected markers
            for marker_id in marker_poses.keys():
                if self.single_board_mode:
                    # In single board mode, first detected marker is the start board
                    self.board_states[marker_id] = True
                else:
                    # In dual board mode, see if marker is on start board
                    self.board_states[marker_id] = (marker_id in self.start_marker_ids)

                if self.board_states[marker_id]:
                    print(f"Marker {marker_id} is on start board")
                else:
                    print(f"Marker {marker_id} is on target board")
            
            # Process each board separately
            for board_type in self.hole_offsets_mm.keys():
                # Filter markers using predefined marker IDs for each board
                print(f"\nDebug marker filtering for {board_type} board:")
                print(f"All marker IDs available: {list(marker_poses.keys())}")
                print(f"Expected marker IDs for this board: {self.start_marker_ids if board_type == 'start' else self.target_marker_ids}")

                board_markers = {
                    marker_id: marker_poses[marker_id] 
                    for marker_id in marker_poses.keys() 
                    if marker_id in (self.start_marker_ids if board_type == 'start' else self.target_marker_ids)
                }

                print(f"Filtered marker IDs: {list(board_markers.keys())}")

                old_board_tvec_markers = {
                    marker_id: original_parameters[marker_id]
                    for marker_id in board_markers.keys()
                    if marker_id in (self.start_marker_ids if board_type == 'start' else self.target_marker_ids)
                }

                print(f"\n=== Processing {board_type} board ===")
                print(f"Detected markers: {sorted(list(board_markers.keys()), reverse=False)}")

                # Skip if no markers found for this board type
                if not board_markers:
                    continue

                # Calculate board transform using the first two markers
                current_marker_ids = sorted(list(board_markers.keys()), reverse=False)
                board_corners = {}
                # print("\nCollecting corner information:")
                for marker_id in board_markers.keys():
                    board_corners[marker_id] = board_markers[marker_id][2]

                    # NOTE: DEBUG prints
                    # print(f"Marker {marker_id} corners shape: {board_corners[marker_id].shape}")
                    # print(f"Corner coordinates: \n{board_corners[marker_id].reshape(4,2)}")

                if len(board_markers.keys()) == 2:
                    # print(f"\nCalculating board transform using markers {current_marker_ids[0]} and {current_marker_ids[1]}")
                    ref_rvec, ref_tvec = self.calculate_board_transform(
                        board_corners,
                        current_marker_ids[0],
                        current_marker_ids[1],
                        config.CAMERA_MATRIX,
                        config.DIST_COEFFS
                    )
                    
                    if ref_rvec is not None and ref_tvec is not None:

                        # NOTE: debug prints, remove later
                        print("\nBoard transform results:")
                        print(f"ref_rvec: {ref_rvec.flatten()}")
                        print(f"ref_tvec: {ref_tvec.flatten()}")

                        marker_T_cm = np.eye(4)
                        # Convert rvec/tvec to transformation matrix
                        R, _ = cv2.Rodrigues(ref_rvec)
                        marker_T_cm[:3, :3] = R
                        marker_T_cm[:3, 3] = ref_tvec.flatten()
                        print(f"vectors converted to T_cm: {marker_T_cm}")
                        # ------------------------------------------------------------
                        
                        # NOTE: visualization, draw board physical boundaries and axis
                        # Draw board boundary
                        board_physical_corners = config.BOARD_CORNERS
                        board_image_points, _ = cv2.projectPoints(
                            board_physical_corners,
                            ref_rvec,
                            ref_tvec,
                            config.CAMERA_MATRIX,
                            config.DIST_COEFFS
                        )
                        board_points = board_image_points.reshape(-1, 2).astype(np.int32)
                        cv2.polylines(cv_image, [board_points], True, (0, 255, 255), 2)  # Yellow color for board outline
                        cv2.drawFrameAxes(
                            cv_image,
                            config.CAMERA_MATRIX,
                            config.DIST_COEFFS,
                            ref_rvec,
                            ref_tvec,
                            self.marker_size * 4  # Make axes four times  marker size for visibility,
                        )
                        # ------------------------------------------------------------
                        # NOTE: debug prints, remove later
                        # # Compare with individual marker poses
                        # print("\nComparing with individual marker poses:")
                        # for marker_id in current_marker_ids:
                        #     orig_rvec, orig_tvec = old_board_tvec_markers[marker_id]
                        #     print(f"\nMarker {marker_id}:")
                        #     print(f"Original rvec: {orig_rvec.flatten()}")
                        #     print(f"Original tvec: {orig_tvec.flatten()}")
                        #     original_marker_T_cm = np.eye(4)
                        #     # Convert rvec/tvec to transformation matrix
                        #     R, _ = cv2.Rodrigues(orig_rvec)
                        #     original_marker_T_cm[:3, :3] = R
                        #     original_marker_T_cm[:3, 3] = orig_tvec.flatten()
                        #     print(f"ORIGINAL vectors converted to T_cm: {original_marker_T_cm}")
                        # ------------------------------------------------------------
                        
                        # Calculate board Euler angles
                        euler_angles = cv2.RQDecomp3x3(R)[0]
                        
                        # Add board pose information text
                        text_lines = [
                            f"{board_type} Board Pose:",
                            f"X: {ref_tvec[0,0]:.3f}m Y: {ref_tvec[1,0]:.3f}m Z: {ref_tvec[2,0]:.3f}m",
                            f"Roll: {euler_angles[0]:.1f}° Pitch: {euler_angles[1]:.1f}° Yaw: {euler_angles[2]:.1f}°"
                        ]
                        
                        # Position text at top-left corner of image
                        text_x = 25  # Fixed x position from left edge
                        text_y_start = 25  # Starting y position from top

                        # Adjust y position based on board type to prevent overlap
                        if board_type == "start":
                            text_y = text_y_start
                        else:  # target board
                            text_y = text_y_start + 70  # Offset for second board's text

                        for i, line in enumerate(text_lines):
                            cv2.putText(
                                cv_image,
                                line,
                                (text_x, text_y + i * 20),  # 20 pixels spacing between lines
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0) if board_type == "start" else (0, 0, 255),  # Green for start, Red for target
                                2
                            )
                    else:
                        print(f"Failed to calculate board transform for {board_type} board")
                        continue
                else:
                    print(f"Need exactly 2 markers, but found {len(board_markers.keys())}")
                    ref_rvec, ref_tvec = None, None

                ref_id = min(board_markers.keys())
                original_rvec, original_tvec = old_board_tvec_markers[ref_id]
                try:
                    # Add board type text using the first marker's corners from board_markers
                    first_marker_id = current_marker_ids[0]
                    first_marker_corners = board_markers[first_marker_id][2]  # [2] index gets the corners
                    
                    cv2.putText(
                        cv_image,
                        f"{board_type} Board",
                        (int(first_marker_corners[0][0][0]), int(first_marker_corners[0][0][1]) - 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0) if board_type == "start" else (0, 0, 255),
                        2
                    )
                except Exception as e:
                    print(f"Error adding board type text: {e}")
                
                # Process holes for this board
                if not offset_version:
                    holes_camera_frame[board_type] = self.process_holes(
                        cv_image,
                        board_type,
                        ref_rvec,
                        # original_rvec,
                        ref_tvec,
                        # original_tvec
                    )
                else:
                    holes_camera_frame[board_type] = self.process_holes_offset_vesrsion(
                        cv_image,
                        board_type,
                        ref_rvec,
                        ref_tvec
                    )
        else:
            print("No ArUco marker detected in this frame\n")
        
        cv2.imshow('ArUco Markers', cv_image)
        cv2.waitKey(1)
        # input("Press Enter to close the window with camera image...")  # Keep the program running
        
        return holes_camera_frame

    def process_holes(self, cv_image, board_type, ref_rvec, ref_tvec) -> List[np.ndarray]:
        """
        Process holes for a given board type and reference marker pose.
        
        Args:
            cv_image: Image to draw visualizations on
            board_type: 'start' or 'target'
            ref_rvec: Reference marker rotation vector
            ref_tvec: Reference marker translation vector
            avg_z: Average Z component from all markers
        
        Returns:
            List of transformation matrices (T_ch) for each hole
        """
        processed_holes = []
        
        # Convert rvec to rotation matrix
        ref_R, _ = cv2.Rodrigues(ref_rvec)
        # Make sure translation is a column vector
        ref_t = ref_tvec.reshape((3, 1))

        # --- Compute and visualize hole positions ---
        for idx, (off_x_mm, off_y_mm) in enumerate(self.hole_offsets_mm[board_type]):
            # Convert mm to meters
            off_x_m = off_x_mm / 1000.0
            off_y_m = off_y_mm / 1000.0
            
            try:
                # Hole position in the local (marker) coordinate system (z=0 assumed)
                hole_local_t = np.array([[off_x_m], [off_y_m], [0.0]], dtype=np.float32)
                
                # Transform hole to camera coordinate system: hole_in_camera = R * hole_local + t
                hole_in_camera = ref_R @ hole_local_t + ref_t
                
                # Create T_ch (transformation from hole to camera)
                T_ch = np.eye(4)
                T_ch[:3, :3] = ref_R  # Rotation from marker (same as hole)
                T_ch[:3, 3] = (hole_in_camera).flatten()
                
                # NOTE: CALCULATE T_rh ONLY TO visualize coords in robot frame
                # ------------------------------------------------------------
                T_rh = config.T_RC @ T_ch
                
                # Adjust the marker frame to align with robot's preferred orientation
                R_align = np.array([
                    [1,  0,  0],
                    [0, -1,  0],
                    [0,  0, -1]
                ])
                
                T_rh = T_rh.copy()
                T_rh[:3, :3] = T_rh[:3, :3] @ R_align
                
            except Exception as e:
                print(f"Error processing hole {idx}: {e}")


            # Visualization code
            self._visualize_hole(
                cv_image, 
                hole_local_t, 
                ref_rvec, 
                ref_tvec, 
                off_x_m, 
                off_y_m, 
                T_ch, 
                T_rh, 
                ref_R
            )
            # ------------------------------------------------------------
                
            # NOTE: append T_ch to processed_holes, not T_rh
            processed_holes.append(T_ch.copy())
                
        
        return processed_holes
    



    def process_holes_offset_vesrsion(self, cv_image, board_type, ref_rvec, ref_tvec) -> List[np.ndarray]:
        """
        Process holes for a given board type and reference marker pose.
        
        Args:
            cv_image: Image to draw visualizations on
            board_type: 'start' or 'target'
            ref_rvec: Reference marker rotation vector
            ref_tvec: Reference marker translation vector
            avg_z: Average Z component from all markers
        
        Returns:
            List of transformation matrices (T_ch) for each hole
        """
        processed_holes_offset_vesrsion = []
        
        # Convert rvec to rotation matrix
        ref_R, _ = cv2.Rodrigues(ref_rvec)
        # Make sure translation is a column vector
        ref_t = ref_tvec.reshape((3, 1))

        # --- Compute and visualize hole positions ---
        for idx, (off_x_mm, off_y_mm) in enumerate(self.hole_offsets_mm[board_type]):
            # Convert mm to meters
            off_x_m = off_x_mm / 1000.0
            off_y_m = off_y_mm / 1000.0
            
            try:
                # Hole position in the local (marker) coordinate system (z=0 assumed)
                hole_local_t = np.array([[off_x_m], [off_y_m], [0.1]], dtype=np.float32)
                
                # Transform hole to camera coordinate system: hole_in_camera = R * hole_local + t
                hole_in_camera = ref_R @ hole_local_t + ref_t
                
                # Create T_ch (transformation from hole to camera)
                T_ch = np.eye(4)
                T_ch[:3, :3] = ref_R  # Rotation from marker (same as hole)
                T_ch[:3, 3] = (hole_in_camera).flatten()
                
            #     # NOTE: CALCULATE T_rh ONLY TO visualize coords in robot frame
            #     # ------------------------------------------------------------
            #     T_rh = config.T_RC @ T_ch
                
            #     # Adjust the marker frame to align with robot's preferred orientation
            #     R_align = np.array([
            #         [1,  0,  0],
            #         [0, -1,  0],
            #         [0,  0, -1]
            #     ])
                
            #     T_rh = T_rh.copy()
            #     T_rh[:3, :3] = T_rh[:3, :3] @ R_align
                
            except Exception as e:
                print(f"Error processing hole offset version {idx}: {e}")


            # # Visualization code
            # self._visualize_hole(
            #     cv_image, 
            #     hole_local_t, 
            #     ref_rvec, 
            #     ref_tvec, 
            #     off_x_m, 
            #     off_y_m, 
            #     T_ch, 
            #     T_rh, 
            #     ref_R
            # )
            # ------------------------------------------------------------
                
            # NOTE: append T_ch to processed_holes, not T_rh
            processed_holes_offset_vesrsion.append(T_ch.copy())
                
        
        return processed_holes_offset_vesrsion
    



    def _visualize_hole(self, cv_image, hole_local_t, ref_rvec, ref_tvec, 
                       off_x_m, off_y_m, T_ch, T_rh, hole_rot_mat):
        """Helper method to handle hole visualization."""
        # Draw hole boundary
        half_size = (config.CUBE_HOLE_SIZE / 2)
        square_corners = np.array([
            [-half_size,  half_size, 0], # top-left
            [ half_size,  half_size, 0], # top-right
            [ half_size, -half_size, 0],  # bottom-right
            [-half_size, -half_size, 0], # bottom-left
        ], dtype=np.float32)
        try:
            # Transform square corners to hole's position
            square_corners_local = square_corners + np.array([off_x_m, off_y_m, 0])
            square_image_points, _ = cv2.projectPoints(
                square_corners_local,
                ref_rvec,
                ref_tvec,
                config.CAMERA_MATRIX,
                config.DIST_COEFFS
            )
        except Exception as e:
            print(f"Error projecting points for visualizing hole: {e}")
        
        try:
            # Convert hole rotation to Euler angles
            hole_euler = cv2.RQDecomp3x3(hole_rot_mat)[0]
        
            # Draw the square
            square_points = square_image_points.reshape(-1, 2).astype(np.int32)
            cv2.polylines(cv_image, [square_points], True, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error drawing square for visualizing hole: {e}")
        
        try:
            # Setup text position
            # projectPoints returns shape (N, 1, 2) where N is number of points
            text_x = int(square_image_points[0, 0, 0]) + 10  # First point, x coordinate
            text_y = int(square_image_points[0, 0, 1])       # First point, y coordinate
            line_spacing = 20
        except Exception as e:
            print(f"Error setting text position for visualizing hole: {e}")
        
        try:
            # Draw position texts and Euler angles
            pos_text_camera = f"Camera: X={float(T_ch[0,3]):.3f}m Y={float(T_ch[1,3]):.3f}m Z={float(T_ch[2,3]):.3f}m"
            cv2.putText(cv_image, pos_text_camera, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            pos_text_robot = f"Robot: X={float(T_rh[0,3]):.3f}m Y={float(T_rh[1,3]):.3f}m Z={float(T_rh[2,3]):.3f}m"
            cv2.putText(cv_image, pos_text_robot, (text_x, text_y + line_spacing),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            euler_text = f"Roll={hole_euler[0]:.1f}° Pitch={hole_euler[1]:.1f}° Yaw={hole_euler[2]:.1f}°"
            cv2.putText(cv_image, euler_text, (text_x, text_y + line_spacing * 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        except Exception as e:
            print(f"Error drawing coords and Euler angles for visualizing hole: {e}")
            

    def detect_and_process_markers(self, cv_image, photo_number=None) -> Tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect ArUco markers and process their poses.
        
        Args:
            cv_image: Input image
            photo_number: Current photo number for multiple photos mode
        
        Returns:
            Tuple of (marker_poses, processed_image, corners, ids)
            - marker_poses: Dictionary mapping marker IDs to their (rvec, tvec)
            - processed_image: Image with visualizations
            - corners: Detected marker corners
            - ids: Detected marker IDs
        """
        marker_poses = {}
        
        # Detect markers
        corners, ids, _ = self.detector.detectMarkers(cv_image)
        
        if ids is not None and len(ids) > 0:
            # Draw detected markers
            cv_image = aruco.drawDetectedMarkers(cv_image.copy(), corners, ids)
            
            # Process each detected marker
            for i, marker_id in enumerate(ids):
                marker_id = marker_id[0]  # Extract scalar value
                objPoints = np.array([
                    [-self.marker_size/2,  self.marker_size/2, 0], # top-left
                    [ self.marker_size/2,  self.marker_size/2, 0], # top-right
                    [ self.marker_size/2, -self.marker_size/2, 0], # bottom-right
                    [-self.marker_size/2, -self.marker_size/2, 0]  # bottom-left
                ], dtype=np.float32)
                
                success, rvec, tvec = cv2.solvePnP(
                    objPoints,
                    corners[i],
                    config.CAMERA_MATRIX,
                    config.DIST_COEFFS
                )
                
                if success:
                    display_rvec, display_tvec = (rvec, tvec)
                    
                    # Draw axes and store poses
                    cv2.drawFrameAxes(
                        cv_image,
                        config.CAMERA_MATRIX,
                        config.DIST_COEFFS,
                        display_rvec,
                        display_tvec,
                        self.marker_size/2
                    )
                    
                    marker_poses[marker_id] = (display_rvec, display_tvec, corners[i])
                    
                    # Visualize marker information
                    self._visualize_marker_info(
                        cv_image,
                        marker_id,
                        corners[i],
                        display_rvec,
                        display_tvec,
                        photo_number=photo_number
                    )
                else:
                    print(f"Error solving PnP for marker {marker_id}")
        else:
            print("No ArUco marker detected in this frame\n")
        
        return marker_poses, cv_image, corners, ids

    def _visualize_marker_info(self, cv_image, marker_id, corners, rvec, tvec, euler_angles=None, photo_number=None):
        """
        Visualize marker position and orientation information.
        
        Args:
            cv_image: Image to draw on
            marker_id: ID of the marker
            corners: Marker corners in image
            rvec: Rotation vector
            tvec: Translation vector
            euler_angles: Euler angles
            photo_number: Current photo number for multiple photos mode
        """
        # Convert rotation vector to Euler angles
        position = tvec.flatten()
        if euler_angles is None:
            rot_mat, _ = cv2.Rodrigues(rvec)
            euler_angles = cv2.RQDecomp3x3(rot_mat)[0]
        if photo_number is not None:
            print(f"\nMarker {marker_id} - Photo {photo_number if photo_number is not None else 'single'}:")
            print(f"Position: X={position[0]:.3f}m Y={position[1]:.3f}m Z={position[2]:.3f}m")
            print(f"Rotation: Roll={euler_angles[0]:.1f}° Pitch={euler_angles[1]:.1f}° Yaw={euler_angles[2]:.1f}°")
        else:
            # Draw position text
            pos_text = f"ID {marker_id}: X={position[0]:.3f}m Y={position[1]:.3f}m Z={position[2]:.3f}m"
            cv2.putText(
                cv_image,
                pos_text,
                (int(corners[0][0][0]), int(corners[0][0][1]) - 10),  # Fixed corner coordinate access
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
            
            # Draw rotation text
            euler_text = f"Roll={euler_angles[0]:.1f}° Pitch={euler_angles[1]:.1f}° Yaw={euler_angles[2]:.1f}°"
            cv2.putText(
                cv_image,
                euler_text,
                (int(corners[0][0][0]), int(corners[0][0][1]) - 30),  # Fixed corner coordinate access
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2
            )


    def calculate_board_transform(self, marker_corners: dict, start_board_id, second_marker_id , camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the transform of the board relative to the camera using two ArUco markers.

        Args:
            marker_poses (dict): Dictionary mapping marker IDs to their (rvec, tvec).
            camera_matrix (np.ndarray): Camera calibration matrix.
            dist_coeffs (np.ndarray): Distortion coefficients.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (rvec, tvec) representing the rotation and translation
                                        of the board relative to the camera frame. Returns None if
                                        required markers are not found or solvePnP fails.
        """
        if start_board_id not in marker_corners.keys() or second_marker_id not in marker_corners.keys():
            print(f"Required markers ({start_board_id}, {second_marker_id}) for board transform are not detected.")
            return None, None

        # Get rvec and tvec for the two reference markers
        ref_corners = marker_corners[start_board_id].reshape(4, 2)
        second_corners = marker_corners[second_marker_id].reshape(4, 2)

        # Convert marker rvec and tvec to corner points in the image
        marker_size = self.marker_size

        # Combine all points into a single array
        # img_points = np.vstack([ref_corners.squeeze(axis=1), second_corners.squeeze(axis=1)])
        img_points = np.vstack([ref_corners, second_corners])

        # Define the corresponding 3D points in the board's coordinate system
        marker_size = self.marker_size
        ref_points = np.array([
            [-marker_size/2, marker_size/2, 0],  # top-left
            [marker_size/2, marker_size/2, 0],   # top-right
            [marker_size/2, -marker_size/2, 0],  # bottom-right
            [-marker_size/2, -marker_size/2, 0]  # bottom-left
        ])

        # NOTE: offset should be set in meters
        second_points = ref_points + np.array([0.18, 0.14, 0])  # Adjust for the offset between the markers

        obj_points = np.vstack([ref_points, second_points])

        # Ensure correct data types and shapes
        obj_points = obj_points.astype(np.float32)
        img_points = img_points.astype(np.float32)

        # SolvePnP to calculate the board's pose
        success, rvec, tvec = cv2.solvePnP(
            obj_points,
            img_points,
            config.CAMERA_MATRIX,
            config.DIST_COEFFS,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            print("Failed to calculate board transform.")
            return None, None

        # print(f"Board rvec: {rvec.flatten()}, tvec: {tvec.flatten()}")
        return rvec, tvec



def main():
    """Modified main function to handle multiple photos."""

    camera = BaslerCamera()
    camera.connect_by_name(config.CAMERA_NAME)
    camera.open()
    camera.set_parameters()
    
    aruco_node = CameraArUcoNode(config.START_BOARD_CSV, config.TARGET_BOARD_CSV)
    

    # x = aruco_node.camera_callback(camera)

    camera.close()

if __name__ == "__main__":
    main()
