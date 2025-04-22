def calculate_board_transform(self, aruco_corners: dict, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> SE3:
        """Calculate board transform from ArUco marker transforms using solvePnP.
        
        Args:
            aruco_corners: Dict mapping marker IDs to their corners and IDs
            camera_matrix: Camera calibration matrix
            dist_coeffs: Distortion coefficients
            
        Returns:
            SE3 transform of the board in camera frame
        """
        if self.ref_marker_id not in aruco_corners or self.second_marker_id not in aruco_corners:
            return None
            
        # Get corners for both markers
        ref_corners = aruco_corners[self.ref_marker_id]
        second_corners = aruco_corners[self.second_marker_id]
        
        # Define 3D points for both markers in board coordinate system
        marker_size = self.MARKER_SIZE
        
        # First marker at origin
        ref_points = np.array([
            [-marker_size/2, marker_size/2, 0],  # top-left
            [marker_size/2, marker_size/2, 0],   # top-right
            [marker_size/2, -marker_size/2, 0],  # bottom-right
            [-marker_size/2, -marker_size/2, 0]  # bottom-left
        ])
        
        # Second marker offset by [180, 140] mm
        second_points = ref_points + np.array([180, 140, 0])
        
        # Combine all points
        obj_points = np.vstack([ref_points, second_points])
        img_points = np.vstack([ref_corners, second_corners])
        
        # Solve PnP with all 8 points
        success, rvec, tvec = cv2.solvePnP(
            obj_points, img_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return None
            
        # Convert to SE3 transform
        R, _ = cv2.Rodrigues(rvec)
        self.board_transform = SE3(
            translation=tvec.flatten(),
            rotation=SO3(rotation_matrix=R)
        )
        
        return self.board_transform