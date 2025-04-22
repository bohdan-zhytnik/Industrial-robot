from basler_camera import BaslerCamera
from camera_aruco_node import SingleMarkerDetector
import cv2

import config

camera: BaslerCamera = BaslerCamera()

camera.connect_by_name(config.CAMERA_NAME)
camera.open()
camera.set_parameters()

csv_file_path = "csvs/positions_plate_03-04.csv"

camera_aruco_node = SingleMarkerDetector()

img = camera.grab_image()

rvec, tvec = camera_aruco_node.get_rvec_tvec_of_first_marker(img)

print("rvec: \n", rvec)
print("tvec: \n", tvec)

cv2.namedWindow('Camera image', cv2.WINDOW_NORMAL)
cv2.imshow('Camera image', img)
cv2.waitKey(0)



camera.close()
