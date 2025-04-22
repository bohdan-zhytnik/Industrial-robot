#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_camera.py: Experimental scripts for testing
image acquisition.
"""

__author__ = "Pavel Krsek"
__maintainer__ = "Pavel Krsek"
__email__ = "pavel.krsek@cvut.cz"
__copyright__ = "Copyright \xa9 2024 RMP, CIIRC CVUT in Prague\nAll Rights Reserved."
__license__ = "Use for lectures B3B33ROB1"
__version__ = "1.0"
__date__ = "2024/10/30"
__status__ = "Development"
__credits__ = []
__all__ = []

# OpenCV library for image processing
import cv2
import argparse
# Our Basler camera interface
from basler_camera import BaslerCamera
from detect_holes import CameraArUcoNode


import config

def main() -> None:
    camera: BaslerCamera = BaslerCamera()

    # Camera can be connected based on its' IP or name:
    # Camera for robot CRS 93
    # camera.connect_by_ip("192.168.137.107")
    print("test0")
    camera.connect_by_name(config.CAMERA_NAME)
    # Camera for robot CRS 97
    #   camera.connect_by_ip("192.168.137.106")
    #   camera.connect_by_name("camera-crs97")
    print("test1")
    # Open the communication with the camera
    camera.open()
    # Set capturing parameters from the camera object.
    # The default parameters (set by constructor) are OK.
    # When starting the params should be send into the camera.
    print("test2")
    camera.set_parameters()

    # Take one image from the camera
    print("test3")
    img = camera.grab_image()
    # If the returned image has zero size,
    # the image was not captured in time.

    csv_file_path = "csvs/positions_plate_start.csv"

    camera_aruco_node = CameraArUcoNode(csv_file_path, csv_file_path, 0)

    x = camera_aruco_node.camera_callback(camera)

    # Close communication with the camera before finish.
    camera.close()


if __name__ == '__main__':
    main()
