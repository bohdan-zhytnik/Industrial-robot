#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

import numpy as np
from nptyping import NDArray, Shape
# dataclass for camera class definition
from dataclasses import dataclass
# camera interface libraries
from pypylon import pylon
# import type Any
from typing import Any


# ==========================================================================
#
#         ***    CAMERA CLASS    ***
#
# ==========================================================================
@dataclass(init=False)
class BaslerCamera:
    """Class represents the basledr camera  interface"""

    #: Camera IP address as string
    ip_address: str
    #: Camera gama correction (default is 1.0)
    gamma: float
    #: Camera gain (default is 0.0, switch off)
    gain: float
    #: Camera exposure time [ms]]
    #: (default value 0.0 mean auto exposure on)
    exposure_time: float
    #: Image capture framerate (frames per sec.)
    #: (default value 0.0 mean automatic / maximal)
    frame_rate: float
    #: Time out for obtaining the image from camera [ms]
    grab_timeout: int
    #: List of attributes to be save to config file.
    #: The attributes are exported/ imported as dictionary
    #: by methodss get_as_dict and set_from_dict.
    config_attrs: list[str]

    def __init__(self):
        self.camera: Any = None
        self.converter: Any = None
        self.connected: bool = False
        self.opened: bool = False
        self.ip_address = ""
        self.gamma = 1.0
        self.gain = 0.0
        self.frame_rate = 0
        self.exposure_time = 0.0
        self.grab_timeout = 1000
        self.config_attrs = [
            "ip_address",
            "grab_timeout",
            "exposure_time",
            "frame_rate",
            "gamma",
            "gain"
        ]

    def get_as_dict(self) -> dict[str, Any]:
        """
        The method returns the class parameters as a dictionary.
        This method is used for writing data into the configuration file.
        Default implementation should be reimplemented.

        Returns:
            dict[str, Any]:
                Parameters names and values in the form of a dictionary.
        """

        ret_dict = {}
        for key in self.config_attrs:
            if hasattr(self, key):
                ret_dict[key] = getattr(self, key)
            else:
                raise AttributeError(f"Missing configuration attribute {key}")
        return ret_dict

    def set_from_dict(self, data: dict[str, Any]) -> None:
        """
        The method sets the class parameters from the dictionary.
        This method is used for reading data from the configuration file.
        Default implementation should be reimplemented.

        Args:
            data (dict[str, Any]):
                Parameters names and values in the form of a dictionary.
        """
        for key, value in data.items():
            if (key in self.config_attrs) & hasattr(self, key):
                setattr(self, key, value)

    def set_parameters(self):
        """
        The method sets the camera parameters (in the camera) by values storeed
        in the corresponding attributes of this class. The attribustes must be
        set to desired values before setting the parameters in the camera.
        """
        self.camera.Gamma.SetValue(self.gamma)
        self.camera.GainAuto.SetValue("Off")
        self.camera.Gain.SetValue(int(self.gain))
        if self.exposure_time > 0:
            self.camera.ExposureAuto.SetValue("Off")
            self.camera.ExposureTime.SetValue(self.exposure_time)
        else:
            self.camera.ExposureAuto.SetValue("Continuous")
        if self.frame_rate > 0:
            self.camera.AcquisitionFrameRateEnable.SetValue(True)
            self.camera.AcquisitionFrameRate.SetValue(self.frame_rate)
        else:
            self.camera.AcquisitionFrameRateEnable.SetValue(False)

    # Informations about devicer/ camera
    # print("----")
    # print(device.GetDeviceClass())
    # print(device.GetModelName())
    # print(device.GetSerialNumber())
    # print(device.GetUserDefinedName())
    # print(device.GetFullName())
    # print(device.GetFriendlyName())

    def connect_by_ip(self, ip_addr: str = ""):
        """
        The method conects the camera by its IP address.
        The methods lists cameras nearby and selects
        the camera with appropriate IP address.
        In case of error the exeptions are raised.

        Args:
            ip_addr(str):
                The desired IP address of the camera.
        """
        self.opened = False
        self.connected = False
        self.camera = None
        if ip_addr != "":
            self.ip_address = ip_addr
        if self.ip_address != "":
            tl_factory = pylon.TlFactory.GetInstance()
            devices = tl_factory.EnumerateDevices()
            if len(devices) == 0:
                error_str = "There is not detected any Basler camera."
                raise pylon.RuntimeException(error_str)
            for device in devices:
                success = device.GetIpAddress() == self.ip_address
                if success:
                    break
            if not success:
                error_str = (f"Camera with IP address {self.ip_address}" +
                             " not found.")
                raise pylon.RuntimeException(error_str)
            self.camera = pylon.InstantCamera(tl_factory.CreateDevice(device))
            # The parameter MaxNumBuffer can be used to control the count
            # of buffers allocated for grabbing. The default value of this
            # parameter is 10, but 5 is enough where only last image is used.
            self.camera.MaxNumBuffer = 5
            self.connected = True
        else:
            error_str = "IP address is nod deffined."
            raise TypeError(error_str)

    def connect_by_name(self, name: str = ""):
        """
        The method conects the camera by its user defined name.
        The methods lists cameras nearby and selects
        the camera with appropriate name.
        In case of error the exeptions are raised.

        Args:
            name(str):
                The desired name of the camera.
        """
        self.opened = False
        self.connected = False
        self.camera = None
        if name != "":
            tl_factory = pylon.TlFactory.GetInstance()
            devices = tl_factory.EnumerateDevices()
            if len(devices) == 0:
                error_str = "There is not detected any Basler camera."
                raise pylon.RuntimeException(error_str)
            for device in devices:
                success = device.GetUserDefinedName() == name
                if success:
                    break
            if not success:
                error_str = (f"Camera with IP address {self.ip_address}" +
                             " not found.")
                raise pylon.RuntimeException(error_str)
            self.camera = pylon.InstantCamera(tl_factory.CreateDevice(device))
            # The parameter MaxNumBuffer can be used to control the count
            # of buffers allocated for grabbing. The default value of this
            # parameter is 10, but 5 is enough where only last image is used.
            self.camera.MaxNumBuffer = 5
            self.connected = True
        else:
            error_str = "IDevice name is nod deffined."
            raise TypeError(error_str)

    def open(self):
        """
        The method opens comunication with connected camera and
        prepare covnverter for converting the image from camera
        into the format accepted by OpenCV (cv2).
        """
        if self.connected:
            self.camera.Open()
            self.opened = True
        if self.opened:
            # Converting to opencv bgr format
            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = \
                pylon.PixelType_BGR8packed
            self.converter.OutputBitAlignment = \
                pylon.OutputBitAlignment_MsbAligned

    def __start(self):
        """
        The method starts image capturing when the camera is connected
        and the comunication is open.
        """
        
        # Temporarily hidden as internal method.
        # Start image grabbing causes continuous data sending.
        # This situation, together with a high (maximal) framerate,
        # may lead to overflowing the network and blocking other
        # cameras on the same network. To prevent this, the users
        # should use only the function grab_images that starts and stops
        # grabbing for each image separately.
        
        if self.opened:
            if not self.camera.IsGrabbing():
                # Start grabbing images
                self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def get_image(self, time_out: int = 0) -> NDArray[Shape["*, *, 3"], Any]:
        """
        The method obtain one image from the camera. The method waits for
        the image until the image is obtained or timeout is reached.

        Args:
            time_out(int):
                Timeout for the image obtaining [ms]. When the value 0
                is used the timeout defined by corresponding attribute
                of camera object is applied.

        Returns:
            NDArray[Shape["row, col, 3"], Any]:
                The obtained image is returned. When the image has size 0,
                means that the image was not obtained within the timeout
                or other error ocured.
        """
        image = np.array([])  # camera interface libraries
        if self.camera.IsGrabbing():
            # Wait for an image and then retrieve it.
            # A minimal timeout of 1000 ms is recommended.
            # We must take into accont that first image takes longer time.
            if time_out <= 0:
                time_out = int(self.grab_timeout)
            res = self.camera.RetrieveResult(
                time_out,
                pylon.TimeoutHandling_ThrowException)
            # Image grabbed successfully?
            # Information about error: {res.ErrorCode}, {res.ErrorDescription}
            if res.GrabSucceeded():
                # Access the image data.
                image = self.converter.Convert(res)
                image = image.GetArray()
            res.Release()
        return image

    def grab_image(self, time_out: int = 0) -> NDArray[Shape["*, *, 3"], Any]:
        """
        The method is enccapsulation of the method get_image. This method
        chceck if the grabing of images is started. If not, the grabbing
        is started and than the image is obtained. State of grabbing is
        set back to the state before calling the method.

        Args:
            time_out(int):
                Timeout for the image obtaining [ms]. When the value 0
                is used the timeout defined by corresponding attribute
                of camera object is applied.

        Returns:
            NDArray[Shape["row, col, 3"], Any]:
                The obtained image is returned. When the image has size 0,
                means that the image was not obtained within the timeout
                or other error ocured.
        """
        stop_grab: bool = False
        if not self.camera.IsGrabbing():
            # Start grabbing images
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            stop_grab = True
        image = self.get_image(time_out)
        if stop_grab:
            # Stop grabbing images
            self.camera.StopGrabbing()
        return image

    def __stop(self):
        """
        The method stops capturing of images.
        """
        
        # Temporarily hiden as internal method.
        # See method __start(self).

        if self.opened:
            if self.camera.IsGrabbing():
                # Stop grabbing images
                self.camera.StopGrabbing()

    def close(self):
        """
        The method close communication with the camera.
        """
        if self.opened:
            if self.camera.IsGrabbing():
                # Stop grabbing images
                self.camera.StopGrabbing()
            self.opened = False
            self.camera.Close()
