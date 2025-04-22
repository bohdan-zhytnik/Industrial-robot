# GigaRobot
GigaRobot.


## Install dependencies

install libs in requirements without installing ctu_crs
```bash
pip install -r requirements.txt
```

then install ctu_crs

```bash
pip install -r ctu_crs==1.0.2
```

this will install ctu_crs with numpy>2.0.0. Then uninstall numpy, and install numpy==1.26.4

```bash
pip uninstall numpy
pip install numpy==1.26.4
```

You are ready to go. You should have these libs installed now:

*
ctu_crs               1.0.2
ctu_mars_control_unit 0.1.3
nptyping              2.5.0
numpy                 1.26.4
opencv-contrib-python 4.10.0.84
opencv-python         4.10.0.84
pip                   24.2
pypylon               4.1.0
pyserial              3.5
PyYAML                6.0.2
*

