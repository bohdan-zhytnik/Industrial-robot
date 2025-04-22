from ctu_crs import CRS93
import numpy as np

robot = CRS93()


from ctu_crs import CRS97
import numpy as np

robot = CRS97()

robot.soft_home()
np.atan2 = np.arctan2
robot.initialize()




q = robot.get_q()
robot.move_to_q(q + [0.1, 0.0, 0.0, 0.0, 0.0, 0.0])  # move robot all values in radians
robot.wait_for_motion_stop() # wait until the robot stops
robot.close()  # close the connection




>>> robot.initialize()
>>> robot.move_to_q(q_start)
>>> q_start = q_start + np.deg2rad([-5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
>>> robot.move_to_q(q_start)
>>> np.save("calib_poses/crs93_many_poses/q_start.npy", q_start)
>>> q_start1 = np.load("calib_poses/crs93_many_poses/q_start.npy")
>>> q = robot.get_q()
>>> q = q + np.deg2rad([-5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
>>> robot.move_to_q(q)
>>> q_start = np.load("calib_poses/crs93_many_poses/q_start.npy")
>>> robot.move_to_q(q_start)
>>> q = robot.get_q()
>>> q = q + np.deg2rad([-35.0, 0.0, 0.0, 0.0, 0.0, 0.0])
>>> robot.move_to_q(q)
>>> q = robot.get_q()
>>> q = q + np.deg2rad([-15.0, 0.0, 0.0, 0.0, 0.0, 0.0])
>>> robot.move_to_q(q)
>>> robot.move_to_q(q)
>>> q = robot.get_q()
>>> q = q + np.deg2rad([5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
>>> robot.move_to_q(q)
>>> q = robot.get_q()
>>> q = q + np.deg2rad([5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
>>> robot.move_to_q(q)
>>> q = robot.get_q()
>>> T1 = robot.fk(q)
>>> T2= T1.copy()
>>> T2[2,3]= 
KeyboardInterrupt
>>> T2[2,3] = T1[2,3] +0.05
>>> q_sol_list = robot.ik(T2)
>>> from main import get_best_solution
Imported camera matrix =
[[4.69607316e+03 0.00000000e+00 9.72587700e+02]
 [0.00000000e+00 4.69098069e+03 5.17126680e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
Imported distortion coefficients =
[[-9.57166235e-02 -6.34633398e+00  1.09537794e-03  9.46827899e-04
   7.52701413e+01]]
Imported original T_rc =
[[ 0.00186797  0.99791308 -0.06454455  0.49270554]
 [ 0.99983796 -0.00301943 -0.01774671  0.00470818]
 [-0.01790456 -0.06450094 -0.99775701  1.1849499 ]
 [ 0.          0.          0.          1.        ]]
Firmware version : MARS8 v1.0 build Pi Oct  2 2017 11:06:45
>>> q_best = main.get_best_solution(q_sol_list, robot)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'main' is not defined. Did you mean: 'min'?
>>> import main
>>> q_best = main.get_best_solution(q_sol_list, robot)
get_best_solution function: elif best_sol is None case (that is for the test purpose)
get_best_solution function: elif best_sol is None case (that is for the test purpose)
get_best_solution function: elif best_sol is None case (that is for the test purpose)
get_best_solution function: elif best_sol is None case (that is for the test purpose)
>>> print(q_best)
[-0.36459754 -1.34993371 -1.59552472  3.14113    -1.40863842 -0.04216842]
>>> robot.in_limits(q_best)
True
>>> robot.set_speed(robot._min_speed_irc256_per_ms)
>>> robot.move_to_q(q_best)
>>> 

import main
q_start = np.load("calib_poses/crs93_many_poses/q_start.npy")
T_start = robot.fk(q_start)
T_start_shift = T_start.copy()
T_start_shift[2,3] = T_start[2,3] + 0.1
robot.soft_home()
q_sol_list = robot.ik(T_start_shift)
q_best = main.get_best_solution(q_sol_list, robot)
robot.set_speed(robot._min_speed_irc256_per_ms)
robot.move_to_q(q_best)