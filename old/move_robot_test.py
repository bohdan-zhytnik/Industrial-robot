import config

if config.ROBOT_NAME == "crs93":
    from ctu_crs import CRS93
    robot = CRS93()  # set argument tty_dev=None if you are not connected to robot,
elif config.ROBOT_NAME == "crs97":
    from ctu_crs import CRS97
    robot = CRS97()  #
robot.initialize()

import numpy as np
# Initialize the robot (make sure the robot is connected and properly configured)

robot.initialize(home=True)

# Generate 5 random joint positions within the robot's working zone
# random_positions = generate_random_joint_positions(robot, n_positions=5)

# # Move the robot to each of these positions
# move_robot_to_positions(robot, random_positions)


def generate_random_joint_positions(robot: CRS93, n_positions: int = 5):
    """
    Generate n random joint configurations within the robot's joint limits.
    The configurations are validated to ensure they are within the working zone.

    Args:
        robot (CRSRobot): The robot object with joint limit information.
        n_positions (int): Number of random positions to generate.

    Returns:
        list[np.ndarray]: A list of valid joint configurations (in radians).
    """
    random_positions = []

    for _ in range(n_positions):
        # Generate random joint values within limits
        q_random = np.random.uniform(low=robot.q_min, high=robot.q_max)
        random_positions.append(q_random)

    return random_positions

def move_robot_to_positions(robot: CRS93, positions: list[np.ndarray]):
    """
    Move the robot to a list of joint configurations.

    Args:
        robot (CRSRobot): The robot object to move.
        positions (list[np.ndarray]): A list of joint configurations (in radians).
    """
    for i, q in enumerate(positions):
        print(f"Moving to position {i+1}: {q}")
        robot.move_to_q(q)
        robot.wait_for_motion_stop()
        print(f"Reached position {i+1}.\n")

# Example usage
random_positions = generate_random_joint_positions(robot, n_positions=5)

# Move the robot to each of these positions
move_robot_to_positions(robot, random_positions)
robot.soft_home()

    # Close the robot connection after movement
robot.close()
