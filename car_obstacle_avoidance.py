import numpy as np
from director_env_wrapper import DirectorEnvWrapper
from objects import Robot, RaySensor
from car_obstacle_env import Simulator

import argparse
import numpy as np

from PythonQt import QtGui
from net import Controller
from director import applogic
from director import vtkAll as vtk
from director import objectmodel as om
from director.debugVis import DebugData
from director import visualization as vis
from director.consoleapp import ConsoleApp
from director.timercallback import TimerCallback

class CarObstacleAvoidance(DirectorEnvWrapper):
    def __init__(self):
        self.tick = 0
        self.tick_thresh = 500

        world = World(200, 200)
        self.sim = Simulator(world) # For visualization

        # Add obstacles & locator
        for obstacle in world.generate_obstacles(0.01, 0):
            self.sim.add_obstacle(obstacle)
        self.sim.update_locator()

        # Add target
        target = self.sim.generate_random_position()
        self.sim.add_target(target)

        # Add robot
        self.robot = Robot() # Agent object
        self.robot.attach_sensor(RaySensor())
        self.sim.create_robot(self.robot)

    def add_robot(self, robot):
        """Adds a robot to the simulation.

        Args:
            robot: Robot.
        """
        color = [0.4, 0.85098039, 0.9372549]
        frame_name = "robot{}".format(len(self._robots))
        frame = self._add_polydata(robot.to_polydata(), frame_name, color)
        self._robots.append((robot, frame))
        self._update_moving_object(robot, frame)

    def init_robot_position(self):
        while True:
            self.robot.x, self.robot.y = tuple(np.random.uniform(-75, 75, 2))
            self.robot.theta = np.random.uniform(0, 2 * np.pi)
            if min(self.robot.sensors[0].distances) >= 0.30:
                return

    def step(self, action):
        if self.tick >= self.tick_thresh:
            print("timeout")
            self.sim.reset()

        self.sim.update_obstacle()
        reward, done = self.sim.update_robot()



        self.tick += 1

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass