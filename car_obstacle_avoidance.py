import numpy as np
from director_env_wrapper import DirectorEnvWrapper
from robot_env import Robot, RaySensor

import argparse
import numpy as np
from world import World
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
        self.robot = Robot()
        self.robot.set_target(self.generate_position())
        self.robot.attach_sensor(RaySensor())
        self.set_safe_position(self.robot)

    def generate_position(self):
        return tuple(np.random.uniform(-75, 75, 2))

    def set_safe_position(self, robot):
        while True:
            robot.x, robot.y = self.generate_position()
            robot.theta = np.random.uniform(0, 2 * np.pi)
            if min(robot.sensors[0].distances) >= 0.30:
                return

    def add_target(self, target):
        data = DebugData()
        center = [target[0], target[1], 1]
        axis = [0, 0, 1]  # Upright cylinder.
        data.addCylinder(center, axis, 2, 3)
        om.removeFromObjectModel(om.findObjectByName("target"))
        self._add_polydata(data.getPolyData(), "target", [0, 0.8, 0])

    def step(self, action):
        prev_state = self.robot._state

        gamma = 0.9
        prev_xy = self._state[0], self._state[1]
        prev_state = self._get_state()
        prev_utilities = self._ctrl.evaluate(prev_state)
        super(Robot, self).move(dt)
        next_state = self._get_state()
        next_utilities = self._ctrl.evaluate(next_state)
        print("action: {}, utility: {}".format(
            self._selected_i, prev_utilities[self._selected_i]))

        terminal = self._sensors[0].has_collided()
        curr_reward = self._get_reward(prev_xy)
        total_reward =\
            curr_reward if terminal else \
            curr_reward + gamma * next_utilities[self._selected_i]
        rewards = [total_reward if i == self._selected_i else prev_utilities[i]
                   for i in range(len(next_utilities))]
        print("-----------------------------")
        print(prev_state)
        print(rewards)
        print("-----------------------------")
        self._ctrl.train(prev_state, rewards)

        prev_dx = self.robot._target[0] - prev_state[0]
        prev_dy = self.robot._target[1] - prev_state[1]
        prev_distance = np.sqrt(prev_dx ** 2 + prev_dy ** 2)
        new_dx = self.robot._target[0] - self.robot._state[0]
        new_dy = self.robot._target[1] - self.robot._state[1]
        new_distance = np.sqrt(new_dx ** 2 + new_dy ** 2)
        if self.robot._sensors[0].has_collided():
            return -20
        elif self.robot.at_target():
            return 15
        else:
            delta_distance = prev_distance - new_distance
            angle_distance = -abs(self.robot._angle_to_destination()) / 4
            obstacle_ahead = self.robot._sensors[0].distances[8] - 1
            return delta_distance + angle_distance + obstacle_ahead


    def reset(self):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def at_target(self, threshold=3):
        """Return whether the robot has reached its target.

        Args:
            threshold: Target distance threshold.

        Returns:
            True if target is reached.
        """
        return (abs(self._state[0] - self._target[0]) <= threshold and
                abs(self._state[1] - self._target[1]) <= threshold)

    def _get_reward(self, prev_state):
        prev_dx = self._target[0] - prev_state[0]
        prev_dy = self._target[1] - prev_state[1]
        prev_distance = np.sqrt(prev_dx ** 2 + prev_dy ** 2)
        new_dx = self._target[0] - self._state[0]
        new_dy = self._target[1] - self._state[1]
        new_distance = np.sqrt(new_dx ** 2 + new_dy ** 2)
        if self._sensors[0].has_collided():
            return -20
        elif self.at_target():
            return 15
        else:
            delta_distance = prev_distance - new_distance
            angle_distance = -abs(self._angle_to_destination()) / 4
            obstacle_ahead = self._sensors[0].distances[8] - 1
            return delta_distance + angle_distance + obstacle_ahead

    def _angle_to_destination(self):
        x, y = self._target[0] - self.x, self._target[1] - self.y
        return self._wrap_angles(np.arctan2(y, x) - self.theta)

    def _wrap_angles(self, a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def _get_state(self):
        dx, dy = self._target[0] - self.x, self._target[1] - self.y
        curr_state = [dx / 1000, dy / 1000, self._angle_to_destination()]
        return np.hstack([curr_state, self._sensors[0].distances])
