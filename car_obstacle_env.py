# -*- coding: utf-8 -*-

import argparse
import numpy as np
from PythonQt import QtGui
from director import applogic
from director import vtkAll as vtk
from director import objectmodel as om
from director.debugVis import DebugData
from director import visualization as vis
from director.consoleapp import ConsoleApp
from director.timercallback import TimerCallback
from dueling import DuelingAgent
from objects import World, RaySensor, Robot, Obstacle
import time

class CarObstacleEnv(object):

    """CarObstacleEnv."""

    def __init__(self, horizon=2000, render=False):
        """Constructs the simulator.
        """

        # np.random.seed(2)
        self._app = ConsoleApp()
        self._view = self._app.createView(useGrid=False)

        self.create_world()
        self.create_obstacles()
        self.create_target()
        self.create_robot()
        self.update_locator()

        self._path_data = DebugData()

        self._timestep = 0
        self._horizon = horizon

        # performance tracker
        self._num_targets = 0
        self._num_crashes = 0
        self._run_ticks = 0
        self._render = render
        self.actions = [-np.pi / 2, 0., np.pi / 2]

        # init agent
        self._agent = DuelingAgent(n_actions=3, n_features=13,
            lr=0.001, gamma=0.95, e_start=0.5, e_end=0.05, 
            e_decay=5e-6, replace_iter=200, memory_size=50000, batch_size=32)

    def _activate_rendering(self):
        widget = QtGui.QWidget()
        layout = QtGui.QVBoxLayout(widget)
        layout.addWidget(self._view)
        widget.showMaximized()

        applogic.resetCamera(viewDirection=[0.2, 0, -1])
        self._timer = TimerCallback(targetFps=120)
        self._timer.callback = self.tick
        self._timer.start()

        self._app.start()

    def run(self):
        # reset env
        self._obs = self.reset()
        self._eps_reward = 0
        self._eps_iter = 0
        self._running_reward =0
        self._i_eps = 1

        if self._render:
            self._activate_rendering()
        else:
            # self.last_time = time.time()
            while True:
                self.tick()
                if self._i_eps > 200:
                    self._render = True
                    self._activate_rendering()
            


    def tick(self):
        # cur_time = time.time()
        # print 1000*(cur_time - self.last_time)
        # self.last_time = cur_time
        action = self._agent.epsilon_greedy_policy(self._obs)
        obs_, reward, done, curxy, prevxy = self.step(action)
        if self._render:
            self.update_path(curxy, prevxy)
        self._eps_reward += reward
        self._eps_iter += 1
        # print('----------------------')
        # print(self._obs, action, reward, obs_, done)
        self._agent.train(self._obs, action, reward, obs_, done)
        self._obs = obs_
        if done:
            self._obs = self.reset()
            self._running_reward = self._running_reward*0.95 + 0.05*self._eps_reward
            print '----------------'
            print 'Episode: ', self._i_eps
            print 'Survive: ', self._eps_iter
            print 'Running reward: ', self._running_reward
            print 'Episode reward: ',self._eps_reward
            print 'Epsilon: ', self._agent.epsilon
            self._i_eps += 1
            self._eps_iter = 0
            self._eps_reward = 0
            if self._render:
                self.clear_path()

    def step(self, action):
        self.update_obstacle()

        robot, frame = self._robot

        robot.sensor.set_locator(self.locator)

        obs, reward, done, prevxy = robot.move(self.actions[action])
        # print reward
        # ttt = time.time()
        if self._render:
            self._update_sensor(robot.sensor, "rays")
        # print 1000*(time.time()-ttt)

        # if self._timestep >= self._horizon:
        #     done = True

        self._update_object_pose(robot, frame)

        self._timestep += 1
        return obs, reward, done, robot.get_pos(), prevxy

    def reset(self):
        self._timestep = 0
        self.create_target()
        self.reset_robot()
        return self._robot[0]._get_state()

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def _add_polydata(self, polydata, frame_name, color):
        """Adds polydata to the simulation.

        Args:
            polydata: Polydata.
            frame_name: Frame name.
            color: Color of object.

        Returns:
            Frame.
        """
        om.removeFromObjectModel(om.findObjectByName(frame_name))
        frame = vis.showPolyData(polydata, frame_name, color=color)

        vis.addChildFrame(frame)
        return frame

    def create_world(self):
        self._world = World(200, 200)
        """Initializes the world."""
        # Add world to view.
        om.removeFromObjectModel(om.findObjectByName("world"))
        vis.showPolyData(self._world.to_polydata(), "world")

    def update_path(self, curxy, prevxy):
        start = list(prevxy) + [0]
        end = list(curxy) + [0]
        color = [1, 1, 0]
        self._path_data.addLine(start, end, radius=0.1)
        vis.updatePolyData(self._path_data.getPolyData(), "path", 
                           color=color)

    def clear_path(self):
        om.removeFromObjectModel(om.findObjectByName("path"))
        self._path_data = DebugData()

    def create_target(self):
        self._target_pos = self.generate_random_position()
        data = DebugData()
        center = [self._target_pos[0], self._target_pos[1], 1]
        axis = [0, 0, 1]  # Upright cylinder.
        data.addCylinder(center, axis, 2, 3)
        om.removeFromObjectModel(om.findObjectByName("target"))
        self._add_polydata(data.getPolyData(), "target", [0, 0.8, 0])

    def create_robot(self):
        """Adds a robot to the simulation.

        Args:
            robot: Robot.
        """
        robot = Robot() # Agent object
        robot.attach_sensor(RaySensor())
        robot.set_target(self._target_pos)

        self.init_robot_position(robot)

        color = [0.4, 0.85098039, 0.9372549]
        frame_name = "robot"
        frame = self._add_polydata(robot.to_polydata(), frame_name, color)
        self._robot = (robot, frame)
        self._update_object_pose(robot, frame)

    def create_obstacles(self):
        self._obstacles = []
        for obstacle in self._world.generate_obstacles(0.04, 0):
            """Adds an obstacle to the simulation.
    
            Args:
                obstacle: Obstacle.
            """
            color = [1.0, 1.0, 1.0]
            frame_name = "obstacle{}".format(len(self._obstacles))
            frame = self._add_polydata(obstacle.to_polydata(), frame_name, color)
            self._obstacles.append((obstacle, frame))
            self._update_object_pose(obstacle, frame)

    def _update_object_pose(self, moving_object, frame):
        """Updates moving object's state.

        Args:
            moving_object: Moving object.
            frame: Corresponding frame.
        """
        t = vtk.vtkTransform()
        t.Translate(moving_object.x, moving_object.y, 0.0)
        t.RotateZ(np.degrees(moving_object.theta))
        frame.getChildFrame().copyFrame(t)

    def _update_sensor(self, sensor, frame_name):
        """Updates sensor's rays.

        Args:
            sensor: Sensor.
            frame_name: Frame name.
        """
        vis.updatePolyData(sensor.to_polydata(), frame_name,
                           colorByName="RGB255")

    def update_locator(self):
        """Updates cell locator."""
        d = DebugData()

        d.addPolyData(self._world.to_polydata())
        for obstacle, frame in self._obstacles:
            d.addPolyData(obstacle.to_positioned_polydata())

        self.locator = vtk.vtkCellLocator()
        self.locator.SetDataSet(d.getPolyData())
        self.locator.BuildLocator()

    def generate_random_position(self):
        return tuple(np.random.uniform(-75, 75, 2))

    def init_robot_position(self, robot):
        while True:
            robot.x, robot.y = self.generate_random_position()
            robot.theta = np.random.uniform(0, 2 * np.pi)
            if min(robot.sensor.distances) >= 0.30:
                return

    def reset_robot(self):
        self._robot[0].set_target(self._target_pos)
        self._tick_count = 0
        self.init_robot_position(self._robot[0])
        self._update_object_pose(self._robot[0], self._robot[1])

    def update_obstacle(self):
        for obstacle, frame in self._obstacles:
            if obstacle.velocity != 0.:
                obstacle.move()
                self._update_object_pose(obstacle, frame)
                self.update_locator()