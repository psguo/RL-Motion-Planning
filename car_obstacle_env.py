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
from plot import Plot

class CarObstacleEnv(object):

    """CarObstacleEnv."""

    def __init__(self, horizon=2000, render=False, start_render=200, test_interval=50):
        """Constructs the simulator.
        """

        # np.random.seed(2)
        self.start_render = start_render
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
        # self.actions = [-np.pi / 2, np.pi / 2]
        self.actions = np.linspace(0, 2 * np.pi, num=10, endpoint=False)

        # init agent
        self._agent = DuelingAgent(n_actions=len(self.actions), n_features=30+3,
            lr=0.001, gamma=0.95, e_start=0.5, e_end=0.05, 
            e_decay=5e-6, replace_iter=200, memory_size=50000, batch_size=32)

        self.rewards = []
        self.plot = Plot(env_name='Car Obstacle')
        self.test_interval = test_interval
        self.test_count = 0
        self.is_test = False
        self.success_test = 0
        self.test_results = []
        self.test_epss = []
        self.test_total_count = 100

        self.test_step = 0
        self.max_horizon = 1000

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

                if self._i_eps % self.test_interval == 0:
                    self.is_test = True

                if self._i_eps > self.start_render:
                    np.asarray(self.rewards).dump("rewards.dat")
                    self._agent.save_param()
                    self.plot.plot_success(eps=self.test_epss, success=self.test_results, normalization=self.test_total_count)
                    self._render = True
                    self._activate_rendering()

    def tick(self):
        if self.is_test:
            action = self._agent.greedy_policy(self._obs)
            obs_, reward, done, curxy, info = self.step(action)
            prevxy = info['prevxy']
            if self._render:
                self.update_path(curxy, prevxy)
            self._obs = obs_

            if self.test_step >= self.max_horizon:
                done = True
                self.test_step = 0

            if done:
                if info["is_at_target"]:
                    self.success_test += 1
                self._obs = self.reset()
                self.test_count += 1
                self._eps_iter = 0
                if self._render:
                    self.clear_path()

                if self.test_count % int(self.test_total_count/5) == 0:
                    print "Test progress: ", self.test_count, '/', self.test_total_count

                if self.test_count == self.test_total_count:
                    self._i_eps += 1
                    self.test_count = 0
                    self.is_test = False
                    self.test_results.append(float(self.success_test) / self.test_total_count)
                    self.test_epss.append(self._i_eps)
                    print 'Test Result: ', self.success_test, '/', self.test_total_count
                    print self.test_results
                    self.success_test = 0

            self.test_step += 1

        else:
            # cur_time = time.time()
            # print 1000*(cur_time - self.last_time)
            # self.last_time = cur_time
            action = self._agent.epsilon_greedy_policy(self._obs)
            obs_, reward, done, curxy, info = self.step(action)
            prevxy = info['prevxy']
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
                self.rewards.append(self._eps_reward)
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

        obs, reward, done, info = robot.move(self.actions[action])
        # print reward
        # ttt = time.time()
        if self._render:
            self._update_sensor(robot.sensor, "rays")
        # print 1000*(time.time()-ttt)

        # if self._timestep >= self._horizon:
        #     done = True

        self._update_object_pose(robot, frame)

        self._timestep += 1
        return obs, reward, done, robot.get_pos(), info

    def reset(self):
        self._timestep = 0
        self.reset_obstacles()
        self.create_target()
        self.reset_robot()
        self.update_locator()
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
        target_robot = Robot()  # Target object
        target_robot.attach_sensor(RaySensor())
        self._target_pos = self.init_robot_position(target_robot)

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

    def reset_obstacles(self):
        for obstacle,frame in self._obstacles:
            om.removeFromObjectModel(om.findObjectByName(frame))
        self.create_obstacles()

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
            # print "============================="
            # print robot.sensor.distances
            if min(robot.sensor.distances) >= 0.3:
                # print "success"
                return (robot.x, robot.y)
            # print "Fail"

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