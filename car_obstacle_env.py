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


from objects import World, RaySensor, Robot, Obstacle

class CarObstacleEnv(object):

    """CarObstacleEnv."""

    def __init__(self, horizon=2000):
        """Constructs the simulator.
        """

        self.create_world()
        self.create_obstacles()
        self.create_target()
        self.create_robot()
        self.update_locator()

        self._app = ConsoleApp()
        self._view = self._app.createView(useGrid=False)

        self._timestep = 0
        self._horizon = horizon

        # performance tracker
        self._num_targets = 0
        self._num_crashes = 0
        self._run_ticks = 0



    def run(self):
        self._timer = TimerCallback(targetFps=120)
        self._timer.callback = self.step
        self._timer.start()

        self._app.start()

        # for i_episode in range(20):
        #     observation = env.reset()
        #     for t in range(100):
        #         env.render()
        #         print(observation)
        #         action = np.random.choice([-np.pi / 2, 0., np.pi / 2], 1)[0]
        #         observation, reward, done, info = env.step(action)
        #         if done:
        #             print("Episode finished after {} timesteps".format(t + 1))
        #             break

    def tick(self):
        action = np.random.choice([-np.pi / 2, 0., np.pi / 2], 1)[0]
        self.step(action)

    def step(self, action):
        info = None

        self.update_obstacle()

        robot, frame = self._robot

        for sensor in robot.sensors:
            sensor.set_locator(self.locator)

        # actions = [-np.pi / 2, 0., np.pi / 2]
        obs, reward, done = robot.move(action)

        # if self._timestep >= self._horizon:
        #     done = True

        self._update_object_pose(robot, frame)

        self._timestep += 1
        return obs, reward, done, info

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

        # color = [0.4, 0.85098039, 0.9372549]
        frame_name = "robot"
        frame = self._add_polydata(robot.to_polydata(), frame_name, color)
        self._robot = (robot, frame)
        self._update_object_pose(robot, frame)

    def create_obstacles(self):
        self._obstacles = []
        for obstacle in self._world.generate_obstacles(0.01, 0):
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

    def run(self, display):
        """Launches and displays the simulator.

        Args:
            display: Displays the simulator or not.
        """
        if display:
            widget = QtGui.QWidget()
            layout = QtGui.QVBoxLayout(widget)
            layout.addWidget(self._view)
            widget.showMaximized()

            # Set camera.
            applogic.resetCamera(viewDirection=[0.2, 0, -1])

        # Set timer.
        self._tick_count = 0
        self._timer = TimerCallback(targetFps=120)
        self._timer.callback = self.tick
        self._timer.start()

        self._app.start()

    def tick(self):
        """Update simulation clock."""
        self._tick_count += 1
        self._run_ticks += 1
        if self._tick_count >= 500:
            print("timeout")
            for robot, frame in self._robots:
                self.reset(robot, frame)

        need_update = False
        for obstacle, frame in self._obstacles:
            if obstacle.velocity != 0.:
                obstacle.move()
                self._update_object_pose(obstacle, frame)
                need_update = True

        if need_update:
            self.update_locator()

        for i, (robot, frame) in enumerate(self._robots):
            self._update_object_pose(robot, frame)
            for sensor in robot.sensors:
                sensor.set_locator(self.locator)
            robot.move()
            for sensor in robot.sensors:
                frame_name = "rays{}".format(i)
                self._update_sensor(sensor, frame_name)
                if sensor.has_collided():
                    self._num_crashes += 1
                    print("collided", min(d for d in sensor._distances if d > 0))
                    print("targets hit", self._num_targets)
                    print("ticks lived", self._run_ticks)
                    print("deaths", self._num_crashes)
                    self._run_ticks = 0
                    self._num_targets = 0
                    new_target = self.generate_random_position()
                    for robot, frame in self._robots:
                        robot.set_target(new_target)
                    self.add_target(new_target)
                    self.reset(robot, frame)

            if robot.at_target():
                self._num_targets += 1
                self._tick_count = 0
                new_target = self.generate_random_position()
                for robot, frame in self._robots:
                    robot.set_target(new_target)
                self.add_target(new_target)

    def generate_random_position(self):
        return tuple(np.random.uniform(-75, 75, 2))

    def init_robot_position(self, robot):
        while True:
            robot.x, robot.y = self.generate_random_position()
            robot.theta = np.random.uniform(0, 2 * np.pi)
            if min(robot.sensors[0].distances) >= 0.30:
                return

    def reset_robot(self):
        self._robot[0].set_target(self._target_pos)
        self._tick_count = 0
        self.init_robot_position(self._robot[0])
        self._update_object_pose(self._robot[0], self._robot[1])
        # self._robot[0]._ctrl.save()

    def update_obstacle(self):
        for obstacle, frame in self._obstacles:
            if obstacle.velocity != 0.:
                obstacle.move()
                self._update_object_pose(obstacle, frame)
                self.update_locator()