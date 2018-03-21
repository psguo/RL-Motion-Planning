import numpy as np
from director import vtkAll as vtk
from director.debugVis import DebugData
from director import ioUtils, filterUtils
import math

class MovingObject(object):

    """Moving object."""

    def __init__(self, velocity, polydata):
        """Constructs a MovingObject.

        Args:
            velocity: Velocity.
            polydata: Polydata.
        """
        self._state = np.array([0., 0., 0.]) #x, y, orientation
        self._velocity = float(velocity)
        self._raw_polydata = polydata
        self._polydata = polydata
        self._sensor = None

    @property
    def x(self):
        """X coordinate."""
        return self._state[0]

    @x.setter
    def x(self, value):
        """X coordinate."""
        next_state = self._state.copy()
        next_state[0] = float(value)
        self._update_state(next_state)

    @property
    def y(self):
        """Y coordinate."""
        return self._state[1]

    @y.setter
    def y(self, value):
        """Y coordinate."""
        next_state = self._state.copy()
        next_state[1] = float(value)
        self._update_state(next_state)

    @property
    def theta(self):
        """Yaw in radians."""
        return self._state[2]

    @theta.setter
    def theta(self, value):
        """Yaw in radians."""
        next_state = self._state.copy()
        next_state[2] = float(value) % (2 * np.pi)
        self._update_state(next_state)

    @property
    def velocity(self):
        """Velocity."""
        return self._velocity

    @velocity.setter
    def velocity(self, value):
        """Velocity."""
        self._velocity = float(value)

    @property
    def sensor(self):
        """List of attached sensors."""
        return self._sensor

    def attach_sensor(self, sensor):
        """Attaches a sensor.

        Args:
            sensor: Sensor.
        """
        self._sensor = sensor

    # def _dynamics(self, state, t, action, controller=None):
    #     """Dynamics of the object.

    #     Args:
    #         state: Initial condition.
    #         t: Time.

    #     Returns:
    #         Derivative of state at t.
    #     """
    #     dqdt = np.zeros_like(state)
    #     dqdt[0] = self._velocity * np.cos(state[2])
    #     dqdt[1] = self._velocity * np.sin(state[2])
    #     dqdt[2] = self._control(state, t, action)
    #     return dqdt * t

    def _dynamics(self, state, action):
        """Dynamics of the object.

        Args:
            state: Initial condition.
            t: Time.

        Returns:
            Derivative of state at t.
        """
        dqdt = np.zeros_like(state)
        dqdt[0] = self._velocity * np.cos(action)
        dqdt[1] = self._velocity * np.sin(action)
        # dqdt[2] = action
        # if action == 0:
        #     dqdt[0] = self._velocity * np.cos(state[2])
        #     dqdt[1] = self._velocity * np.sin(state[2])
        # else:
        #     dqdt[0] = 0
        #     dqdt[1] = 0
        # dqdt[2] = action
        return dqdt / 30.0

    def _control(self, state, t, action):
        """Returns the yaw given state.

        Args:
            state: State.
            t: Time.

        Returns:
            Yaw.
        """
        raise NotImplementedError

    def _simulate(self, action):
        """Simulates the object moving.

        Args:
            dt: Time length of step.

        Returns:
            New state.
        """
        self._state[2] = action
        self._state = self._state + self._dynamics(self._state, action)

        return self._state

        return self._state + self._dynamics(self._state, action)

    def move(self, action):
        """Moves the object by a given time step.

        Args:
            dt: Length of time step.
        """
        state = self._simulate(action)
        self._update_state(state)

    def _update_state(self, next_state):
        """Updates the moving object's state.

        Args:
            next_state: New state.
        """
        t = vtk.vtkTransform()
        t.Translate([next_state[0], next_state[1], 0.])
        t.RotateZ(np.degrees(next_state[2]))
        self._polydata = filterUtils.transformPolyData(self._raw_polydata, t)
        self._state = next_state
        if self._sensor is not None:
            self._sensor.update(*self._state)

    def to_positioned_polydata(self):
        """Converts object to visualizable poly data.

        Note: Transformations have been already applied to this.
        """
        return self._polydata

    def to_polydata(self):
        """Converts object to visualizable poly data.

        Note: This is centered at (0, 0, 0) and is not rotated.
        """
        return self._raw_polydata


class Robot(MovingObject):

    """Robot."""

    def __init__(self, velocity=25.0, scale=0.15):
        self._target = (0, 0)

        data = DebugData()
        center = [0, 0, 0]
        axis = [0, 0, 1]
        height = 0.1
        radius = 2
        data.addCylinder(center, axis, height, radius)
        polydata = data.getPolyData()
        super(Robot, self).__init__(velocity, polydata)

    def move(self, action):
        prev_xy = self._state[0], self._state[1]
        super(Robot, self).move(action)

        if self._sensor.has_collided() or self.at_target():
        # if self._sensor.has_collided():
            done = True
        else:
            done = False

        return self._get_state(), self._get_reward(prev_xy), done, prev_xy


    def set_target(self, target):
        self._target = target

    def at_target(self, threshold=3):
        return (abs(self._state[0] - self._target[0]) <= threshold and
                abs(self._state[1] - self._target[1]) <= threshold)

    def _get_reward(self, prev_state):
        prev_dx = self._target[0] - prev_state[0]
        prev_dy = self._target[1] - prev_state[1]
        prev_distance = np.sqrt(prev_dx ** 2 + prev_dy ** 2)
        new_dx = self._target[0] - self._state[0]
        new_dy = self._target[1] - self._state[1]
        new_distance = np.sqrt(new_dx ** 2 + new_dy ** 2)
        if self._sensor.has_collided():
            return -40
        elif self.at_target():
            return 15
        else:
            delta_distance = prev_distance - new_distance
            # angle_distance = -abs(self._angle_to_destination()) / 4
            # obstacle_ahead = self._sensor.distances[8] - 1
            # delta_distance = -new_distance
            return delta_distance

        # if self._sensor.has_collided():
        #     return -1000
        # elif self.at_target():
        #     return 1000
        # else:
        #     return -1

        # if self._sensor.has_collided():
        #     return 0
        # else:
        #     return 1

    def _angle_to_destination(self):
        x, y = self._target[0] - self.x, self._target[1] - self.y
        return self._wrap_angles(np.arctan2(y, x) - self.theta)

    def _wrap_angles(self, a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def _get_state(self):
        dx, dy = self._target[0] - self.x, self._target[1] - self.y
        curr_state = [dx / 1000, dy / 1000, self._angle_to_destination()]
        return np.hstack([curr_state, self._sensor.distances])

    def get_pos(self):
        return self._state[:2]

    def _control(self, state, t, action):
        return action

class Obstacle(MovingObject):

    """Obstacle."""

    def __init__(self, velocity, radius, bounds, height=1.0):
        """Constructs a Robot.

        Args:
            velocity: Velocity of the robot in the forward direction.
            radius: Radius of the obstacle.
        """
        data = DebugData()
        self._bounds = bounds
        self._radius = radius
        self._height = height
        center = [0, 0, height / 2 - 0.5]
        axis = [0, 0, 1]  # Upright cylinder.
        data.addCylinder(center, axis, height, radius)
        polydata = data.getPolyData()
        super(Obstacle, self).__init__(velocity, polydata)

    def _control(self, state, t, action=0):
        """Returns the yaw given state.

        Args:
            state: State.
            t: Time.

        Returns:
            Yaw.
        """
        x_min, x_max, y_min, y_max = self._bounds
        x, y, theta = state
        if x - self._radius <= x_min:
            return np.pi
        elif x + self._radius >= x_max:
            return np.pi
        elif y - self._radius <= y_min:
            return np.pi
        elif y + self._radius >= y_max:
            return np.pi
        return 0.


class RaySensor(object):

    """Ray sensor."""

    def __init__(self, num_rays=10, radius=40, min_angle=-45, max_angle=45):
        """Constructs a RaySensor.

        Args:
            num_rays: Number of rays.
            radius: Max distance of the rays.
            min_angle: Minimum angle of the rays in degrees.
            max_angle: Maximum angle of the rays in degrees.
        """
        self._num_rays = num_rays
        self._radius = radius
        self._min_angle = math.radians(min_angle)
        self._max_angle = math.radians(max_angle)

        self._locator = None
        self._state = [0., 0., 0.]  # x, y, theta

        self._hit = np.zeros(num_rays)
        self._distances = np.zeros(num_rays)
        self._intersections = [[0, 0, 0] for i in range(num_rays)]

        self._update_rays(self._state[2])

    @property
    def distances(self):
        """Array of distances measured by each ray."""
        normalized_distances = [
            self._distances[i] / self._radius if self._hit[i] else 1.0
            for i in range(self._num_rays)
        ]
        return normalized_distances

    def has_collided(self, max_distance=0.05):
        """Returns whether a collision has occured or not.

        Args:
            max_distance: Threshold for collision distance.
        """
        for hit, distance in zip(self._hit, self._distances):
            if hit and distance <= max_distance:
                return True

        return False

    def set_locator(self, locator):
        """Sets the vtk cell locator.

        Args:
            locator: Cell locator.
        """
        self._locator = locator

    def update(self, x, y, theta):
        """Updates the sensor's readings.

        Args:
            x: X coordinate.
            y: Y coordinate.
            theta: Yaw.
        """
        self._update_rays(theta)
        origin = np.array([x, y, 0])
        self._state = [x, y, theta]

        if self._locator is None:
            return

        for i in range(self._num_rays):
            hit, dist, inter = self._cast_ray(origin, origin + self._rays[i])
            self._hit[i] = hit
            self._distances[i] = dist
            self._intersections[i] = inter

    def _update_rays(self, theta):
        """Updates the rays' readings.

        Args:
            theta: Yaw.
        """
        r = self._radius
        angle_step = (self._max_angle - self._min_angle) / (self._num_rays - 1)
        self._rays = [
            np.array([
                r * math.cos(theta + self._min_angle + i * angle_step),
                r * math.sin(theta + self._min_angle + i * angle_step),
                0
            ])
            for i in range(self._num_rays)
        ]

    def _cast_ray(self, start, end):
        """Casts a ray and determines intersections and distances.

        Args:
            start: Origin of the ray.
            end: End point of the ray.

        Returns:
            Tuple of (whether it intersected, distance, intersection).
        """
        tolerance = 0.0                 # intersection tolerance
        pt = [0.0, 0.0, 0.0]            # coordinate of intersection
        distance = vtk.mutable(0.0)     # distance of intersection
        pcoords = [0.0, 0.0, 0.0]       # location within intersected cell
        subID = vtk.mutable(0)          # subID of intersected cell

        hit = self._locator.IntersectWithLine(start, end, tolerance,
                                              distance, pt, pcoords, subID)

        return hit, distance, pt

    def to_polydata(self):
        """Converts the sensor to polydata."""
        d = DebugData()
        origin = np.array([self._state[0], self._state[1], 0])
        for hit, intersection, ray in zip(self._hit,
                                          self._intersections,
                                          self._rays):
            if hit:
                color = [1., 0.45882353, 0.51372549]
                endpoint = intersection
            else:
                color = [0., 0.6, 0.58823529]
                endpoint = origin + ray

            d.addLine(origin, endpoint, color=color, radius=0.05)

        return d.getPolyData()


class World(object):

    """Base world."""

    def __init__(self, width, height):
        """Construct an empty world.

        Args:
            width: Width of the field.
            height: Height of the field.
        """
        self._data = DebugData()

        self._width = width
        self._height = height
        self._add_boundaries()

    def _add_boundaries(self):
        """Adds boundaries to the world."""
        self._x_max, self._x_min = self._width / 2, -self._width / 2
        self._y_max, self._y_min = self._height / 2, -self._height / 2

        corners = [
            (self._x_max, self._y_max, 0),  # Top-right corner.
            (self._x_max, self._y_min, 0),  # Bottom-right corner.
            (self._x_min, self._y_min, 0),  # Bottom-left corner.
            (self._x_min, self._y_max, 0)   # Top-left corner.
        ]

        # Loopback to begining.
        corners.append(corners[0])

        for start, end in zip(corners, corners[1:]):
            self._data.addLine(start, end, radius=0.2)

    def generate_obstacles(self, density=0.05, moving_obstacle_ratio=0.20,
                           seed=None):
        """Generates randomly scattered obstacles to the world.

        Args:
            density: Obstacle to world area ratio, default: 0.1.
            moving_obstacle_ratio: Ratio of moving to stationary obstacles,
                default: 0.2.
            seed: Random seed, default: None.

        Yields:
            Obstacle.
        """
        if seed is not None:
            np.random.seed(seed)

        field_area = self._width * self._height
        obstacle_area = int(field_area * density)

        bounds = self._x_min, self._x_max, self._y_min, self._y_max
        while obstacle_area > 0:
            radius = np.random.uniform(3.0, 5.0)
            center_x_range = (self._x_min + radius, self._x_max - radius)
            center_y_range = (self._y_min + radius, self._y_max - radius)
            center_x = np.random.uniform(*center_x_range)
            center_y = np.random.uniform(*center_y_range)
            theta = np.random.uniform(0., 360.)
            obstacle_area -= np.pi * radius ** 2

            # Only some obstacles should be moving.
            if np.random.random_sample() >= moving_obstacle_ratio:
                velocity = 0.0
            else:
                velocity = np.random.uniform(-30.0, 30.0)

            obstacle = Obstacle(velocity, radius, bounds)
            obstacle.x = center_x
            obstacle.y = center_y
            obstacle.theta = np.radians(theta)
            yield obstacle

    def to_polydata(self):
        """Converts world to visualizable poly data."""
        return self._data.getPolyData()
