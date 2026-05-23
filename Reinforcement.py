from http.cookiejar import reach
from time import sleep

import gymnasium
from dask.array import square
from gymnasium import spaces

import numpy as np
import cv2
import random
import time
import ImageReading
import os
import math
from collections import deque


def collision_with_goal(snake_head, goal):
    if snake_head[0] == goal[0] and snake_head[1] == goal[1]:
        return 1
    else:
        return 0


def collision_with_boundaries(snake_head):
    if snake_head[0] >= 500 or snake_head[0] < 0 or snake_head[1] >= 500 or snake_head[1] < 0:
        return 1
    else:
        return 0
def collision_with_walls(snake_head, wallcount, wall_x, wall_y):
    for i in range(0,wallcount):
        if snake_head[0] == wall_x[i] and snake_head[1] == wall_y[i]:
            return 1
    return 0



class LabyrinthEnv(gymnasium.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self):
        super(LabyrinthEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(4)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-500, high=500,
                                            shape=(3,), dtype=np.float32)
        self.best_distance_score = 30
        self.distance_to_target = 30

        """self.past_distances = []
        self.past_distances.append(13)""" # Done olduğunda uzaklığa göre reward

    def compute_bfs_distances(self):
        """Precomputes the actual path distance from every valid tile to the goal."""
        goal_x, goal_y = self.goal[0], self.goal[1]
        distances = {(goal_x, goal_y): 0}
        queue = deque([(goal_x, goal_y, 0)])
        walls = set(zip(self.wall_x, self.wall_y))

        while queue:
            x, y, dist = queue.popleft()
            # Check 4 grid directions (50 pixels step)
            for dx, dy in [(-50, 0), (50, 0), (0, -50), (0, 50)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < 500 and 0 <= ny < 500:
                    if (nx, ny) not in walls and (nx, ny) not in distances:
                        distances[(nx, ny)] = dist + 1
                        queue.append((nx, ny, dist + 1))
        return distances


    def step(self, action):
        self.prev_actions.insert(0,action)

        self.img = np.zeros((500, 500, 3), dtype='uint8')
        # Display Walls
        for j in range(0, len(self.wall_x)):
            cv2.rectangle(self.img, (self.wall_x[j], self.wall_y[j]), (self.wall_x[j] + 50, self.wall_y[j] + 50), (255, 255, 255), 3)

        # Display Goal
        cv2.rectangle(self.img, (self.goal[0], self.goal[1]), (self.goal[0] + 50, self.goal[1] + 50),
                      (0, 0, 255), 3)
        # Display Snake
        cv2.rectangle(self.img, (self.position[0], self.position[1]), (self.position[0] + 50, self.position[1] + 50), (0, 255, 0), 3)
        cv2.imshow('a', self.img)
        cv2.waitKey(10)

        if action == 0:
            self.position[0] -= 50
        elif action == 1:
            self.position[0] += 50
        elif action == 2:
            self.position[1] -= 50
        elif action == 3:
            self.position[1] += 50



        # On collision kill the snake and print the message
        if collision_with_boundaries(self.position) == 1:
            font = cv2.FONT_HERSHEY_SIMPLEX
            self.img = np.zeros((500, 500, 3), dtype='uint8')
            cv2.putText(self.img, 'You hit the boundaries', (140, 250), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('a', self.img)
            cv2.waitKey(20)
            self.done = True

        # On collision with wall kill the snake
        if collision_with_walls(self.position, len(self.wall_x), self.wall_x, self.wall_y) == 1:
            font = cv2.FONT_HERSHEY_SIMPLEX
            self.img = np.zeros((500, 500, 3), dtype='uint8')
            cv2.putText(self.img, 'You hit a wall', (140, 250), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('a', self.img)
            cv2.waitKey(20)
            self.done = True
        if collision_with_goal(self.position, self.goal) == 1:
            font = cv2.FONT_HERSHEY_SIMPLEX
            img = np.zeros((500, 500, 3), dtype='uint8')
            cv2.putText(img, 'You reached the goal!', (140, 250), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('a', img)
            self.win = True
            cv2.waitKey(3000)

        position_x = self.position[0]
        position_y = self.position[1]


        self.distance_to_target = self.distance_map.get((position_x, position_y), 30)
        self.reached_distance = self.distance_to_target
        #distance_to_target = math.sqrt(square(goal_delta_y/50) + square(goal_delta_x/50))

        self.observation = [position_x, position_y, self.distance_to_target]
        self.observation = np.array(self.observation, dtype=np.float32)
        if self.done == True:
            self.reward = -10

        elif self.done == False:
            """if self.prev_distances[0] - self.distance_to_target > 0 and self.distance_to_target < self.best_distance_score:
                self.reward = (self.best_distance_score - self.distance_to_target +3) * 5"""
            if self.prev_distances[0] - self.distance_to_target > 0:
                self.reward = 5
            else:
                self.reward = -5
            self.prev_distances.insert(0, self.distance_to_target)
        print(self.reward)


        #En düşük ulaşılan mesafeyi tut
        print(f"Best:{self.best_distance_score}")
        print(f"Reached:{self.reached_distance}")
        if self.best_distance_score > self.reached_distance:
            self.best_distance_score = self.reached_distance
            print(f"Best:{self.best_distance_score}")


        if self.win == True:
            self.reward = 100


        info = {}
        self.truncated = False
        return self.observation, self.reward, self.done, self.truncated, info

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.done = False
        self.win = False
        self.reward = 0
        self.img = np.zeros((500, 500, 3), dtype='uint8')
        # Initial Snake and goal position
        root = os.path.dirname(os.path.abspath(__file__))
        self.Img1 = ImageReading.Image(root + "\\Labyrinths\\WideLabyrinth1.png")
        self.walldata, self.position, self.goal = self.Img1.ImageReader()
        self.initial_position = self.position.copy()
        self.walldata = self.walldata.astype(int) * 50
        self.wall_x = self.walldata[0, :].tolist()
        self.wall_y = self.walldata[1, :].tolist()

        position_x = self.position[0]
        position_y = self.position[1]



        self.distance_map = self.compute_bfs_distances()

        self.prev_actions = []
        self.prev_distances = []
        self.prev_distances.append(self.distance_map.get((self.initial_position[0], self.initial_position[1]), 30))


        self.observation = [position_x, position_y, self.distance_to_target]
        self.observation = np.array(self.observation, dtype= np.float32)

        # Gymnasium requires returning (observation, info_dict)
        return self.observation, {}

    def render(self, mode='human'):
        ...

    def close(self):
        ...