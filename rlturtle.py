import math
import turtle

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as nd
import torch

from rlturtle.exception import OffScreenException

class RLTurtle(turtle.Turtle):
    def __init__(self, width, height,
                 randomize_start_pos=False, 
                 n_history_frames=1):
        """Wraps Python's built in turtle so we can read the pixels (approximately)
        
        Access the canvas as a numpy array at RLTurtle.canvas. (Use RLTurtle.plot() to plot it.)
        Access native turtle functions with RLTurtle.turtle.pensize (for example).
        
        :param int width: Width of the window
        :param int height: Height of the window
        :param bool randomize_start_pos: Randomize turtle x, y, and heading on reset (default False)
        :param int n_history_frames: how many previous frames to store."""
        self.turtle = turtle
        self.screen = self.turtle.getscreen()
        self.screen.screensize(width, height)
        self.WIDTH = width
        self.HEIGHT = height
        self.HALF_WIDTH = self.WIDTH // 2
        self.HALF_HEIGHT = self.HEIGHT // 2
        self.canvas = torch.zeros((height, width))
        self.randomize_start_pos = False
        self.n_history_frames = n_history_frames
        self.turtle_icon = np.array([[100,0,0,0,0,0,0,0,0,0],
                                     [0,100,100,0,0,0,0,0,0,0],
                                     [0,100,100,100,100,0,0,0,0,0],
                                     [0,100,100,100,100,100,100,0,0,0],
                                     [0,100,100,100,100,100,100,100,100,0],
                                     [0,100,100,100,100,100,100,100,100,0],
                                     [0,100,100,100,100,100,100,0,0,0],
                                     [0,100,100,100,100,0,0,0,0,0],
                                     [0,100,100,0,0,0,0,0,0,0],
                                     [100,0,0,0,0,0,0,0,0,0]])
        self.turtle_buffer = torch.zeros((5, 5))
        self.reset()
    
    def erase_turtle(self):
        y = int(self.y)
        x = int(self.x)
        if y < 5: y = 5
        if y > self.HEIGHT - 5: y = self.HEIGHT - 5
        if x < 5: x = 5
        if x > self.WIDTH - 5: x = self.WIDTH - 5
        self.canvas[y-5:y+5, x-5:x+5] = self.turtle_buffer
    
    def draw_turtle(self):
        y = int(self.y)
        x = int(self.x)
        if y < 5: y = 5
        if y > self.HEIGHT - 5: y = self.HEIGHT - 5
        if x < 5: x = 5
        if x > self.WIDTH - 5: x = self.WIDTH - 5
        self.turtle_buffer = self.canvas[y-5:y+5, x-5:x+5].clone()
        self.canvas[y-5:y+5, x-5:x+5] = \
            torch.FloatTensor(nd.rotate(self.turtle_icon, self.deg, reshape=False, prefilter=False))
    
    def left(self, amt):
        self.erase_turtle()
        self.deg += amt
        self.deg = self.deg % 360
        self.draw_turtle()
        
    def right(self, amt):
        self.erase_turtle()
        self.deg += (360 - amt)
        self.deg = self.deg % 360
        self.draw_turtle()
    
    def forward(self, amt):
        dy = (math.sin(self.deg * math.pi / 180) * amt)
        dx = (math.cos(self.deg * math.pi / 180) * amt)
        step = 0
        while step < amt:
            y_step = dy / amt
            x_step = dx / amt
            if self.y - y_step < 0 or self.y - y_step > self.HEIGHT - 1 or \
                self.x + x_step < 0 or self.x + x_step > self.WIDTH - 1:
                    raise OffScreenException
                    return False
            self.erase_turtle()
            self.prev_frames.pop(0)
            self.y -= y_step
            self.x += x_step
            self.canvas[int(self.y)][int(self.x)] = 100
            self.draw_turtle()
            self.prev_frames.append(self.canvas.clone())
            step += 1
        self.x = int(self.x)
        self.y = int(self.y)
        turtle.setheading(self.deg)
        turtle.setposition(self.x - self.HALF_WIDTH, self.y - self.HALF_HEIGHT)
        
    def reset(self):
        self.turtle.reset()
        self.canvas = torch.zeros((self.HEIGHT, self.WIDTH))
        self.prev_frames = [self.canvas.clone() for _ in range(0, self.n_history_frames)]
        turtle.penup()
        if self.randomize_start_pos:
            self.x = random.randint(0, self.WIDTH)
            self.y = random.randint(0, self.HEIGHT)
            self.deg = random.randint(0, 360)
        else:
            self.x = self.HALF_WIDTH
            self.y = self.HALF_HEIGHT
            self.deg = 0
        # Turtle treats middle of screen as (0, 0)
        turtle.setposition(self.x - self.HALF_WIDTH, self.y - self.HALF_HEIGHT)
        turtle.pendown()
        self.draw_turtle()
        return self.canvas

    def plot(self):
        return plt.imshow(self.canvas)
