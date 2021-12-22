import numpy as np
from Track import *
white_upper = np.array([179,15,255])
white_lower = np.array([0,0,200])
red_lower = np.array([[160, 100, 20]])
red_upper = np.array([179, 255, 255])
orange_lower = np.array([0,180,100])
orange_upper = np.array([70,255,255])
ranges = [
    [np.array([179,15,255]),np.array([0,0,200])],
    [np.array([[160, 100, 20]]),np.array([179, 255, 255])],
    [np.array([0,180,100]),np.array([70,255,255])]
]

videoLoop(orange_lower,orange_upper)

