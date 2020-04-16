import sys,os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as numpy
from abc import abstractmethod, ABCMeta

# class PopBase(metaclass=ABCMeta):
#     """An abstract base class for simulating multicellular dynamics"""
#     def __init__(self, cell):
