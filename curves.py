from collections import namedtuple
from math import sqrt

Point = namedtuple('Point', ['x', 'y', 't'])
Curves = namedtuple('Curves', ['Real', 'Improved', 'Depth', "Index"])

def pythagoras(a,b):
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2))
