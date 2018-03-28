import math
from math import cos, sin, radians, sqrt, pow, fabs, floor
from typing import List, Tuple

from curves import Point, Curves, pythagoras


#def measure(curve: List[Point], index: int, curves: List[Curves], obstacles: List[Curves]) -> List:
def measure(curve: List[Point], index: int, curves: List[Curves], obstacles: List[Curves]) -> List:
    sum_curvature = 0
    abs_curvature = 0
    length = 0
    speed = 0
    min_speed = math.inf
    max_speed = 0
    simple = 0
    min_obst_dist = math.inf
    sum_obst_dist = 0
    min_neigh_dist = math.inf
    avg_min_neigh_dist = math.inf
    avg_neighbours = 0
    if(len(curve) >= 2):
        simple = pythagoras(curve[0],curve[-1])
        for i in range(1,len(curve)):
            segment = pythagoras(curve[i-1], curve[i])
            length += segment
            min_speed = min(min_speed, segment/(curve[i].t - curve[i-1].t))
            max_speed = max(max_speed, segment/(curve[i].t - curve[i-1].t))
        speed = length/(curve[-1].t - curve[0].t)
    if(len(curve) > 2):
        for i in range(len(curve)-3):
            sum_curvature += curvature(curve[i:i+3])
            abs_curvature += abs(curvature(curve[i:i+3]))
        sum_curvature = sum_curvature/len(curve)
    for point in curve:
        min_obst_dist = min(min_obst_dist, obstacleDistance(point, obstacles))
        sum_obst_dist += obstacleDistance(point, obstacles)
    findN = findNeighbours(curve, index, curves)
    min_neigh_dist = min(findN)
    avg_min_neigh_dist = sum(findN)/len(findN)
    avg_neighbours = sum(numberOfNeighbours(curve, index, curves, 100)) / len(curve)
    return [speed, min_speed, max_speed, length/simple, sum_curvature, abs_curvature, min_obst_dist, min_neigh_dist, avg_min_neigh_dist, avg_neighbours]


def curvature( list: List[Point] ):
    if ( len( list ) < 3 ):
        #Too few points
        sys.exit( 0 ) 
        return -1
    else:
        x1 = list[0].x
        y1 = list[0].y
        x2 = list[1].x
        y2 = list[1].y
        x3 = list[2].x
        y3 = list[2].y

        num = 2*((x2-x1)*(y3-y2)-(y2-y1)*(x3-x2))
        den = math.sqrt( (math.pow((x2-x1),2) + math.pow((y2-y1),2)) * (math.pow((x3-x2),2)+math.pow((y3-y2),2 ))* (math.pow((x1-x3),2)+math.pow((y1-y3),2) ) )

        if ( den == 0 ):
            return 0

        return num/den

def dist(x1,y1, x2,y2, x3,y3): # x3,y3 is the point
    px = x2-x1
    py = y2-y1

    something = px*px + py*py
    if(something == 0.0):
        return math.inf

    u =  ((x3 - x1) * px + (y3 - y1) * py) / float(something)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = x1 + u * px
    y = y1 + u * py

    dx = x - x3
    dy = y - y3

    # Note: If the actual distance does not matter,
    # if you only want to compare what this function
    # returns to other results of this function, you
    # can just return the squared distance instead
    # (i.e. remove the sqrt) to gain a little performance

    dist = math.sqrt(dx*dx + dy*dy)

    return dist


def obstacleDistance(point: Point, obst: List[Curves]) -> float:
    minDist = math.inf
    for line in obst:
        minDist = min(minDist, dist(line.Real[0].x, line.Real[0].y, line.Real[1].x, line.Real[1].y, point.x, point.y))
    return minDist

def findNeighbours(curve: List[Point], index: int, others: List[Curves]):
    startTime = curve[0].t
    stopTime = curve[len(curve) - 1].t
    start = []
    stop = []
    i = 0
    for other in others:
        if(other.Index == index):
            i+=1
            continue
        if(other.Improved[0].t <= startTime and other.Improved[len(other.Improved)-1].t >= startTime):
            t = 0
            while(other.Improved[t].t < startTime):
                t+=1
            start.append( (i,t) )
        elif(stopTime >= other.Improved[0].t and startTime <= other.Improved[0].t):
            t = 0
            while(curve[t].t < other.Improved[0].t):
                t+=1
            stop.append( (i,t) )
        i+=1

    result = []
    i = 0
    for point in curve:
        minDist = math.inf
        for index in start:
            if(len(others[index[0]].Improved) > index[1] + i):
                minDist = min(minDist, pythagoras(point, others[index[0]].Improved[index[1] + i]))
        for index in stop:
            if(i >= index[1] and len(others[index[0]].Improved) > i - index[1]):
                minDist = min(minDist, pythagoras(point, others[index[0]].Improved[i-index[1]]))
        if(minDist != math.inf):
            result.append(minDist)
        i += 1

    return result

def numberOfNeighbours(curve: List[Point], index: int, others: List[Curves], radius: float) -> List[int]:
    startTime = curve[0].t
    stopTime = curve[len(curve) - 1].t
    start = []
    stop = []
    i = 0
    for other in others:
        if(other.Index == index):
            i+=1
            continue
        if(other.Improved[0].t <= startTime and other.Improved[len(other.Improved)-1].t >= startTime):
            t = 0
            while(other.Improved[t].t < startTime):
                t+=1
            start.append( (i,t) )
        elif(stopTime >= other.Improved[0].t and startTime <= other.Improved[0].t):
            t = 0
            while(curve[t].t < other.Improved[0].t):
                t+=1
            stop.append( (i,t) )
        i+=1

    result = []
    i = 0
    for point in curve:
        neighbours = 0
        for index in start:
            if(len(others[index[0]].Improved) > index[1] + i):
                dist = pythagoras(point, others[index[0]].Improved[index[1] + i])
                if(dist < radius):
                    neighbours += 1
        for index in stop:
            if(i >= index[1] and len(others[index[0]].Improved) > i - index[1]):
                dist = pythagoras(point, others[index[0]].Improved[i-index[1]])
                if(dist < radius):
                  neighbours += 1
        result.append(neighbours)
        i += 1

    return result
