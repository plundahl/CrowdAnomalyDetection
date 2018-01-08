import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from math import cos, sin, radians, sqrt, pow, fabs, floor
from collections import namedtuple
from typing import List
import time
import math
import random
from tens import tens, Node

random.seed(1)

start_time = time.time()

Point = namedtuple('Point', ['x', 'y', 't'])
Curves = namedtuple('Curves', ['Real', 'Improved'])

def pythagoras(a,b):
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2))

minTime = 0
maxTime = 4400
intervals = 2000

def openCurveFile(name: str):
    with open(name) as f:
        trainingCurves = []
        curves = int(f.readline().split()[0])
        for i in range(0,curves):
            nrOfPoints = int(f.readline().split()[0])
            tmpPoints = []
            for j in range (0, nrOfPoints):
                line = f.readline()
                word = line.split()
                tmpPoints.append( Point(float(word[0]), float(word[1]), int(word[2])))
            currentP = 0
            tmpCurve = []
            for j in range (1, intervals + 1):
                currTime = minTime + (maxTime/intervals)*j
                if currentP == 0 and tmpPoints[0].t > currTime:
                    continue
                while currentP+1 < nrOfPoints:
                    if tmpPoints[currentP+1].t < currTime:
                        currentP += 1
                    else:
                        break
                if currentP+1 == nrOfPoints:
                    break
                
                start = tmpPoints[currentP]
                stop = tmpPoints[currentP+1]

                x = (start.x + ((stop.x - start.x)/(stop.t - start.t))*(currTime-start.t))
                y = (start.y + ((stop.y - start.y)/(stop.t - start.t))*(currTime-start.t))
                tmpCurve.append( Point(x,y,currTime))
            
            trainingCurves.append(Curves(tmpPoints, tmpCurve))
    return trainingCurves

trainingCurves = openCurveFile("students001.vsp")

random.shuffle(trainingCurves)
trainingCurves = trainingCurves[:100]

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


Dyads = []

def messure(curves: List[Point]):
    sum_curvature = 0
    abs_curvature = 0
    length = 0
    speed = 0
    simple = 0
    if(len(curves) >= 2):
        simple = pythagoras(curves[0],curves[-1])
        for i in range(1,len(curves)):
            length += pythagoras(curves[i-1], curves[i])
        speed = length/(curves[-1].t - curves[0].t)
    if(len(curves) > 2):
        for i in range(len(curves)-3):
            sum_curvature += curvature(curves[i:i+3])
            abs_curvature += abs(curvature(curves[i:i+3]))
        sum_curvature = sum_curvature/len(curves)
    return [length, speed, simple, sum_curvature, abs_curvature]

Measurments = []
for curves in trainingCurves:
    Measurments.append(messure(curves.Improved))

pop = []
tmp = 0
for i in range(0, len(trainingCurves)):
    for j in range(0, len(trainingCurves)):
        if i<=j:
            break
        dyad = [fabs(x[0]-x[1]) for x in zip(Measurments[i], Measurments[j])]
        Dyads.append(dyad)
        pop.append(Node(tmp, dyad))
        tmp+=1

print("Setup --- %s seconds ---" % (time.time() - start_time))
start_time = time.time()

F = []
result = 0
def prof():
    global result
    result = tens(pop)
prof()
#cProfile.run("prof()")
for root in result:
    F.append(root.getList())

print("T-ENS --- %s seconds ---" % (time.time() - start_time))
start_time = time.time()

"""
Nearest Neighbors
"""
testCurves = openCurveFile("students003.vsp")

score = []
for curves in testCurves:
    tmp = messure(curves.Improved)
    tmpCurves = []
    tmpDiff = []
    for i in range(0, len(trainingCurves)):
        tmpCurves.append(i)
        tmpDiff.append(
            [fabs(x[0]-x[1]) for x in zip(Measurments[i], tmp)]
        )
    nearest = []
    for i in range(len(tmp)):
        nearest.extend([x for _,x in sorted(zip(tmpDiff,tmpCurves), key=lambda item: item[0][i] )][:10])
    depthList = []
    for i in nearest:
        dyad = [fabs(x[0]-x[1]) for x in zip(Measurments[i], tmp)]
        depthList.append(Node(1,dyad).get_depth(result))
    score.append(sum(depthList)/len(depthList))

sortedScore = [x for _,x in sorted(zip(score,range(0,len(score))))]


print("Nearest --- %s seconds ---" % (time.time() - start_time))

"""
with open("students003.vsp") as f:
    for line in f:
        word = line.split()
        if(len(word) == 8):
            lista.append([float(word[0]),float(word[1]),radians(float(word[3]))])
"""

plt.figure(1)
xmax = 400
ymax = 400


plt.subplot(121)
plt.title('10 most anomalous')
i = 0
for x in sortedScore[-10:]:
    i+=1
    for y in range(1,len(testCurves[x].Real)):
        point1 = point = testCurves[x].Real[y-1]
        point2 = testCurves[x].Real[y]
        plt.plot([point1[0],point2[0]],[point1[1],point2[1]], color='C'+str(i%10))

plt.subplot(122)
plt.title('10 most nominal paths')
i = 0
for x in sortedScore[:10]:
    i+=1
    for y in range(1,len(testCurves[x].Real)):
        point1 = point = testCurves[x].Real[y-1]
        point2 = testCurves[x].Real[y]
        plt.plot([point1[0],point2[0]],[point1[1],point2[1]], color='C'+str(i%10))

print(sortedScore[-10:])
print(sortedScore[:10])

plt.axis([-xmax,xmax,-ymax,ymax])
plt.show()

