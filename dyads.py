import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from math import cos, sin, radians, sqrt, pow, fabs, floor
import numpy
from collections import namedtuple
from typing import List, Tuple
import time
import math
import random
import sys
from functools import reduce
sys.setrecursionlimit(10000)


from tens import tens, Node



Point = namedtuple('Point', ['x', 'y', 't'])
Curves = namedtuple('Curves', ['Real', 'Improved', 'Depth', "Index"])

def pythagoras(a,b):
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2))

stepSize = 5

def openCurveFile(name: str) -> Tuple[List[Curves],List[Curves]]:
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
            j = 0
            while(True):
                currTime = stepSize*j
                j += 1
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
            
            trainingCurves.append(Curves(tmpPoints, tmpCurve, -1, i))
        
        obstacles = []
        curves = int(f.readline().split()[0])
        for i in range(0,curves):
            line = f.readline()
            word = line.split()
            tmpPoints = []
            tmpPoints.append( Point(float(word[0]), float(word[1]), 0))
            tmpPoints.append( Point(float(word[2]), float(word[3]), 0))
            obstacles.append(Curves(tmpPoints, [], -1, 0))
 
    return (trainingCurves, obstacles)

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




def messure(curve: List[Point], index: int, curves: List[Curves], obstacles: List[Curves]) -> List:
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

def createTrainingData(messurments: List[List]) -> List[Node]:
    pop = []
    tmp = 0
    for i in range(0, len(messurments)):
        for j in range(0, len(messurments)):
            if i <= j:
                break
            dyad = [fabs(x[0]-x[1]) for x in zip(messurments[i], messurments[j])]
            pop.append(Node(tmp, dyad))
            tmp+=1
    print(len(pop))
    return tens(pop)

"""
Nearest Neighbors
"""

def calculateScore(nonDominatedTree: List[Node], trainingMeasurments: List[List], testCurves: List[Curves], testObstacles: List[Curves]) -> List[Curves]:
    score = []
    result = []
    for curves in testCurves:
        tmp = messure(curves.Improved, curves.Index, testCurves, testObstacles)
        tmpCurves = []
        tmpDiff = []
        for i in range(0, len(trainingMeasurments)):
            tmpCurves.append(i)
            tmpDiff.append(
                [fabs(x[0]-x[1]) for x in zip(trainingMeasurments[i], tmp)]
            )
        nearest = []
        for i in range(len(tmp)):
            nearest.extend([x for _,x in sorted(zip(tmpDiff,tmpCurves), key=lambda item: item[0][i] )][:10])
        depthList = []
        for i in nearest:
            dyad = [fabs(x[0]-x[1]) for x in zip(trainingMeasurments[i], tmp)]
            depthList.append(Node(1,dyad).get_depth(nonDominatedTree))
        score.append(sum(depthList)/len(depthList))
        result.append( Curves(curves.Real, curves.Improved, sum(depthList)/len(depthList), curves.Index))

    sortedScore = [x for _,x in sorted(zip(score,range(0,len(score))))]
    return sorted(result, key=lambda curve: curve.Depth)



def plotResult(outliersTraining: List[Curves], badBehaviour: List[Curves], nonCapturedBehaviour: List[Curves], outliersTest: List[Curves]):
    Plot = namedtuple('Plot', ['plot', 'curve'])
    plt.subplots()
    plt.subplots_adjust(bottom=0.20)

    plt.figure(1)
    xmax = 400
    ymax = 400

    plots = [
        Plot(plt.subplot(221), outliersTraining),
        Plot(plt.subplot(222), badBehaviour),
        Plot(plt.subplot(223), nonCapturedBehaviour),
        Plot(plt.subplot(224), outliersTest)
    ]

   
    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
    sfreq = Slider(axfreq, 'Freq', 1, 100, valinit=10)

    def update(val):
        nr = floor(val)
        print(nr)
        for p in plots:
            p.plot.cla()
            p.plot.axis([-xmax,xmax,-ymax,ymax])
            i = 0
            threshold = len(p.curve)
            if nr > 10:
                threshold = 10-nr
            #Print greyed out:
            """
            for curve in p.curve[threshold:]:
                for y in range(1,len(curve.Real)):
                    point1 = point = curve.Real[y-1]
                    point2 = curve.Real[y]
                    p.plot.plot([point1[0],point2[0]],[point1[1],point2[1]], color='#ffc0c0')
            """
            #Print colerized:
            for curve in p.curve[-nr:threshold]:
                i+=1
                for y in range(1,len(curve.Real)):
                    point1 = point = curve.Real[y-1]
                    point2 = curve.Real[y]
                    p.plot.plot([point1[0],point2[0]],[point1[1],point2[1]], color='C'+str(i%10),label=str(curve.Index))
                p.plot.text(-300,30*i,str(curve.Index),color='C'+str(i%10))
            

        plots[0].plot.set_title('Training outilers: ' + str(plots[0].curve[-nr].Depth) )
        plots[1].plot.set_title('Bad behaviour: ' + str(plots[1].curve[-nr].Depth) )
        plots[2].plot.set_title('Non captured: ' + str(plots[2].curve[-nr].Depth) )
        plots[3].plot.set_title('Test outliers: ' + str(plots[3].curve[-nr].Depth) )
 
    update(10)

    sfreq.on_changed(update)

    plt.show()

saveFile = namedtuple("saveFile",["i", "j", "k", "l"])
def main():
    start_time=time.time()
    [trainingCurves, trainingObstacles] = openCurveFile("students003.vsp")
    #[trainingCurves, trainingObstacles] = openCurveFile("uni_examples.vsp")
    print(len(trainingCurves))
    #random.shuffle(trainingCurves)
    #trainingCurves = trainingCurves[:118]
    #for c in trainingCurves:
    #    findNeighbours(c.Improved,trainingCurves)

    print("OpenCurve --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    
    trainingMeasurments = []
    for curves in trainingCurves:
        trainingMeasurments.append(messure(curves.Improved, curves.Index, trainingCurves, trainingObstacles))

    trainingTree = createTrainingData(trainingMeasurments)

    print("Training Depth: ", len(trainingTree))

    [testCurves, testObstacles] = openCurveFile("clustering.vsp")
    #testCurves = testCurves[:118]
    testMeasurments = []
    for curves in testCurves:
        testMeasurments.append(messure(curves.Improved, curves.Index, testCurves, testObstacles))
    testTree = createTrainingData(testMeasurments)
    
    print("Test Depth: ", len(testTree))
        
    print("T-ENS --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()

    numpy.set_printoptions(precision=5, linewidth=1000)
    print("[speed, min_speed, max_speed, length/simple, sum_curvature, abs_curvature, min_obst_dist, min_neigh_dist, avg_min_neigh_dist, avg_nighbours]")
    
    print( numpy.asarray(list(map(lambda z: z/len(trainingMeasurments), reduce(lambda x,y: [a + b for a, b in zip(x,y)], trainingMeasurments) ) ) ) )
    print( numpy.asarray(list(map(lambda z: z/len(testMeasurments), reduce(lambda x,y: [a + b for a, b in zip(x,y)], testMeasurments) ) ) ) )

    outliersTraining = calculateScore(trainingTree, trainingMeasurments, trainingCurves, trainingObstacles)
    badBehaviour = calculateScore(trainingTree, trainingMeasurments, testCurves, testObstacles)
    nonCapturedBehaviour = calculateScore(testTree, testMeasurments, trainingCurves, trainingObstacles)
    outliersTest = calculateScore(testTree, testMeasurments, testCurves, testObstacles)

    print("Nearest --- %s seconds ---" % (time.time() - start_time))

    print()
    print("SCORE: ", sum(x.Depth for x in badBehaviour))
    print("MISSING: ", sum(x.Depth for x in nonCapturedBehaviour))
        
    plotResult(outliersTraining, badBehaviour, nonCapturedBehaviour, outliersTest)

if __name__ == "__main__":
    main()
