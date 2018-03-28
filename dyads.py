from math import fabs, floor
import numpy

import sys
import time

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from collections import namedtuple
from functools import reduce
from typing import List, Tuple

from tens import tens, Node
from measure import measure
from curves import Point, Curves, pythagoras

sys.setrecursionlimit(10000)


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
        obstacleInfo = f.readline()
        if "line obstacles" in obstacleInfo:
            curves = int(obstacleInfo.split()[0])
            for i in range(0,curves):
                line = f.readline()
                word = line.split()
                tmpPoints = []
                tmpPoints.append( Point(float(word[0]), float(word[1]), 0))
                tmpPoints.append( Point(float(word[2]), float(word[3]), 0))
                obstacles.append(Curves(tmpPoints, [], -1, 0))
 
    return (trainingCurves, obstacles)


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
        tmp = measure(curves.Improved, curves.Index, testCurves, testObstacles)
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
    if len(sys.argv) < 3:
        print("Usage: python3 dyads.py TRAINING-FILE TEST-FILE")
        return
    trainingFile = sys.argv[1]
    testFile = sys.argv[2]
    start_time=time.time()
    [trainingCurves, trainingObstacles] = openCurveFile(trainingFile)
    print(len(trainingCurves))

    print("OpenCurve --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    
    trainingMeasurments = []
    for curves in trainingCurves:
        trainingMeasurments.append(measure(curves.Improved, curves.Index, trainingCurves, trainingObstacles))

    trainingTree = createTrainingData(trainingMeasurments)

    print("Training Depth: ", len(trainingTree))

    [testCurves, testObstacles] = openCurveFile(testFile)
    testMeasurments = []
    for curves in testCurves:
        testMeasurments.append(measure(curves.Improved, curves.Index, testCurves, testObstacles))
    testTree = createTrainingData(testMeasurments)
    
    print("Test Depth: ", len(testTree))
        
    print("T-ENS --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()

    numpy.set_printoptions(precision=5, linewidth=1000)
    print("[speed, min_speed, max_speed, length/simple, sum_curvature, abs_curvature, min_obst_dist, min_neigh_dist, avg_min_neigh_dist, avg_nighbours]")
    
    print( "training: ", numpy.asarray(list(map(lambda z: z/len(trainingMeasurments), reduce(lambda x,y: [a + b for a, b in zip(x,y)], trainingMeasurments) ) ) ) )
    print( "test    : ", numpy.asarray(list(map(lambda z: z/len(testMeasurments), reduce(lambda x,y: [a + b for a, b in zip(x,y)], testMeasurments) ) ) ) )

    outliersTraining = calculateScore(trainingTree, trainingMeasurments, trainingCurves, trainingObstacles)
    badBehaviour = calculateScore(trainingTree, trainingMeasurments, testCurves, testObstacles)
    nonCapturedBehaviour = calculateScore(testTree, testMeasurments, trainingCurves, trainingObstacles)
    outliersTest = calculateScore(testTree, testMeasurments, testCurves, testObstacles)

    print("Nearest --- %s seconds ---" % (time.time() - start_time))

    print()
    print("SCORE:   ", sum(x.Depth for x in badBehaviour) /len(testCurves) )
    print("MISSING: ", sum(x.Depth for x in nonCapturedBehaviour) /len(trainingCurves) )
        
    plotResult(outliersTraining, badBehaviour, nonCapturedBehaviour, outliersTest)

if __name__ == "__main__":
    main()
