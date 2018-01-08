import matplotlib.pyplot as plt
from math import cos, sin, radians, sqrt, pow, fabs
from collections import namedtuple
import time
import random
from tens import tens, Node

random.seed(1)

start_time = time.time()

Point = namedtuple('Point', ['x', 'y', 't'])
length = 10

#plt.arrow(0.2, 0, 0.5, 0.5)
#plt.arrow(2, 0, 0.5, 0.5, head_width=0.05, head_length=0.1, fc='k', ec='k')

def pythagoras(a,b):
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2))

minTime = 0
maxTime = 4400
intervals = 2000

lista = [[]]
trainingCurves = []
with open("students001.vsp") as f:
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
        trainingCurves.append(tmpCurve)

random.shuffle(trainingCurves)
trainingCurves = trainingCurves[:100]

Dyads = []
val_length = []
val_speed = []
val_simple_curve = []
for curves in trainingCurves:
    if(len(curves) < 2):
        val_length.append(0)
        val_speed.append(0)
        val_simple_curve.append(0)
    else:
        val_simple_curve.append(pythagoras(curves[0],curves[-1]))
        length = 0
        for i in range(1,len(curves)):
            length += pythagoras(curves[i-1], curves[i])
        val_length.append(length)
        val_speed.append(length/(curves[-1].t - curves[0].t))


pop = []
for i in range(0, len(trainingCurves)):
    for j in range(0, len(trainingCurves)):
        if i<=j:
            break
        dyad = [
            fabs(val_length[i] - val_length[j]),
            fabs(val_speed[i] - val_speed[j]),
            fabs(val_simple_curve[i] - val_simple_curve[j]),
            i,j
        ]
        Dyads.append(dyad)
        pop.append(Node(i, dyad, 3))

domN = 0
def dominates(a,b):
    global domN
    domN = domN + 1
    return a[0] <= b[0] and a[1] <= b[1] and a[2] <= b[2] and (a[0] < b[0] or a[1] < b[1] or a[2] < b[2])


S = []
n = []
F = []
for p in range(0,len(Dyads)):
    S.append([])
    n.append(0)


result = tens(pop)
for root in result:
    F.append(root.getList())

"""
for p in range(0,len(Dyads)-1):
    for q in range(p+1, len(Dyads)):
        if dominates(Dyads[p],Dyads[q]):
            S[p].append(q)
            n[q] = n[q] + 1
        elif dominates(Dyads[q],Dyads[p]):
            n[p] = n[p] + 1
            S[q].append(p)
    if n[p] == 0:
        F[0].append(p)

print(F[0])
print("--- %s comparissons ---" % domN)
i = 0
while F[i]:
    H = []
    for p in F[i]:
        for q in S[p]:
            n[q] = n[q] - 1
            if n[q] == 0:
                H.append(q)
    i=i+1
    F.append(H)

"""
"""
print(len(F))
print(F[-2])
print(F[-3])
print(F[-4])
print(Dyads[F[-2][0]])
print(Dyads[F[-3][0]])
print(Dyads[F[-4][0]])
"""

"""
Nearest Neighbors
"""
testCurves = []
with open("students001.vsp") as f:
    curves = int(f.readline().split()[0])
    for i in range(0,curves):
        nrOfPoints = int(f.readline().split()[0])
        tmpPoints = []
        lista1 = []
        for j in range (0, nrOfPoints):
            line = f.readline()
            word = line.split()
            tmpPoints.append( Point(float(word[0]), float(word[1]), int(word[2])))
            lista1.append([float(word[0]),float(word[1]),radians(float(word[3]))])
        lista.append(lista1)
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
        testCurves.append(tmpCurve)


score = []
for curves in testCurves:
    if(len(curves) < 2):
        length = 0
        speed = 0
        curve = 0
    else:
        curve = (pythagoras(curves[0],curves[-1]))
        length = 0
        for i in range(1,len(curves)):
            length += pythagoras(curves[i-1], curves[i])
        speed = (length/(curves[-1].t - curves[0].t))
    tmpCurves = []
    tmpDiff = []
    for i in range(0, len(trainingCurves)):
        tmpCurves.append(i)
        tmpDiff.append(
                fabs(val_length[i] - length) +
                fabs(val_speed[i] - speed) +
                fabs(val_simple_curve[i] - curve)
        )
    nearest = [x for _,x in sorted(zip(tmpDiff,tmpCurves))][:10]
    depthList = []
    for i in nearest:
        dyad = [
                fabs(val_length[i] - length),
                fabs(val_speed[i] - speed),
                fabs(val_simple_curve[i] - curve)
        ]
        depth = 0
        stop = False
        for dList in F:
            if stop:
                break
            for dVal in dList:
                if dominates(dyad, Dyads[dVal]):
                   stop = True
                   break
            depth += 1
        depthList.append(depth)
    score.append(sum(depthList)/len(depthList))

sortedScore = [x for _,x in sorted(zip(score,range(0,len(score))))]


print("--- %s seconds ---" % (time.time() - start_time))

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
    for y in range(1,len(lista[x])):
        point1 = point = lista[x][y-1]
        point2 = lista[x][y]
        plt.plot([point1[0],point2[0]],[point1[1],point2[1]], color='C'+str(i%10))

plt.subplot(122)
plt.title('10 most nominal paths')
i = 0
for x in sortedScore[:10]:
    i+=1
    for y in range(1,len(lista[x])):
        point1 = point = lista[x][y-1]
        point2 = lista[x][y]
        plt.plot([point1[0],point2[0]],[point1[1],point2[1]], color='C'+str(i%10))

print(sortedScore[-10:])
print(sortedScore[:10])

plt.axis([-xmax,xmax,-ymax,ymax])
plt.show()

