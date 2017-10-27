import matplotlib.pyplot as plt
import random
from math import cos, sin, radians, floor

length = 1000
xs = []
ys = []
S = []
n = []
F = [[]]

for x in range (0,length):
    xs.append(random.uniform(1,10000))
    ys.append(random.uniform(1,10000))
    S.append([])
    n.append(0)
plt.plot(xs,ys,'x')

def dominates(a,b):
    return xs[a] <= xs[b] and ys[a] <= ys[b] and (xs[a] < xs[b] or ys[a] < ys[b])

for p in range(0,length):
    for q in range(0, length):
        if dominates(p,q):
            S[p].append(q)
        elif dominates(q,p):
            n[p] = n[p] + 1
    if n[p] == 0:
        F[0].append(p)

i = 0
while F[i]:
    H = []
    for p in F[i]:
        for q in S[p]:
            n[q] = n[q] - 1
            if n[q] == 0:
                H.append(q)
        if i <= 5:
            plt.plot(xs[p],ys[p],'o', color=str(i*0.2))
    i=i+1
    F.append(H)

#plt.axis([0,2*xmax,0,2*ymax])
plt.show()
