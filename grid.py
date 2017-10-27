import matplotlib.pyplot as plt
from math import cos, sin, radians, floor

length = 10

#plt.arrow(0.2, 0, 0.5, 0.5)
#plt.arrow(2, 0, 0.5, 0.5, head_width=0.05, head_length=0.1, fc='k', ec='k')

lista = []
"""
with open("students001.vsp") as f:
    for line in f:
        word = line.split()
        if(len(word) == 8):
            lista.append([float(word[0]),float(word[1]),radians(float(word[3]))])
with open("students003.vsp") as f:
    for line in f:
        word = line.split()
        if(len(word) == 8):
            lista.append([float(word[0]),float(word[1]),radians(float(word[3]))])
"""

def reader(file, list):
    print("hej")
    file.readline()
    line = file.readline()
    while line:
        points = int(line.split()[0])
        line=file.readline()
        for x in range(1, points):
            line=file.readline()
            word = line.split()
            lista.append([float(word[0]),float(word[1]),radians(float(word[3]))])
        line = file.readline()


with open("students001.vsp") as f:
    reader(f, lista)


prev = []
#for point in lista:
#    plt.arrow(point[0],point[1],length*sin(point[2]),length*cos(point[2]), head_width=5)

xmax = 400
ymax = 400
resolution = 20

matrix = [['nan']*int(2*ymax/resolution) for i in range(int(2*xmax/resolution))]
for point in lista:
    x = int((xmax+point[0])/resolution)
    y = int((ymax+point[1])/resolution)
    print(x,y)
    matrix[x][y] = point[2]

x = -resolution
for xlist in matrix:
    x = x + resolution
    y = -resolution
    for point in xlist:
        y = y + resolution
        if point != 'nan':
            plt.arrow(x,y,length*sin(point),length*cos(point), head_width=5)

plt.axis([0,2*xmax,0,2*ymax])
plt.show()
