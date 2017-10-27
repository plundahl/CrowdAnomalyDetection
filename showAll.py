import matplotlib.pyplot as plt
from math import cos, sin, radians

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
for point in lista:
    plt.arrow(point[0],point[1],length*sin(point[2]),length*cos(point[2]), head_width=5)

xmax = 400
ymax = 400
plt.axis([-xmax,xmax,-ymax,ymax])
plt.show()
