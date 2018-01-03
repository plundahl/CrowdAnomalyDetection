import matplotlib.pyplot as plt
from math import cos, sin, radians, floor
import time
import random
import cProfile

from tens import tens, Node

length = 1000
xs = []
ys = []
xs2 = []
ys2 = []
S = []
n = []
F = [[]]

#plt.subplot(121)
random.seed(2)

for x in range (0,length):
    xs.append(random.uniform(1,100))
    ys.append(random.uniform(1,100))
    xs2.append(random.uniform(1,100))
    ys2.append(random.uniform(1,100))
    S.append([])
    n.append(0)
#    print([x, xs[x], ys[x]])
plt.plot(xs,ys,'x')

dom_tests = 0
def dominates(a,b):
    global dom_tests
    dom_tests = dom_tests + 1
    #return xs[a] <= xs[b] and ys[a] <= ys[b] and (xs[a] < xs[b] or ys[a] < ys[b])
    return xs[a] <= xs[b] and ys[a] <= ys[b] and xs2[a] <= xs2[b] and ys2[a] <= ys2[b] and (xs[a] < xs[b] or ys[a] < ys[b] or xs2[a] < xs2[b] or ys2[a] < ys2[b])


start_time = time.time()
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
        if i <= 9:
            plt.plot(xs[p],ys[p],'o', color='C'+str(i%10))
    i=i+1
    F.append(H)

print("--- FAST %s seconds ---" % (time.time() - start_time))
print("--- FAST: %s ---" % dom_tests)
dom_tests = 0

class Leaf(object):
    def __init__(self, index):
        self.index = index
        self.children = []
    
    def add(self, newLeaf, log=False):
        if log:
            print(str(self.index) + " ADD " + str(newLeaf.index))
            for child in self.children:
                print("#" + str(child.index))
                
        dominated = []
        random.shuffle(self.children)
        for child in self.children:
            if log:
                print(">" + str(child.index))
            if dominates(child.index, newLeaf.index):
                if log:
                    print("dominated")
                child.add(newLeaf)
                return
            elif dominates(newLeaf.index, child.index):
                if log:
                    print("dominates")
                dominated.append(child)
                newLeaf.add(child)
        for child in dominated:
            self.children.remove(child)
        self.children.append(newLeaf)
        return

    def merge(self):
        if len(self.children) == 1:
            self.children = self.children[0].children
        else:
            if len(self.children) > 1:
                self.children = self.mergeLists([self.children[i:i + 2] for i in range(0, len(self.children), 2)])
                self.merge()

    def mergeLists(self, pairs=[]):
        tmpList=[]
        for pair in pairs:
            if len(pair) == 1:
                tmpList.append(pair[0])
            else:
                merged = []
                for child1 in pair[0].children:
                    add = True
                    for child2 in pair[1].children:
                        if dominates(child2.index, child1.index):
                            child2.add(child1)
                            add = False
                            break
                    if add:
                        merged.append(child1)
                for child1 in pair[1].children:
                    add = True
                    for child2 in pair[0].children:
                        if dominates(child2.index, child1.index):
                            child2.add(child1)
                            add = False
                            break
                    if add:
                        merged.append(child1)
                tmpLeaf = Leaf(-2)
                tmpLeaf.children = merged
                tmpList.append(tmpLeaf)
        return tmpList

    def print(self):
        tmp = []
        for child in self.children:
            tmp.append(child.print())
        return {str(self.index) : tmp}

    def depth(self):
        if len(self.children) == 0:
            return 1
        tmp = []
        for child in self.children:
            tmp.append(child.depth())
        return max(tmp) + 1

    def size(self):
        tmp = []
        for child in self.children:
            tmp.append(child.depth())
        return sum(tmp) + 1


def treePareto():
    pop = []
    for index in range(length):
        pop.append(Node(index, [xs[index], ys[index], xs2[index], ys2[index]], 4))
    result = tens(pop)

    print("--- First TREE: %s ---" % dom_tests)

    #for child in Root.children:
    #    print("--- %s depth ---" % child.depth())
    #    print("--- %s size ---" % child.size())

    #plt.subplot(122)
    i = 0
    for root in result:
        tmp = root.getList()
        #    for child2 in child.children:
        #        tmpRoot.add(child2)
            #if i <= 9:
                #plt.plot(xs[child.index]+1,ys[child.index]+1,'o', color='C'+str(i%10))
        #Root = tmpRoot
        #print({i:sorted(tmp)})
        if sorted(tmp) != sorted(F[i]):
            print("ERROR!! on front %s" % i)
            exit()
        i = i + 1
    if(i+1 != len(F)):
        print("ERROR!! F:{} != TREE:{}".format(len(F), i+1))
        exit()

#cProfile.run("treePareto()")

start_time = time.time()
treePareto()
print("--- TENS %s seconds ---" % (time.time() - start_time))
i = 0
for x in F:
    #print({i:sorted(x)})
    i = i + 1

#plt.axis([0,2*xmax,0,2*ymax])
#plt.show()
