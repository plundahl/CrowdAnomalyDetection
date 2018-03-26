import random
from typing import List

class Node(object):
    def __init__(self, index: int, dyads: list):
        self.index = index
        self.size = len(dyads)
        self.dyad = dyads
        self.objSeq = list(range(1,self.size))
        random.shuffle(self.objSeq)
        self.branch = [None]*self.size

    def getList(self):
        result = [self.index]
        for elem in self.branch:
            if(elem != None):
                result = result + elem.getList()
        return result

    def get_depth(self, fronts: List):
        depth = 0
        for root in fronts:
            if(check_tree(self, root, True, False)):
                break
            depth += 1
        return depth


def tens(population: list) -> List[Node]:
    population = sorted(population, key=lambda x: x.dyad[0])
    F = []
    k = 0
    #for i in range(1):
    while(len(population) > 0):
        deleted = []
        F.append(population[0])
        for p in population:
            if(update_tree(p,F[k])):
                deleted.append(p)

        tmp=[]
        for p in deleted:
            tmp.append(p.index)
            population.remove(p)
        k = k+1
    return F

def update_tree(p: Node, tree: Node):
    if(p == tree):
        return True
    elif check_tree(p, tree, True):
        return True
    return False

def check_tree(p: Node, tree: Node, add_pos: bool, modify_tree = True):
    if(tree==None):
        return True
    if(not modify_tree):
        if(p.dyad[0] < tree.dyad[0]):
            return True

    m = find_min_m(p, tree)
    if(m == -1):
        return False
    else:
        for i in range(m+1):
            if(check_tree(p, tree.branch[i], i==m and add_pos, modify_tree) == False):
                return False
        if(tree.branch[m] == None and add_pos and modify_tree):
            tree.branch[m] = p
        return True

def find_min_m(p: Node, tree: Node):
    for m in range(p.size-1):
        if(p.dyad[tree.objSeq[m]] < tree.dyad[tree.objSeq[m]]):
            return m
    return -1

