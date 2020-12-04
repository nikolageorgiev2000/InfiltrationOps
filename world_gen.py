#Wave Function Collapse lib
import numpy as np
import math
import random

def get_input():
    # assume input dimensions >= 3x3
    width = 15
    return np.eye(width, dtype=int)+np.eye(width, dtype=int)[::-1]

def process_input(input):
    n = len(input)
    m = len(input[0])
    input_patched = []
    temp = 0
    for i in range(n-2):
        input_patched.append([])
        for j in range(m-2):
            input_patched[i].append(tuple(input[i:i+3, j:j+3].flatten()))
    counter = {}
    for i in input_patched:
        for j in i:
            counter[j] = counter.get(j,0)+1

    # no longer need patch position after this
    patches = list(set([j for i in input_patched for j in i]))
    freqs = np.array([counter[i] for i in patches])
    
    adjac = {}
    for i in range(n-2):
        for j in range(m-2):
            # top 0 right 1 bottom 2 left 3
            adjac.setdefault(input_patched[i][j],[set(),set(),set(),set()])
            if i>0:
                adjac[input_patched[i][j]][0].add(input_patched[i-1][j])
            if j<m-3:
                adjac[input_patched[i][j]][1].add(input_patched[i][j+1])
            if i<n-3:
                adjac[input_patched[i][j]][2].add(input_patched[i+1][j])
            if j>0:
                adjac[input_patched[i][j]][3].add(input_patched[i][j-1])

    return (patches, freqs, adjac)

def init_output(rows,cols,patches):
    return [[[True for _ in patches] for _ in range(cols)] for _ in range(rows)]

def is_collapsed(output):
    for i in output:
        for j in i:
            # at least one uncollapsed cell
            if(sum(j)>1):
                return False
    return True

def observe(output, freqs, logged_freqs, patches, entropies):
    min_entropy = float("inf")
    # find selection of min-entropy candidate cells to observe
    candidates = []
    for i in range(len(output)):
        for j in range(len(output[i])):
            supersum = sum(output[i][j])
            if(supersum == 0):
                # no patch options exist for this cell, so quit
                return None
            if(supersum == 1):
                # already collapsed
                continue
            if(not entropies[i][j][1]):
                temp = calc_entropy(freqs, logged_freqs, patches, output[i][j])
                entropies[i][j] = (temp,True)
            if(entropies[i][j][0] < min_entropy):
                candidates = [(i,j)]
                min_entropy = entropies[i][j][0]
            elif(entropies[i][j][0] == min_entropy):
                candidates.append([i,j])
    # randomly chose a candidate and observe (collapse wave function) on them
    # with a random choice of possible patches using their probability in the input
    cand_pos = random.choice(candidates)
    patch_probs = np.array(output[cand_pos[0]][cand_pos[1]])*freqs
    patch_probs = patch_probs / np.sum(patch_probs)
    chosen_patch = np.random.choice(range(len(patches)), p = patch_probs)
    output[cand_pos[0]][cand_pos[1]] = [False for _ in patches]
    output[cand_pos[0]][cand_pos[1]][chosen_patch] = True
    # print(list(zip(patches,cell_probs)))

    return cand_pos

def propagate(output, adjac, pos, entropies):
    pass

x = 0

def calc_entropy(freqs, logged_freqs, patches, superpos):
    global x
    # (1 if possible patch else 0) * (Prob of patch) * (binary information provided by patch probability) -> Shannon Entropy
    entropy = sum(np.array(superpos)*freqs*logged_freqs)
    x+=1
    return entropy

def one_index(arr):
    # for i in range(len(arr)):
    #     if(arr[i]==1):
    #         return i
    return int(np.nonzero(arr)[0])

def generate_world(rows,cols):
    print("INPUT")
    input = get_input()
    # print(input)
    print("PROCESS")
    patches, freqs, adjac = process_input(input)
    # print(patches)
    # print(freqs)
    # print(adjac)
    print("OUTPUT INIT")
    output = init_output(rows, cols, patches)
    c = 0

    print("COLLAPSING")
    # boolean represents if the entropy is up to date
    entropies = [[(0, False) for j in i] for i in output]

    logged_freqs = np.log2(1/freqs)
    while(not is_collapsed(output)):
        collapsed_pos = observe(output, freqs, logged_freqs, patches, entropies)
        if(collapsed_pos==None):
            # got stuck, found non-collapsable cell
            return None
        propagate(output, adjac, collapsed_pos, entropies)
        c+=1

    print(f"Loops: {c}")
    print("CONVERTING")
    print(f"PATCHES: {len(patches)}")
    print(f"Entropies calculated: {x}")
    return np.array([[patches[one_index(j)][0] for j in i] for i in output])



print(generate_world(50,50))
