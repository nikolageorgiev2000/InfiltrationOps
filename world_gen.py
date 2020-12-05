#Wave Function Collapse lib
import numpy as np
import math
import random

def get_input():
    # assume input dimensions >= 3x3
    # width = 15
    # mat_in = np.eye(width, dtype=int)+3*np.eye(width, dtype=int)[::-1]
    # mat_in[0]+=10
    # mat_in[1]+=7
    mat_in = np.array([[' ',' ',' ',' ','O'],[' ','X','–','X',' '],[' ','|',' ','|',' '],[' ','X','–','X',' '],[' ',' ',' ',' ',' ']])
    # Add horizontal, vertical, diagonal flips to input matrix
    mat_horiz = mat_in[::-1]
    mat_diag = mat_horiz.transpose()[::-1].transpose()
    mat_vert = mat_diag[::-1]
    mat_flips = np.concatenate([mat_in,mat_horiz])
    mat_flips = np.concatenate([mat_flips, np.concatenate([mat_vert,mat_diag])], axis=1)
    # print(mat_flips)
    return mat_flips

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
    indices = {p:i for i,p in enumerate(patches)}
    freqs = np.array([counter[i] for i in patches])
    indexed_input = [[indices[j] for j in i] for i in input_patched]

    adjac = {}
    for i in range(n-2):
        for j in range(m-2):
            # top 0 right 1 bottom 2 left 3
            adjac.setdefault(indexed_input[i][j],[set(),set(),set(),set()])
            if i>0:
                adjac[indexed_input[i][j]][0].add(indexed_input[i-1][j])
            if j<m-3:
                adjac[indexed_input[i][j]][1].add(indexed_input[i][j+1])
            if i<n-3:
                adjac[indexed_input[i][j]][2].add(indexed_input[i+1][j])
            if j>0:
                adjac[indexed_input[i][j]][3].add(indexed_input[i][j-1])
    
    return (indexed_input, patches, freqs, adjac)


def init_output(rows,cols,patches,adjac):

    domains = [[[True for _ in patches] for _ in range(cols)] for _ in range(rows)]
    adjac_counter = [[[[len(a) for a in adjac[p]] for p in range(len(patches))] for _ in range(cols)] for _ in range(rows)]

    return domains, adjac_counter

def is_collapsed(domains):
    for i in domains:
        for j in i:
            # at least one uncollapsed cell
            if(sum(j)>1):
                return False
    return True

def observe(domains, freqs, logged_freqs, patches, entropies):
    min_entropy = float("inf")
    # find selection of min-entropy candidate cells to observe
    candidates = []
    for i in range(len(domains)):
        for j in range(len(domains[i])):
            supersum = sum(domains[i][j])
            if(supersum == 0):
                # no patch options exist for this cell, so quit
                return None
            if(supersum == 1):
                # already collapsed
                continue
            if(not entropies[i][j][1]):
                temp = calc_entropy(freqs, logged_freqs, patches, domains[i][j])
                entropies[i][j] = (temp,True)
            if(entropies[i][j][0] < min_entropy):
                candidates = [(i,j)]
                min_entropy = entropies[i][j][0]
            elif(entropies[i][j][0] == min_entropy):
                candidates.append([i,j])
    # randomly chose a candidate and observe (collapse wave function) on them
    # with a random choice of possible patches using their probability in the input
    cand_pos = random.choice(candidates)
    patch_probs = np.array(domains[cand_pos[0]][cand_pos[1]])*freqs
    patch_probs = patch_probs / np.sum(patch_probs)
    chosen_patch = np.random.choice(range(len(patches)), p = patch_probs)
    domains[cand_pos[0]][cand_pos[1]] = [False for _ in patches]
    domains[cand_pos[0]][cand_pos[1]][chosen_patch] = True
    # print(list(zip(patches,cell_probs)))

    #TODO: return cand_pos,domain_delta instead, so the domain isn't changed within observation, only chosen. It is changed in propagate()

    return cand_pos

def propagate(indexed_input, domains, adjac, start_pos, domain_delta, entropies):
    stack = []
    # for each 1 in the domain_delta add the needed remove_tile_possibility arguments as a tuple in stack
    while(len(stack)>0):
        # perform tile removal, updating entropies that need recalculation, changed domains and appending new removals stack through reference
        remove_tile_possibility(*stack.pop(-1))



    #DONE; TODO: Initialize array, where we store the count of the support, i.e. for a given cell/tile/side, we count how many tiles in the domain on adjacent cell can be placed next to the tile in question.        
    #TODO: go through adjacent cell's tile values that we know a removed tile on the current cell can be adjacent to and subtract 1. if any of the neighbor tile values are 0, we know this tile has no possible neighbors and can be removed from domain

def remove_tile_possibility():


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
    indexed_input, patches, freqs, adjac = process_input(input)
    # print(patches)
    # print(freqs)
    # print(adjac)
    print("OUTPUT INIT")
    domains, adjac_counter = init_output(rows, cols, patches, adjac)

    print("COLLAPSING")
    # boolean represents if the entropy is up to date
    entropies = [[(0, False) for j in i] for i in domains]

    logged_freqs = np.log2(1/freqs)
    c = 0
    while(not is_collapsed(domains)):
        collapsed_pos = observe(domains, freqs, logged_freqs, patches, entropies)
        if(collapsed_pos==None):
            # got stuck, found non-collapsable cell
            return None
        propagate(indexed_input, domains, adjac, collapsed_pos, entropies)
        c+=1

    print(f"Loops: {c}")
    print("CONVERTING")
    print(f"PATCHES: {len(patches)}")
    print(f"Entropies calculated: {x}")
    return np.array([[patches[one_index(j)][0] for j in i] for i in domains])



print(generate_world(10,10))
