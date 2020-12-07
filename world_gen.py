# Wave Function Collapse lib
import numpy as np
import random
import copy
import math


class WorldGen:

    def __init__(self):

        self.patch_index = {}
        self.patches = []
        self.freqs = []
        self.logged_freqs = []
        self.adjac = []

        self.domains = []
        self.adjac_counter = []
        self.entropies = []

    def get_input(self):
        # assume input dimensions >= 3x3
        width = 25
        mat_in = [
            [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
                ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
            [' ', ' ', '•', '•', '•', 'D', 'D', '•',
                '•', ' ', ' ', ' ', ' ', '•', '•', '•'],
            [' ', ' ', '•', ' ', ' ', ' ', ' ', ' ',
                '•', ' ', ' ', ' ', ' ', '•', ' ', '•'],
            [' ', ' ', '•', ' ', ' ', ' ', ' ', ' ',
                '•', '•', '•', '•', '•', '•', 'D', '•'],
            ['•', '•', '•', ' ', ' ', ' ', ' ', ' ',
                ' ', ' ', ' ', ' ', ' ', ' ', ' ', '•'],
            ['•', ' ', 'X', ' ', ' ', ' ', ' ', ' ',
                ' ', ' ', ' ', ' ', '•', 'D', 'X', '•'],
            ['•', ' ', 'X', ' ', ' ', ' ', ' ', ' ',
                ' ', ' ', ' ', ' ', '•', ' ', ' ', '•'],
            ['•', ' ', 'D', ' ', ' ', ' ', ' ', ' ',
                ' ', ' ', ' ', ' ', '•', ' ', ' ', '•'],
            ['•', '•', '•', '•', '•', 'X', 'X', '•',
                'X', 'X', '•', '•', '•', '•', '•', '•'],
            [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
                ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
            [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
                ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
            [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
                ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
            [' ', ' ', '•', '•', '•', 'D', 'D', '•',
                '•', ' ', ' ', ' ', ' ', '•', '•', '•'],
            [' ', ' ', '•', ' ', ' ', ' ', ' ', ' ',
                '•', ' ', ' ', ' ', ' ', '•', ' ', '•'],
            [' ', ' ', '•', ' ', ' ', ' ', ' ', ' ',
                '•', '•', '•', '•', '•', '•', ' ', '•'],
            [' ', ' ', '•', '•', '•', 'D', 'D', '•',
                '•', ' ', ' ', ' ', ' ', ' ', ' ', '•'],
            [' ', ' ', 'X', ' ', ' ', ' ', ' ', ' ',
                ' ', ' ', ' ', ' ', '•', 'D', '•', '•'],
            [' ', '•', '•', ' ', ' ', ' ', ' ', ' ',
                ' ', ' ', ' ', ' ', '•', ' ', ' ', '•'],
            [' ', 'D', ' ', ' ', ' ', ' ', ' ', ' ',
                ' ', ' ', ' ', ' ', '•', ' ', ' ', '•'],
            [' ', '•', 'X', '•', '•', 'X', 'X', '•',
                '•', '•', '•', '•', '•', '•', '•', '•'],
            [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
                ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        ]
        mat_in = np.array([np.concatenate([[' ', ' '], x, [' ', ' ']])
                           for x in mat_in])
        for i in range(len(mat_in)):
            for j in range(len(mat_in[i])):
                if(mat_in[i][j] == 'D' or mat_in[i][j] == 'X'):
                    mat_in[i][j] = '•'

        return mat_in

    def process_input(self, input):
        n = len(input)
        m = len(input[0])
        input_patched = []
        for i in range(n-2):
            input_patched.append([])
            for j in range(m-2):
                input_patched[i].append(tuple(input[i:i+3, j:j+3].flatten()))

        # update dictionary of patches with their corresponding index (if new)
        for i in input_patched:
            for j in i:
                if(j not in self.patch_index):
                    self.patch_index[j] = len(self.patch_index)

        # add new patches to list and update global frequency of all seen patches
        self.patches += [None for _ in range(
            len(self.patch_index) - len(self.patches))]
        self.freqs += [0 for _ in range(len(self.patch_index) -
                                        len(self.freqs))]
        self.adjac += [[set(), set(), set(), set()]
                       for _ in range(len(self.patch_index) - len(self.adjac))]
        for i in input_patched:
            for j in i:
                self.patches[self.patch_index[j]] = j
                self.freqs[self.patch_index[j]] += 1

        # potential attempt at reducing the impact of basic patches that repeat often in favor of more complex ones
        def patch_entropy(n):
            count = {}
            for i in self.patches[n]:
                count[i] = count.setdefault(i, 0)+1
            prod = 1
            for i in count.keys():
                prod *= count[i]
            return math.log2(prod)
        # self.freqs = [x*patch_entropy(i) for i, x in enumerate(self.freqs)]
        # Activate by uncommenting ^^^

        self.logged_freqs = np.log2(1/np.array(self.freqs))

        # update adjacency list with new information from input
        indexed_input = [[self.patch_index[j]
                          for j in i] for i in input_patched]
        for i in range(n-2):
            for j in range(m-2):
                # top 0 right 1 bottom 2 left 3
                if i > 0:
                    self.adjac[indexed_input[i][j]][0].add(
                        indexed_input[i-1][j])
                if j < m-3:
                    self.adjac[indexed_input[i][j]][1].add(
                        indexed_input[i][j+1])
                if i < n-3:
                    self.adjac[indexed_input[i][j]][2].add(
                        indexed_input[i+1][j])
                if j > 0:
                    self.adjac[indexed_input[i][j]][3].add(
                        indexed_input[i][j-1])

    def init_output(self, rows, cols, init_world=None, free_value_index=None):

        self.domains = [[[True for _ in self.patches]
                         for _ in range(cols)] for _ in range(rows)]

        self.adjac_counter = [[[[len(a) for a in self.adjac[p]] for p in range(
            len(self.patches))] for _ in range(cols)] for _ in range(rows)]

        self.entropies = [[(-1, False) for _ in range(cols)]
                          for _ in range(rows)]

        if(init_world is not None):
            patched_init_world = []
            for i in range(min(len(init_world)-2, rows-2)):
                patched_init_world.append([])
                for j in range(min(len(init_world[i])-2, cols-2)):
                    patched_init_world[i].append(
                        tuple(init_world[i:i+3, j:j+3].flatten()))

            for i in range(len(patched_init_world)):
                for j in range(len(patched_init_world[i])):
                    if(init_world[i][j] != free_value_index):

                        # FAST, UGLY
                        # preset_patch = self.patch_index[patched_init_world[i][j]]
                        # domain_delta = self.domains[i][j] ^ np.where(
                        #     np.arange(len(self.patches)) == preset_patch, True, False)  # ^ XOR

                        # FAILS ON FIRST LOOP
                        # preset_patches = [True if p[0]==init_world[i][j] else False for p in self.patches]
                        # domain_delta = self.domains[i][j] ^ np.array(preset_patches)  # ^ XOR

                        # TODO: use pattern matching between fixed tiles and possible patches they could be in, improving compatibility with generated surroundings
                        def patch_match(p):
                            for k in range(len(p)):
                                if(patched_init_world[i][j][k] != free_value_index and p[k] != patched_init_world[i][j][k]):
                                    return False
                            return True
                        preset_patches = [patch_match(p) for p in self.patches]
                        domain_delta = self.domains[i][j] ^ np.array(preset_patches)  # ^ XOR
                        # print(patched_init_world[i][j], [p for n,p in enumerate(self.patches) if preset_patches[n]])

                        self.propagate((i, j), domain_delta)

    def observe(self):
        min_entropy = float("inf")
        # find selection of min-entropy candidate cells to observe
        candidates = []
        for i in range(len(self.domains)):
            for j in range(len(self.domains[i])):
                supersum = sum(self.domains[i][j])
                if(supersum == 0):
                    # no patch options exist for this cell, so quit
                    print(i, j)
                    return None
                if(supersum == 1):
                    # already collapsed
                    continue
                if(not self.entropies[i][j][1]):
                    temp = self.calc_entropy(self.domains[i][j])
                    self.entropies[i][j] = (temp, True)
                if(self.entropies[i][j][0] < min_entropy):
                    candidates = [(i, j)]
                    min_entropy = self.entropies[i][j][0]
                elif(self.entropies[i][j][0] == min_entropy):
                    candidates.append([i, j])
        # randomly chose a candidate and observe (collapse wave function) on them
        # with a random choice of possible patches using their probability in the input
        cand_pos = random.choice(candidates)
        patch_probs = np.array(
            self.domains[cand_pos[0]][cand_pos[1]])*self.freqs
        patch_probs = patch_probs / np.sum(patch_probs)
        chosen_patch = np.random.choice(
            range(len(self.patches)), p=patch_probs)

        # TODO: return cand_pos,domain_delta instead, so the domain isn't changed within observation, only chosen. It is changed in propagate()
        domain_delta = self.domains[cand_pos[0]][cand_pos[1]] ^ np.where(
            np.arange(len(self.patches)) == chosen_patch, True, False)  # ^ XOR
        assert sum(domain_delta) == sum(self.domains[cand_pos[0]][cand_pos[1]]) - sum(np.where(
            range(len(self.patches)) == chosen_patch, True, False))
        # print(chosen_patch,cand_pos)
        return cand_pos, domain_delta

    def propagate(self, start_pos, domain_delta):
        # (pos, tile to remove from domain)
        # print(entropies)
        stack = []
        # for each True in the domain_delta add the needed remove_tile_possibility arguments as a tuple in stack
        for i in range(len(domain_delta)):
            if(domain_delta[i]):
                stack.append((start_pos, i))
        for (x, y), t in stack:
            self.domains[x][y][t] = False
        while(len(stack) > 0):
            # perform tile removal, updating entropies that need recalculation, changed domains and appending new removals stack through reference
            # print(stack)
            cell_tile = stack.pop(-1)
            self.remove_tile_possibility(stack, *cell_tile)

        # DONE; TODO: Initialize array, where we store the count of the support, i.e. for a given cell/tile/side, we count how many tiles in the domain on adjacent cell can be placed next to the tile in question.
        #       go through adjacent cell's tile values that we know a removed tile on the current cell can be adjacent to and subtract 1. if any of the neighbor tile values are 0, we know this tile has no possible neighbors and can be removed from domain
        # print(domains)

    def remove_tile_possibility(self, stack, pos, tile_ind):
        # iterate through all 4 directions of tile,
        #       subtract 1 from adjac_counter of all possible tiles in neighbor's domain common to adjac,
        #       if 0, add neighbor (pos,tile) to stack
        x, y = pos
        neighbor_pos = []
        if(x > 0):
            neighbor_pos.append((x-1, y, 0))
        if(y < len(self.domains[0])-1):
            neighbor_pos.append((x, y+1, 1))
        if(x < len(self.domains)-1):
            neighbor_pos.append((x+1, y, 2))
        if(y > 0):
            neighbor_pos.append((x, y-1, 3))
        for i, j, d in neighbor_pos:
            for t in self.adjac[tile_ind][d]:
                if(self.domains[i][j][t]):
                    self.adjac_counter[i][j][t][(d+2) % 4] -= 1
                    if(self.adjac_counter[i][j][t][(d+2) % 4] == 0):
                        self.domains[i][j][t] = False
                        # assert sum(self.domains[i][j])>=1, f"{i,j,t, self.patches[t], stack}"
                        stack.append(((i, j), t))
                        self.entropies[i][j] = (-1, False)

    def calc_entropy(self, superpos):
        # (1 if possible patch else 0) * (Prob of patch) * (binary information provided by patch probability) -> Shannon Entropy
        entropy = sum(np.array(superpos)*self.freqs*self.logged_freqs)
        return entropy

    def is_collapsed(self):
        for i in self.domains:
            for j in i:
                # at least one uncollapsed cell
                if(sum(j) > 1):
                    return False
        return True

    def is_complete(self):
        for i in self.domains:
            for j in i:
                # at least one uncollapsed cell
                if(sum(j) != 1):
                    return False
        return True

    def one_index(self, arr):
        # for i in range(len(arr)):
        #     if(arr[i]==1):
        #         return i
        return int(np.nonzero(arr)[0])

    def generate_world(self, rows, cols, init_world=None, free_value_index=None):
        print("PROCESS")

        self.init_output(rows, cols, init_world, free_value_index)
        temp_doms = copy.deepcopy(self.domains)
        temp_adjcount = copy.deepcopy(self.adjac_counter)
        temp_ents = copy.deepcopy(self.entropies)

        c = 0
        while(not self.is_complete()):
            print("OUTPUT INIT")

            # self.init_output(rows, cols,init_world)
            # more efficient to copy than recalculate
            self.domains = copy.deepcopy(temp_doms)
            self.adjac_counter = copy.deepcopy(temp_adjcount)
            self.entropies = copy.deepcopy(temp_ents)

            print(f"COLLAPSING, loop {c}")
            # boolean represents if the entropy is up to date

            while(not self.is_collapsed()):
                temp = self.observe()
                if(temp):
                    collapsed_pos, domain_delta = temp
                else:
                    print("Impossible cell.")
                    break
                if(collapsed_pos == None):
                    # got stuck, found non-collapsable cell
                    return None
                self.propagate(collapsed_pos, domain_delta)
                c += 1

        print(f"Loops: {c}")
        print("CONVERTING")
        print(f"PATCHES: {len(self.patches)}")
        return np.array([[self.patches[self.one_index(j)][0] for j in i] for i in self.domains])
