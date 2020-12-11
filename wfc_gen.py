# Wave Function Collapse lib
import numpy as np
import random
import copy
import math
import sys


class WorldGen:

    def __init__(self):
        """ Input example worlds in 2D grid using process_input, then generate new ones using generate. """

        self.patch_index = {}
        self.patches = []
        self.freqs = []
        self.logged_freqs = []
        self.adjac = []

        self.domains = []
        self.adjac_counter = []
        self.entropies = []

    def process_input(self, input):
        """ Iterates through each 3x3 patch of tile grid input assigning them unique indices (patch_index), keeps track of their occurence count (frequency) and possible adjacent patches (adjac)."""
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

        self.logged_freqs = np.log2(np.array(self.freqs))

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

    def init_output(self, rows, cols, init_tiles=None, free_value_index=None):
        """ Initialize output for generation function with a domain, adjac_counter and entropy value for each tile in output grid. Propagate any preset tiles (init_tiles) to reflect changes in output. """

        self.domains = [[[True for _ in self.patches]
                         for _ in range(cols)] for _ in range(rows)]

        self.adjac_counter = [[[[len(a) for a in self.adjac[p]] for p in range(
            len(self.patches))] for _ in range(cols)] for _ in range(rows)]

        old_adjac_counter = copy.deepcopy(self.adjac_counter)

        self.entropies = [[(-1, False) for _ in range(cols)]
                          for _ in range(rows)]

        if(init_tiles is not None):
            patches_init_tiles = []
            for i in range(min(len(init_tiles)-2, rows)):
                patches_init_tiles.append([])
                for j in range(min(len(init_tiles[i])-2, cols)):
                    patches_init_tiles[i].append(
                        tuple(init_tiles[i:i+3, j:j+3].flatten()))

            for i in range(len(patches_init_tiles)):
                for j in range(len(patches_init_tiles[i])):
                    if(init_tiles[i][j] != free_value_index):

                        # use pattern matching between fixed tiles and possible patches they could be in, improving compatibility with generated surroundings
                        def patch_match(p):
                            for k in range(len(p)):
                                if(patches_init_tiles[i][j][k] != free_value_index and p[k] != patches_init_tiles[i][j][k]):
                                    return False
                            return True
                        preset_patches = [patch_match(p) for p in self.patches]
                        # any 0 in preset that is 1 in domain needs to be 1 in delta (i.e. patch removed)
                        domain_delta = self.domains[i][j] & ~np.array(
                            preset_patches)  # ^ XOR

                        self.propagate((i, j), domain_delta)

    def observe(self):
        """ Find the tile which has the least entropy of possible patches that it is the top left tile to and randomly select one of those patches using their frequency. Return the tile position and the delta in domain required (only True for the selected patch now) """
        min_entropy = float("inf")
        # find selection of min-entropy candidate cells to observe
        candidates = []
        for i in range(len(self.domains)):
            for j in range(len(self.domains[i])):
                supersum = sum(self.domains[i][j])
                if(supersum == 0):
                    # no patch options exist for this cell, so quit
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
        domain_delta = self.domains[cand_pos[0]][cand_pos[1]] ^ np.where(
            np.arange(len(self.patches)) == chosen_patch, True, False)  # ^ XOR
        assert sum(domain_delta) == sum(self.domains[cand_pos[0]][cand_pos[1]]) - sum(np.where(
            range(len(self.patches)) == chosen_patch, True, False))
        return cand_pos, domain_delta

    def propagate(self, start_pos, domain_delta):
        """ Take the domain delta and apply the change to the tile position given. Maintain a stack as a depth first search removes tile possibilities from its neighbors. """
        # (pos, patch to remove from domain)
        stack = []
        # for each True in the domain_delta add the needed remove_tile_possibility arguments as a tuple in stack
        for i in range(len(domain_delta)):
            if(domain_delta[i]):
                stack.append((start_pos, i))
        # update selected patch domain
        for (x, y), p in stack:
            self.domains[x][y][p] = False
        while(len(stack) > 0):
            # perform tile removal, updating entropies that need recalculation, changed domains and appending new removals stack through reference
            cell_tile = stack.pop(-1)
            self.remove_tile_possibility(stack, *cell_tile)

    def remove_tile_possibility(self, stack, pos, patch_ind):
        """ The tile_ind at the given position has been marked as not possible. This affects the possible patches the neighboring tiles could be top left of, since if a patch has 0 possible neighbors it cannot exist either. """
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
            for p in self.adjac[patch_ind][d]:
                if(self.domains[i][j][p]):
                    # update possible adjacent patches in opposite direction (d+2)%4 now that the input patch is not possible at pos
                    self.adjac_counter[i][j][p][(d+2) % 4] -= 1
                    # if there are no available adjacent patches p can be a neighbor of, it is not possible either
                    if(self.adjac_counter[i][j][p][(d+2) % 4] == 0):
                        self.domains[i][j][p] = False
                        stack.append(((i, j), p))
                        self.entropies[i][j] = (-1, False)

    def calc_entropy(self, superpos):
        """ Calculate the entropy of a tile in output grid using its possible patches. The lower the entropy, . """
        # (1 if possible patch else 0) * (Prob of patch) * (binary information provided by patch probability) -> Shannon Entropy
        entropy = -sum(np.array(superpos)*self.freqs*self.logged_freqs)
        return entropy

    def is_collapsed(self):
        """ Check each tile has collapsed i.e. no tile has more than 1 possible patch """
        for i in self.domains:
            for j in i:
                # at least one uncollapsed cell
                if(sum(j) > 1):
                    return False
        return True

    def is_complete(self):
        """ Check output is ready i.e. each tile has exactly one possible patch. If they have less, there is no possible patch at that location. """
        for i in self.domains:
            for j in i:
                # cell is either uncollapsed or impossible to select a patch for e.g. other patch selections made the tile domain fully False
                if(sum(j) != 1):
                    return False
        return True

    def generate(self, rows, cols, init_tiles=None, free_value_index=None):
        """ Takes output size in rows, cols and gives the option for preset tiles to accomodate for, as well as a free_value tile, where the algorithm can choose what to place according to the preset tiles. """

        print("INIT OUTPUT")
        self.init_output(rows, cols, init_tiles, free_value_index)

        temp_doms = copy.deepcopy(self.domains)
        temp_adjcount = copy.deepcopy(self.adjac_counter)
        temp_ents = copy.deepcopy(self.entropies)

        c = 0
        while(not self.is_complete()):
            # more efficient to copy than recalculate
            self.domains = copy.deepcopy(temp_doms)
            self.adjac_counter = copy.deepcopy(temp_adjcount)
            self.entropies = copy.deepcopy(temp_ents)

            print(f"COLLAPSING, loop {c}")

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
        return np.array([[self.patches[int(np.nonzero(j)[0])][0] for j in i] for i in self.domains])
