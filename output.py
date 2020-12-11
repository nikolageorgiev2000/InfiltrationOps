import numpy as np
import math
import cv2
import time
import wfc_gen
from multiprocessing import Pool
import concurrent.futures
import copy


GRID_ROWS = 2
GRID_COLS = 3

CHUNK_HEIGHT = 2
CHUNK_WIDTH = 3

INPUT_IMAGE = "floorplan_walls.png"
PRESET = "floorplan_borders.png"


def gen(g, preset, free_value_index):
    return (g.generate(CHUNK_HEIGHT, CHUNK_WIDTH, preset, free_value_index))


if __name__ == '__main__':

    image = cv2.imread(INPUT_IMAGE)
    cv2.imwrite(f"ExampleGens/input.png", image)
    preset = cv2.imread(PRESET)

    colors = set()
    for i in image:
        for j in i:
            colors.add(tuple(j))
    for i in preset:
        for j in i:
            colors.add(tuple(j))
    colors = list(colors)
    print(f"COLORS: {colors}")
    image = np.array([[colors.index(tuple(j)) for j in i] for i in image])
    preset = np.array([[colors.index(tuple(j)) for j in i] for i in preset])
    mat_in = image

    # wall: (0, 0, 0) -> black
    # ground: (255, 255, 255) -> white
    # free-tile: (255, 112, 255) -> pink
    # door: (39,115,176) -> brown
    # window: (235, 173, 64) -> blue

    # Add horizontal, vertical, diagonal flips to input matrix
    mat_horiz = mat_in[::-1]
    mat_diag = mat_horiz.transpose()[::-1].transpose()
    mat_vert = mat_diag[::-1]
    mat_90 = np.rot90(mat_in)
    mat_180 = np.rot90(mat_90)
    mat_270 = np.rot90(mat_180)

    def convert(arr, conv):
        return np.array([[conv[c] for c in i] for i in arr])

    generator = wfc_gen.WorldGen()

    start_processing_input = time.process_time_ns()
    generator.process_input(mat_in)
    generator.process_input(mat_horiz)
    generator.process_input(mat_diag)
    generator.process_input(mat_vert)
    generator.process_input(mat_90)
    generator.process_input(mat_180)
    generator.process_input(mat_270)
    end_processing_input = time.process_time_ns()
    print(f"{(end_processing_input - start_processing_input)/10**9} seconds to process input")

    start_gen = time.time()

    # multiple processes rather than threads, since Python can only have one thread reading instructions from GIL at a time, making performance equivalent to serial execution in this case
    with Pool() as p:
        systems = p.starmap(gen, [(copy.deepcopy(generator), preset, colors.index(
            (255, 112, 255))) for _ in range(GRID_ROWS*GRID_COLS)])

    end_gen = time.time()
    print(f"{(end_gen - start_gen)} seconds to generate world")

    cv2.namedWindow('image', cv2.WINDOW_GUI_NORMAL)

    def display(mat):
        # imshow is 0.0-1.0 for floats or 0-255 for ints
        cv2.imshow("image", mat)
        cv2.waitKey()

    output_images = np.array([convert(s, colors) for s in systems])

    for i in range(len(systems)):
        system = convert(systems[i], colors)
        # display(system)
        cv2.imwrite(f"ExampleGens/output{i}.png", system)

    out_rows = [np.hstack(output_images[GRID_COLS*i:GRID_COLS*(i+1)])
                for i in range(GRID_ROWS)]
    overall = np.concatenate(out_rows)

    display(overall)

    cv2.destroyAllWindows()
