import numpy as np
import cv2
import time
import random
import world_gen

width = 50

world = world_gen.WorldGen()

image = cv2.imread("floorplan_walls.png")
colors = set()
for i in image:
    for j in i:
        colors.add(tuple(j))
colors = list(colors)
print(colors)
image = np.array([[colors.index(tuple(j)) for j in i] for i in image])
mat_in = image


# Add horizontal, vertical, diagonal flips to input matrix
mat_horiz = mat_in[::-1]
mat_diag = mat_horiz.transpose()[::-1].transpose()
mat_vert = mat_diag[::-1]
mat_90 = np.rot90(mat_in)
mat_180 = np.rot90(mat_90)
mat_270 = np.rot90(mat_180)


def convert(arr,conv):
    return np.array([[conv[c] for c in i] for i in arr])

world.process_input(mat_in)
world.process_input(mat_horiz)
world.process_input(mat_diag)
world.process_input(mat_vert)
world.process_input(mat_90)
world.process_input(mat_180)
world.process_input(mat_270)

preset = np.array([[colors.index(tuple(j)) for j in i] for i in cv2.imread("floorplan_borders32.png")])

# world.process_input(preset)
# system = world.generate_world(64, 64, preset)
system = world.generate_world(64, 64, preset)

cv2.namedWindow('image', cv2.WINDOW_GUI_NORMAL)


def display(mat):
    #imshow is 0.0-1.0 for floats or 0-255 for ints
    cv2.imshow("image",mat)
    cv2.waitKey()


display(convert(mat_in,colors))
# display(convert(mat_horiz,colors))
# display(convert(mat_diag,colors))
# display(convert(mat_vert,colors))
# display(convert(mat_180,colors))
# display(convert(mat_270,colors))


system = convert(system,colors)
display(system)

cv2.destroyAllWindows()