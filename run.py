import map
import output

start()

while(True):
    update()


def start():
    test_map = map.Map((5,5))
    test_map.print_map()

def update():

    output.display()