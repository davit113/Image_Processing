import math
PI = math.pi


def calculateObDistanceInPx(type: int, area: float) -> int:
    if(type == 0):   #triangle
        side = math.sqrt(area * 4 / math.sqrt(3))
        return 1 #to be implemented
    elif(type == 1):  #circle
        radius = math.sqrt((area/PI))
        return 1 #to be implemented
    else:
        return -1