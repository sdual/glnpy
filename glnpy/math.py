import math


def logit(p: float) -> float:
    return math.log(p / 1.0 - p)
