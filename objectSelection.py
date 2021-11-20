import random

def objectSelection(candidates):
    if candidates is None:
        return None
    

    return candidates[0]


def objectSelection_from_yolo(candidates):
    if candidates is None:
        return None
    pred = candidates.pred[0]
    return candidates[0]