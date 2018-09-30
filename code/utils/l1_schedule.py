import numpy as np


def logicnn(epoch, step, num_batches, value):
    return value ** (float(step) / num_batches)


def constant(epoch, step, num_batches, value):
    return 1.0


def step(epoch, step, num_batches, value):
    if epoch == 0:
        return 1.0
    else:
        return 0.0


def interpolate(epoch, step, num_batches, value):
    return value
