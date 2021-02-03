import numba
import numpy
from numba import cuda
# from pyculib import blas, rand, sparse
import pygame


_N = 2
N = 32 * _N  # mapped to threadIdx atm
SCALE = 32

ITERATIONS = 2  # > iterations = quality


class Fluid:
    def __init__(self, diffusion, viscosity, dt):
        self.dt = dt
        self.diffusion = diffusion
        self.viscosity = viscosity

        self.s = numpy.zeros(N * N, dtype="float32")
        self.density = numpy.zeros(N * N, dtype="float32")
        self.rectangles = []

        self.vel_x = numpy.zeros(N * N, dtype="float32")
        self.vel_y = numpy.zeros(N * N, dtype="float32")

        self.vel_x0 = numpy.zeros(N * N, dtype="float32")
        self.vel_y0 = numpy.zeros(N * N, dtype="float32")

    def add_density(self, x, y, amount):
        self.density[IX(x, y)] += amount

    def add_velocity(self, x, y, amountX, amountY):
        self.vel_x[IX(x, y)] += amountX
        self.vel_y[IX(x, y)] += amountY

    def step(self):
        i = 10  # filler


def set_boundary(b, x):
    threadIdx_x = cuda.threadIdx.x
    threadIdx_y = cuda.threadIdx.y
    for iter in range(32 * _N):
        # Skip top edge and bottom edge
        # if cuda.threadIdx.y == 0 or cuda.threadIdx.y == 31:
        #     print("thread y = 0 or 31")
        #     print(x.density[:5])
        #     continue

        # Skip left edge and right edge
        if iter == 0 or iter == 32 * _N - 1:
            continue

        if b == 2:
            print("HOLY SHIT WE WINNIN!")

    return x


def IX(i, j):
    return i + (j * N)


def main():
    fluid = Fluid(0, 0.0001, 0.2)

    an_array = numpy.asarray([0 for i in range(32)])
    threadsperblock = 32
    blockspergrid = (len(an_array) + (threadsperblock - 1)) // threadsperblock
    print(threadsperblock, blockspergrid)
    set_boundary[blockspergrid, threadsperblock](2, fluid)

    print("Done. " + str(an_array[0]))


if __name__ == "__main__":
    main()