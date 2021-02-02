import numba
from pyculib import blas, rand, sparse
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

        self.s = [0.0 for i in range(N * N)]
        self.density = [0.0 for i in range(N * N)]
        self.rectangles = []

        self.vel_x = [0.0 for i in range(N * N)]
        self.vel_y = [0.0 for i in range(N * N)]

        self.vel_x0 = [0.0 for i in range(N * N)]
        self.vel_y0 = [0.0 for i in range(N * N)]

    def add_density(self, x, y, amount):
        self.density[IX(x, y)] += amount

    def add_velocity(self, x, y, amountX, amountY):
        self.vel_x[IX(x, y)] += amountX
        self.vel_y[IX(x, y)] += amountY

    def step(self):
        i = 10  # filler


def set_boundary(b, x):
    for iter in range(32 * _N):
        # Skip top edge and bottom edge
        # if threadIdx.y == 0 or threadIdx.y == 31:
        #     continue

        # Skip left edge and right edge
        # if iter == 0 or iter == 32 * _N - 1:
        #     continue

        # i = threadIdx.x + iter
        # j = threadIdx.y

        if b == 2:
            x[IX(i, j)] = -x[]



def IX(i, j):
    return i + (j * N)


def main():
    fluid = Fluid(0, 0.0000001, 0.2)


if __name__ == "__main__":
    main()