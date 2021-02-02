import numpy as np
from random import randint
from perlin_noise import PerlinNoise
import pygame

pygame.init()

N = 64
iter = 4
SCALE = 4
window = pygame.display.set_mode((N * SCALE, N * SCALE))


class Vector:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    @staticmethod
    def fromAngle(angle, magnitude):
        self = Vector()
        self.x = magnitude * np.cos(angle)
        self.y = magnitude * np.sin(angle)
        return self

    def dotProduct(self, magnitude):
        self.x *= magnitude
        self.y *= magnitude


class FluidSquare:
    def __init__(self, diffusion, viscosity, dt):
        self.size = N
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

    def update(self):
        diffuse(1, self.vel_x0, self.vel_x, self.viscosity, self.dt)
        diffuse(2, self.vel_y0, self.vel_y, self.viscosity, self.dt)

        project(self.vel_x0, self.vel_y0, self.vel_x, self.vel_y)

        advect(1, self.vel_x, self.vel_x0, self.vel_x0, self.vel_x0, self.dt)
        advect(2, self.vel_y, self.vel_y0, self.vel_y0, self.vel_y0, self.dt)

        project(self.vel_x, self.vel_y, self.vel_x0, self.vel_y0)

        diffuse(0, self.s, self.density, self.diffusion, self.dt)
        advect(0, self.density, self.s, self.vel_x, self.vel_y, self.dt)

    def addDensity(self, x, y, amount):
        self.density[IX(x, y)] += amount
        print("x = " + str(x))
        print("y = " + str(y))

    def addVelocity(self, x, y, amountX, amountY):
        self.vel_x[IX(x, y)] += amountX
        self.vel_y[IX(x, y)] += amountY

    def renderD(self):
        for i in range(0, N):
            for j in range(0, N):
                x = i * SCALE
                y = j * SCALE
                d = self.density[IX(i, j)]
                d_ = int(np.floor(d))
                if d_ > 255:
                    d_ = 255

                # try:
                    # canvas.delete(self.rectangles[IX(i, j)])
                    # self.rectangles[IX(i, j)] =
                pygame.draw.rect(window, (d_, d_, d_), (x, y, x + SCALE, y + SCALE))
                # except IndexError:
                    # self.rectangles.append(canvas.create_rectangle(x, y, x + SCALE, y + SCALE, fill=rgb_hex(d_, 100, 100)))

    def renderV(self, canvas):
        for i in range(N):
            for j in range(N):
                x = i * SCALE
                y = j * SCALE
                vx = self.vel_x[IX(i, j)]
                vy = self.vel_y[IX(i, j)]

                if not (abs(vx) < 0.1 and abs(vy) <= 0.1):
                    canvas.create_line(x, y, x + vx * SCALE, y + vy * SCALE, fill="white")


def set_boundary(b, x):
    for i in range(1, N - 1):
        if b == 2:
            x[IX(i, 0)] = -x[IX(i, 1)]
            x[IX(i, N - 1)] = -x[IX(i, N - 2)]
        else:
            x[IX(i, 0)] = x[IX(i, 1)]
            x[IX(i, N - 1)] = x[IX(i, N - 2)]

    for j in range(1, N - 1):
        for i in range(1, N - 1):
            if b == 1:
                x[IX(0, j)] = -x[IX(1, j)]
                x[IX(N - 1, j)] = -x[IX(N - 2, j)]
            else:
                x[IX(0, j)] = x[IX(1, j)]
                x[IX(N - 1, j)] = x[IX(N - 2, j)]

    x[IX(0, 0)] = 0.33 * (x[IX(0, 1)] + x[IX(1, 0)])
    x[IX(0, N - 1)] = 0.33 * (x[IX(1, N - 1)] + x[IX(0, N - 2)])
    x[IX(N - 1, 0)] = 0.33 * (x[IX(N - 2, 0)] + x[IX(N - 1, 1)])
    x[IX(N - 1, N - 1)] = 0.33 * (x[IX(N - 2, N - 1)] + x[IX(N - 1, N - 2)])


def lin_solve(b, x, x0, a, c):
    cRecip = 1.0 / c
    for k in range(iter):
        for j in range(1, N - 1):
            for i in range(1, N - 1):
                x[IX(i, j)] = (x0[IX(i, j)] + a * (x[IX(i + 1, j)] + x[IX(i - 1, j)] + x[IX(i, j + 1)] + x[IX(i, j - 1)])) * cRecip

        set_boundary(b, x)


def diffuse(b, x, x0, diff, dt):
    a = dt * diff * (N - 2) * (N - 2)
    lin_solve(b, x, x0, a, 1 + 6 * a)


def project(velocX, velocY, p, div):
    for j in range(1, N - 1):
        for i in range(1, N - 1):
            div[IX(i, j)] = -0.5 * (
                    velocX[IX(i + 1, j)] - velocX[IX(i - 1, j)] + velocY[IX(i, j + 1)] - velocY[IX(i, j - 1)]) / N
            p[IX(i, j)] = 0

    set_boundary(0, div)
    set_boundary(0, p)
    lin_solve(0, p, div, 1, 6)

    for j in range(1, N - 1):
        for i in range(1, N - 1):
            velocX[IX(i, j)] -= 0.5 * (p[IX(i + 1, j)] - p[IX(i - 1, j)]) * N
            velocY[IX(i, j)] -= 0.5 * (p[IX(i, j + 1)] - p[IX(i, j - 1)]) * N

    set_boundary(1, velocX)
    set_boundary(2, velocY)


def advect(b, d, d0, velocX, velocY, dt):
    for j in range(1, N - 1):
        for i in range(1, N - 1):
            x = i - dt * velocX[IX(i, j)]
            y = j - dt * velocY[IX(i, j)]

            if x < 0.5:
                x = 0.5
            if x > N - 2:
                x = N - 2

            i0 = np.floor(x)
            i0i = int(i0)
            i1 = i0 + 1.0
            i1i = int(i1)

            if y < 0.5:
                y = 0.5
            if y > N - 2:
                y = N - 2

            j0 = np.floor(y)
            j0i = int(j0)
            j1 = j0 + 1.0
            j1i = int(j1)

            s1 = x - i0
            s0 = 1.0 - s1
            t1 = y - j0
            t0 = 1.0 - t1

            d[IX(i, j)] = s0 * (t0 * d0[IX(i0i, j0i)] + t1 * d0[IX(i0i, j1i)]) + \
                          s1 * (t0 * d0[IX(i1i, j0i)] + t1 * d0[IX(i1i, j1i)])

    set_boundary(b, d)


def rgb_hex(r, g, b):
    if r > 255:
        r = 255
    elif r < 0:
        r = 0

    if g > 255:
        g = 255
    elif g < 0:
        g = 0

    if b > 255:
        b = 255
    elif b < 0:
        b = 0

    return "#%02x%02x%02x" % (r, g, b)


def IX(i, j):
    return i + (j * N)


def main():
    fs = FluidSquare(viscosity=0.0000001, diffusion=0, dt=0.2)
    print(N // 2)

    t = 0
    run = True
    while run:
        pygame.time.delay(100)

        # Add density
        cx = int((0.5 * N))
        cy = int((0.5 * N))
        for i in range(-1, 1):
            for j in range(-1, 1):
                fs.addDensity(cx + i, cy + j, 150)

        for i in range(32):
            angle = np.pi
            velocity = Vector.fromAngle(angle, 10)
            velocity.dotProduct(0.2)
            t += 0.01
            fs.addVelocity(cx, cy, velocity.x, velocity.y)

        fs.update()
        fs.renderD()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        pygame.display.update()

    pygame.quit()


if __name__ == "__main__":
    main()
