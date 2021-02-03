import numpy as np
import pygame
import time

pygame.init()

N = 64
iter = 8
SCALE = 16
window = pygame.display.set_mode((N * SCALE, N * SCALE))

set_boundary_calls = 0
lin_solve_calls = 0
project_calls = 0
advect_calls = 0
diffuse_calls = 0
set_boundary_time = 0.0
lin_solve_time = 0.0
project_time = 0.0
advect_time = 0.0
diffuse_time = 0.0

class Vector:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    @staticmethod
    def from_angle(angle, magnitude):
        self = Vector()
        self.x = magnitude * np.cos(angle)
        self.y = magnitude * np.sin(angle)
        return self

    def dot_product(self, magnitude):
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

    def add_density(self, x, y, amount):
        self.density[IX(x, y)] += amount

    def add_velocity(self, x, y, amountX, amountY):
        self.vel_x[IX(x, y)] += amountX
        self.vel_y[IX(x, y)] += amountY

    def render_density(self):
        for i in range(0, N):
            for j in range(0, N):
                density = int(np.floor(self.density[IX(i, j)]))

                if density > 255:
                    density = 255
                elif density < 0:
                    density = 0

                velocity = np.sqrt(np.square(self.vel_x[IX(i, j)]) + np.square(self.vel_y[IX(i, j)]))
                velocity = abs(velocity)
                velocity = int(np.floor(velocity))

                if velocity > 255:
                    velocity = 255
                elif velocity < 0:
                    velocity = 0

                # try:
                    # canvas.delete(self.rectangles[IX(i, j)])
                    # self.rectangles[IX(i, j)] =
                # if density != 0.0 or velocity != 0.0:
                pygame.draw.rect(window, (density, 0, velocity), (i * SCALE, j * SCALE, (i + 1) * SCALE, (j + 1) * SCALE))
                # except IndexError:
                    # self.rectangles.append(canvas.create_rectangle(x, y, x + SCALE, y + SCALE, fill=rgb_hex(d_, 100, 100)))


def set_boundary(b, x):
    global set_boundary_calls
    set_boundary_calls += 1

    start = time.time()

    if b == 2:
        for i in range(1, N - 1):
            x[IX(i, 0)] = -x[IX(i, 1)]
            x[IX(i, N - 1)] = -x[IX(i, N - 2)]
            x[IX(0, i)] = x[IX(1, i)]
            x[IX(N - 1, i)] = x[IX(N - 2, i)]

    else:
        for i in range(1, N - 1):
            x[IX(i, 0)] = x[IX(i, 1)]
            x[IX(i, N - 1)] = x[IX(i, N - 2)]
            x[IX(0, i)] = -x[IX(1, i)]
            x[IX(N - 1, i)] = -x[IX(N - 2, i)]

    x[IX(0, 0)] = 0.33 * (x[IX(0, 1)] + x[IX(1, 0)])
    x[IX(0, N - 1)] = 0.33 * (x[IX(1, N - 1)] + x[IX(0, N - 2)])
    x[IX(N - 1, 0)] = 0.33 * (x[IX(N - 2, 0)] + x[IX(N - 1, 1)])
    x[IX(N - 1, N - 1)] = 0.33 * (x[IX(N - 2, N - 1)] + x[IX(N - 1, N - 2)])

    end = time.time()

    global set_boundary_time
    set_boundary_time += end - start


def lin_solve(b, x, x0, a, c):
    global lin_solve_calls
    lin_solve_calls += 1

    start = time.time()

    for k in range(iter):
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                x[IX(i, j)] = (
                                      x0[IX(i, j)] +
                                      a * (
                                              x[IX(i + 1, j)] +
                                              x[IX(i - 1, j)] +
                                              x[IX(i, j + 1)] +
                                              x[IX(i, j - 1)]
                                      )
                               ) / c

        pause = time.time()
        set_boundary(b, x)
        unpause = time.time()
        start += unpause - pause

    end = time.time()

    global lin_solve_time
    lin_solve_time += end - start


def diffuse(b, x, x0, diff, dt):
    global diffuse_calls
    diffuse_calls += 1

    start = time.time()

    a = dt * diff * (N - 2) * (N - 2)

    pause = time.time()
    lin_solve(b, x, x0, a, 1 + 6 * a)
    unpause = time.time()
    start += unpause - pause

    end = time.time()

    global diffuse_time
    diffuse_time += end - start


def project(velocX, velocY, p, div):
    global project_calls
    project_calls += 1

    start = time.time()

    for j in range(1, N - 1):
        for i in range(1, N - 1):
            div[IX(i, j)] = -0.5 * (
                    velocX[IX(i + 1, j)] - velocX[IX(i - 1, j)] + velocY[IX(i, j + 1)] - velocY[IX(i, j - 1)]) / N
            p[IX(i, j)] = 0

    pause = time.time()
    set_boundary(0, div)
    set_boundary(0, p)
    lin_solve(0, p, div, 1, 6)
    unpause = time.time()
    start += unpause - pause

    for j in range(1, N - 1):
        for i in range(1, N - 1):
            velocX[IX(i, j)] -= 0.5 * (p[IX(i + 1, j)] - p[IX(i - 1, j)]) * N
            velocY[IX(i, j)] -= 0.5 * (p[IX(i, j + 1)] - p[IX(i, j - 1)]) * N

    pause = time.time()
    set_boundary(1, velocX)
    set_boundary(2, velocY)
    unpause = time.time()
    start += unpause - pause

    end = time.time()

    global project_time
    project_time += end - start


def advect(b, d, d0, velocX, velocY, dt):
    global advect_calls
    advect_calls += 1

    start = time.time()

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

    pause = time.time()
    set_boundary(b, d)
    unpause = time.time()
    start += unpause - pause

    end = time.time()

    global advect_time
    advect_time += end - start


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
    fs = FluidSquare(viscosity=0.000000001, diffusion=0, dt=0.2)
    print()

    cx = N // 2
    cy = N // 2

    t = 0
    run = True
    while run:
        # Add density
        for i in range(-1, 1):
            for j in range(-1, 1):
                fs.add_density(cx + i, cy + j, np.random.randint(0, 255))

        angle = np.pi / 2  # np.random.random() * np.pi * 2
        velocity = Vector.from_angle(angle, 100)
        # velocity.dotProduct(0.2)
        t += 0.01
        fs.add_velocity(cx, cy, velocity.x, velocity.y)

        fs.update()
        fs.render_density()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        pygame.display.update()

    print("Calls portfolio:")
    print("\tset_boundary: " + str(set_boundary_calls))
    print("\tdiffuse: " + str(diffuse_calls))
    print("\tadvect: " + str(advect_calls))
    print("\tproject: " + str(project_calls))
    print("\tlin_solve: " + str(lin_solve_calls))

    print()

    print("Time portfolio:")
    print("\tset_boundary: " + str(set_boundary_time))
    print("\tdiffuse: " + str(diffuse_time))
    print("\tadvect: " + str(advect_time))
    print("\tproject: " + str(project_time))
    print("\tlin_solve: " + str(lin_solve_time))

    pygame.quit()


if __name__ == "__main__":
    main()
