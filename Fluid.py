import numpy as np
from tkinter import *
from random import randint

N = 128
iter = 4
SCALE = 4


class FluidSquare:
    def __init__(self, diffusion, viscosity, dt):
        self.size = N
        self.dt = dt
        self.diffusion = diffusion
        self.viscosity = viscosity

        self.s = [[0 for j in range(N)] for i in range(N)]
        self.density = [[0 for j in range(N)] for i in range(N)]
        self.rectangles = []

        self.vel_x = [[0 for j in range(N)] for i in range(N)]
        self.vel_y = [[0 for j in range(N)] for i in range(N)]

        self.vel_x0 = [[0 for j in range(N)] for i in range(N)]
        self.vel_y0 = [[0 for j in range(N)] for i in range(N)]

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
        self.density[x][y] += amount

    def addVelocity(self, x, y, amountX, amountY):
        self.vel_x[x][y] += amountX
        self.vel_y[x][y] += amountY

    def renderD(self, canvas):
        for i in range(N):
            for j in range(N):
                x = i * SCALE
                y = j * SCALE
                d = self.density[i][j]
                d_ = int(np.floor(d))
                if d_ > 99:
                    d_ = 99

                if d_ != 0:
                    print("d = " + str(d_))

                try:
                    canvas.delete(self.rectangles[i + (j * self.size)])
                    self.rectangles[i + (j * self.size)] = canvas.create_rectangle(x, y, x + SCALE, y + SCALE, fill="gray" + str(randint(50, 99)))
                except IndexError:
                    # Ignore
                    print("Ignored")
                    self.rectangles.append(canvas.create_rectangle(x, y, x + SCALE, y + SCALE, fill="gray" + str(randint(50, 99))))


    def renderV(self, canvas):
        for i in range(N):
            for j in range(N):
                x = i * SCALE
                y = j * SCALE
                vx = self.vel_x[i][j]
                vy = self.vel_y[i][j]

                if not (abs(vx) < 0.1 and abs(vy) <= 0.1):
                    canvas.create_line(x, y, x + vx * SCALE, y + vy * SCALE, fill="white")


def set_boundary(b, x):
    for j in range(1, N - 1):
        for i in range(1, N - 1):
            if b == 2:
                x[i][0] = -x[i][1]
                x[i][N - 1] = -x[i][N - 2]
            else:
                x[i][0] = x[i][1]
                x[i][N - 1] = x[i][N - 2]

    for j in range(1, N - 1):
        for i in range(1, N - 1):
            if b == 1:
                x[0][j] = -x[1][j]
                x[N - 1][j] = -x[N - 1][j]
            else:
                x[0][j] = x[1][j]
                x[N - 1][j] = x[N - 1][j]

    x[0][0] = 0.33 * (x[0][1] + x[1][0])

    x[0][N - 1] = 0.33 * (x[1][N - 1] + x[0][N - 2])

    x[N - 1][0] = 0.33 * (x[N - 2][0] + x[N - 1][1])

    x[N - 1][N - 1] = 0.33 * (x[N - 2][N - 1] + x[N - 1][N - 2])


def lin_solve(b, x, x0, a, c):
    cRecip = 1.0 / c
    for k in range(iter):
        for j in range(1, N - 1):
            for i in range(1, N - 1):
                x[i][j] = (x0[i][j] + a * (x[i + 1][j] + x[i - 1][j] + x[i][j + 1] + x[i][j - 1])) * cRecip

        set_boundary(b, x)


def diffuse(b, x, x0, diff, dt):
    a = dt * diff * (N - 2) * (N - 2)
    lin_solve(b, x, x0, a, 1 + 6 * a)


def project(velocX, velocY, p, div):
    for j in range(1, N - 1):
        for i in range(1, N - 1):
            div[i][j] = -0.5 * (
                    velocX[i + 1][j] - velocX[i - 1][j] + velocY[i][j + 1] - velocY[i][j - 1]) / N
            p[i][j] = 0

    set_boundary(0, div)
    set_boundary(0, p)
    lin_solve(0, p, div, 1, 6)

    for j in range(1, N - 1):
        for i in range(1, N - 1):
            velocX[i][j] -= 0.5 * (p[i + 1][j] - p[i - 1][j]) * N
            velocY[i][j] -= 0.5 * (p[i][j + 1] - p[i][j - 1]) * N

    set_boundary(1, velocX)
    set_boundary(2, velocY)


def advect(b, d, d0, velocX, velocY, dt):
    dtx = dt * (N - 2)
    dty = dt * (N - 2)

    for j in range(1, N - 1):
        for i in range(1, N - 1):
            tmp1 = dtx * velocX[i][j]
            tmp2 = dty * velocY[i][j]
            x = i - tmp1
            y = j - tmp2

            if x < 0.5:
                x = 0.5
            if x > N + 0.5:
                x = N + 0.5

            i0 = int(np.floor(x))
            i1 = int(i0 + 1.0)

            if y < 0.5:
                y = 0.5
            if y > N + 0.5:
                y = N + 0.5

            j0 = int(np.floor(y))
            j1 = int(j0 + 1.0)

            s1 = x - i0
            s0 = 1.0 - s1
            t1 = y - j0
            t0 = 1.0 - t1

            d[i][j] = s0 * (t0 * (0 * d0[i][j] + 0 * d0[i][j]) + (t1 * (0 * d0[i][j] + 0 * d0[i][j]))) + \
                      s1 * (t0 * (0 * d0[i1][j1] + 0 * d0[i1][j1]) + (t1 * (0 * d0[i1][j1] + 0 * d0[i1][j1])))

    set_boundary(b, d)


def main():
    fs = FluidSquare(viscosity=0.0000001, diffusion=0, dt=0.2)
    fs.addDensity(50, 50, 100)

    root = Tk()
    root.geometry(str(N * SCALE) + "x" + str(N * SCALE))
    c = Canvas(root, height=N * SCALE, width=N * SCALE, bg="black")

    c.pack()

    for time in range(1):
        root.update_idletasks()
        root.update()

        cx = int((0.5 * N) / SCALE)
        cy = int((0.5 * N) / SCALE)
        for i in range(-1, 1):
            for j in range(-1, 1):
                fs.addDensity(cx + i, cy + j, randint(50, 150))

        for i in range(N):
            for j in range(N):
                if fs.density[i][j] != 0:
                    print("IT DOESNT EQUAL ZERO!!")

        fs.update()
        fs.renderD(c)
        fs.renderV(c)

    c.destroy()
    root.quit()


if __name__ == "__main__":
    main()
