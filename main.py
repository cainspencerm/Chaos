from time import sleep

import numpy as np
from random import normalvariate, uniform
import math
from tkinter import *

LENGTH = 800
WIDTH = 1200


class Particle:
    def __init__(self, auto_init, canvas, x_pos=0, y_pos=0, oval=None, x_vel=0, y_vel=0, x_acc=0, y_acc=0, scale=0, alpha=0, lifetime=0):
        self.canvas = canvas
        self.x_pos = x_pos
        self.x_pos_birth = x_pos
        self.y_pos = y_pos
        self.y_pos_birth = y_pos
        self.oval = oval
        self.x_vel = x_vel
        self.y_vel = y_vel
        self.x_acc = x_acc
        self.y_acc = y_acc
        self.mass = scale
        self.alpha = alpha
        self.lifetime = lifetime
        self.movement = True

        if auto_init:
            self.respawn(canvas)

    def update(self):
        if not self.movement:
            return

        # Change velocity
        self.x_vel += self.x_acc
        self.y_vel += self.y_acc

        if math.sqrt(math.pow(self.x_acc, 2) + math.pow(self.y_acc, 2)) > 0.1:
            self.x_acc *= (1 - 0.5)
            self.y_acc *= (1 - 0.5)

        # Damping velocity
        self.x_vel *= (1 - 0.01)
        self.y_vel *= (1 - 0.01)

        # Change position
        self.x_pos += self.x_vel
        self.y_pos += self.y_vel

        self.canvas.delete(self.oval)
        self.oval = self.canvas.create_oval(math.floor(self.x_pos),
                                            math.floor(self.y_pos),
                                            math.floor(self.x_pos) + self.mass,
                                            math.floor(self.y_pos) + self.mass,
                                            fill="white")

        if self.x_pos > WIDTH or self.y_pos > LENGTH or self.x_pos < 0 or self.y_pos < 0:
            self.respawn(self.canvas)
            return

        # Count life down
        self.lifetime -= 1
        if self.lifetime == 0:
            self.respawn(self.canvas)

    def respawn(self, canvas):
        if self.oval is not None:
            canvas.delete(self.oval)

        self.canvas = canvas
        chooser = uniform(1, 4)
        if chooser == 1:  # x = 0, y = random
            self.x_pos = 0
            self.y_pos = uniform(0, LENGTH)
        elif chooser == 2:  # x = max, y = random
            self.x_pos = WIDTH
            self.y_pos = uniform(0, LENGTH)
        elif chooser == 3:  # x = random, y = 0
            self.x_pos = uniform(0, WIDTH)
            self.y_pos = 0
        else:  # x = random, y = max
            self.x_pos = uniform(0, WIDTH)
            self.y_pos = LENGTH

        self.x_pos = 0 # uniform(1, WIDTH - 1)
        self.y_pos = normalvariate(0, LENGTH)

        self.x_pos_birth = self.x_pos
        self.y_pos_birth = self.y_pos

        self.oval = canvas.create_oval(math.floor(self.x_pos),
                                       math.floor(self.y_pos),
                                       math.floor(self.x_pos) + self.mass,
                                       math.floor(self.y_pos) + self.mass,
                                       fill="white")
        self.x_vel = uniform(0, 0.1)
        self.y_vel = 0
        self.x_acc = 0.01 # uniform(-0.1, 0.1) * 0.1
        self.y_acc = 0 # uniform(-0.1, 0.1) * 0.1
        self.mass = uniform(1, 4)
        self.alpha = self.mass
        self.lifetime = 10000

    def attract(self, other):
        if self == other:
            return

        g = 0.000667
        r_dist = math.sqrt(math.pow((other.x_pos - self.x_pos), 2) + math.pow((other.y_pos - self.y_pos), 2))
        r_dist_x = other.x_pos - self.x_pos
        r_dist_y = other.y_pos - self.y_pos

        if r_dist_x == 0:
            r_dist_x = 1

        if r_dist == 0:
            r_dist = 1

        theta = math.atan(r_dist_y / r_dist_x)

        # get magnitude of force
        force = g * (self.mass * 10) * (other.mass * 10) / math.pow(r_dist, 2)

        acc_x = force * (math.cos(theta) * r_dist)
        acc_y = force * (math.sin(theta) * r_dist)

        self.x_acc += acc_x
        self.y_acc += acc_y

        other.x_acc -= acc_x
        other.y_acc -= acc_y


class Vortex:
    def __init__(self, x, y, size, canvas):
        self.x = x
        self.y = y
        self.size = size  # diameter
        self.mass = self.size / 10
        self.oval = canvas.create_oval(self.x - (self.size / 2), self.y - (self.size / 2), self.x + (self.size / 2), self.y + (self.size / 2), fill="blue")

    def inside(self, particle):
        return math.sqrt(pow(particle.x_pos - self.x, 2) + pow(particle.y_pos - self.y, 2)) < self.size / 2

    def repel(self, particle):
        g = 0.000667
        distance = math.sqrt(pow(particle.x_pos - self.x, 2) + pow(particle.y_pos - self.y, 2))
        distance *= 20
        force = g * self.mass * particle.mass / math.pow(distance, 2)

        dist_x = particle.x_pos - self.x
        dist_y = particle.y_pos - self.y
        theta = math.atan(dist_y / dist_x)

        acc_x = force * (abs(math.cos(theta)) * distance)
        acc_y = force * (abs(math.sin(theta)) * distance)

        if dist_x > 0:
            particle.x_acc += acc_x
        else:
            particle.x_acc -= acc_x

        if dist_y > 0:
            particle.y_acc += acc_y
        else:
            particle.y_acc -= acc_y


def create_particles(canvas, size=2):
    particles = []
    for i in range(0, size):
        particles.append(Particle(True, canvas))

    return particles


def main():
    root = Tk()
    root.geometry("1200x800")
    c = Canvas(root, height=LENGTH, width=WIDTH, bg="black")

    print("Creating particles...")
    particles = create_particles(c, 1000)
    vortexes = [Vortex(200, 400, 100, c),
                Vortex(400, 400, 100, c),
                Vortex(600, 400, 100, c),
                Vortex(800, 400, 100, c),
                Vortex(1000, 400, 100, c)]
    c.pack()

    while True:
        root.update_idletasks()
        root.update()
        # c.create_line(a.x_pos, a.y_pos, a.x_pos + a.x_vel + a.y_acc, a.y_pos + a.y_vel + a.y_acc, fill="white")

        for vortex in vortexes:
            for i in range(len(particles)):
                # for j in range(i, len(particles)):

                    # particles[i].attract(particles[j])

                vortex.repel(particles[i])
                particles[i].update()


if __name__ == "__main__":
    main()
