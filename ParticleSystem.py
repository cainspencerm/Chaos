class ParticleSystem:
    def __init__(self):
        self.particles = []

    def draw(self):
        return

    def update(self):
        # Remove dead particles
        for particle in self.particles:
            if particle.lifespan <= 0:
                self.particles.remove(particle)
            else:
                # Update acc, vel, pos
                particle.update()

    def add(self, particle):
        self.particles.append(particle)


class Particle:
    def __init__(self):
        self.pos = 0, 0  # x, y
        self.vel = 0, 0  # r, theta
        self.acc = 0, 0  # r, theta

        self.lifespan = 255  # frames

    def update(self):
        self.vel += self.acc  # TODO Calculate polar coordinates
        self.pos += self.vel  # TODO Calculate polar coordinates

        self.lifespan -= 1

