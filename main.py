from dataclasses import dataclass
import taichi as ti
from taichi import math as tm
import numpy as np
import random

ti.init(arch=ti.metal)

TiParticleStateMatrix = ti.types.ndarray(dtype=ti.f32, ndim=3) # each particle has a list containing a position vector and a velocity vector

DT = np.float32(0.03)
G = np.float32(400.0)
BORDER_INFLUENCE_RANGE = np.float32(20.0)
PART_INFLUENCE_RANGE = np.float32(45.0)
PART_FORCE = np.float32(1.1)
BOUNCE_DAMP = np.float32(0.95)
FRICTION = np.float32(0.95)
BROWNIAN_MOTION = np.float32(0.2)
 
@dataclass
class Particle:
    p: list[float]
    v: list[float]

    def __init__(self, p: list[float], v: list[float]=[0.0, 0.0]):
        self.p = p
        self.v = v

@ti.func
def distance_to_force(d: ti.f32) -> ti.f32:
    o = 0.0
    d = ti.abs(d)
    if d < 1:
        o = -4 * pow(d - 1, 3)
    return o

@ti.kernel
def increment_fluid(particle_states: TiParticleStateMatrix, sceneWidth: ti.i32, sceneHeight: ti.i32):
    num_parts = particle_states.shape[0]
    for i in range(num_parts):
        part_x = particle_states[i, 0, 0]
        part_y = particle_states[i, 0, 1]

        if part_x < BORDER_INFLUENCE_RANGE:
            particle_states[i, 1, 0] += distance_to_force(part_x/BORDER_INFLUENCE_RANGE)*BORDER_INFLUENCE_RANGE * DT
        if part_y < BORDER_INFLUENCE_RANGE:
            particle_states[i, 1, 1] += distance_to_force(part_y/BORDER_INFLUENCE_RANGE)*BORDER_INFLUENCE_RANGE * DT
        if part_x > sceneWidth - BORDER_INFLUENCE_RANGE:
            particle_states[i, 1, 0] -= distance_to_force((sceneWidth-part_x)/BORDER_INFLUENCE_RANGE)*BORDER_INFLUENCE_RANGE * DT
        if part_y > sceneHeight - BORDER_INFLUENCE_RANGE:
            particle_states[i, 1, 1] -= distance_to_force((sceneWidth-part_y)/BORDER_INFLUENCE_RANGE)*BORDER_INFLUENCE_RANGE * DT

        for other_particle_index in range(num_parts):
            if other_particle_index != i:
                relative_pos = tm.vec2(
                    part_x - particle_states[other_particle_index, 0, 0],
                    particle_states[i, 0, 1] - particle_states[other_particle_index, 0, 1]
                )
                r = relative_pos.norm()
                force = distance_to_force(r/PART_INFLUENCE_RANGE)*PART_INFLUENCE_RANGE
                particle_states[i, 1, 0] += relative_pos[0] * force * DT * PART_FORCE
                particle_states[i, 1, 1] += relative_pos[1] * force * DT * PART_FORCE
        

        particle_states[i, 0, 0] = particle_states[i, 0, 0] + ti.float32(particle_states[i, 1, 0] * DT)
        particle_states[i, 0, 1] = particle_states[i, 0, 1] + ti.float32(particle_states[i, 1, 1] * DT)
        
        particle_states[i, 1, 1] -= ti.float32(G * DT)

        # brownian boop
        particle_states[i, 1, 0] += ti.randn(dt=ti.float32) * BROWNIAN_MOTION
        particle_states[i, 1, 1] += ti.randn(dt=ti.float32) * BROWNIAN_MOTION

        if (particle_states[i, 0, 0] < 0):
            particle_states[i, 0, 0] = 0
            particle_states[i, 1, 0] = ti.abs(particle_states[i, 1, 0]) * BOUNCE_DAMP
            particle_states[i, 1, 1] *= ti.float32(FRICTION)

        elif (particle_states[i, 0, 0] > sceneWidth):
            particle_states[i, 0, 0] = sceneWidth
            particle_states[i, 1, 0] = -ti.abs(particle_states[i, 1, 0]) * BOUNCE_DAMP
            particle_states[i, 1, 1] *= ti.float32(FRICTION)

        if (particle_states[i, 0, 1] < 0):
            particle_states[i, 0, 1] = 0
            particle_states[i, 1, 1] = ti.abs(particle_states[i, 1, 1]) * BOUNCE_DAMP
            particle_states[i, 1, 0] *= ti.float32(FRICTION)

        elif (particle_states[i, 0, 1] > sceneHeight):
            particle_states[i, 0, 1] = sceneHeight
            particle_states[i, 1, 1] = -ti.abs(particle_states[i, 1, 1]) * BOUNCE_DAMP
            particle_states[i, 1, 0] *= ti.float32(FRICTION)

@ti.kernel
def draw_ui(pixels: ti.template(), particle_positions: TiParticleStateMatrix, part_color: ti.types.vector(3, ti.f32), part_rad: ti.i32): #type: ignore
    screen_x, screen_y = pixels.shape[:2]
    num_parts = particle_positions.shape[0]
    for part_index in range(num_parts):
        particle_x = ti.cast(particle_positions[part_index, 0, 0], ti.i32)
        particle_y = ti.cast(particle_positions[part_index, 0, 1], ti.i32)
        for i in range(-part_rad, part_rad):
            for j in range(-part_rad, part_rad):
                if ti.pow(i, 2) + ti.pow(j, 2) < ti.pow(part_rad, 2):
                    if \
                        (particle_x + i < screen_x) \
                        and (particle_y + j < screen_y) \
                        and (particle_x + i >= 0) \
                        and (particle_y + j >= 0) \
                    :
                        pixels[particle_x + i, particle_y + j] = part_color

class FluidScene:
    part_radius: int
    part_color: list[float]
    particles: list[Particle]
    scene_width: int
    scene_height: int
    t: float
    gui: ti.GUI
    pixels: ti.Field

    def __init__(self, w: int, h: int, particle_count: int):
        self.part_radius = 10
        self.part_color = tm.vec3([0.0, 0.0, 1.0])
        self.particles = []
        self.scene_width = w
        self.scene_height = h
        self.t = 0
        self.gui = ti.GUI("fluid", res=(w, h))
        self.gui.fps_limit = 60
        self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=(self.scene_width, self.scene_height))

        for i in range(particle_count):
            self.particles.append(Particle(
                [
                    random.random() * w, 
                    random.random() * h
                ],
                [
                    random.random() - 1.0,
                    random.random() - 1.0
                ]
            ))

    def run(self):
        particle_states = []
        for particle in self.particles:
            particle_states.append(
                [particle.p, particle.v]
            )
        particle_states = np.array(particle_states, dtype=np.float32)
        while self.gui.running:
            self.t += DT
            increment_fluid(
                particle_states,
                self.scene_width,
                self.scene_height
            )
            draw_ui(
                self.pixels,
                particle_states,
                self.part_color,
                6
            )
            self.gui.set_image(self.pixels)
            self.gui.show()
            self.pixels.fill(1.0)

s = FluidScene(
    w=800,
    h=600,
    particle_count=400
)

s.run()