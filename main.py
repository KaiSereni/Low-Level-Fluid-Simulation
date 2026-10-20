from dataclasses import dataclass
import taichi as ti
from taichi import math as tm
import numpy as np
import random

ti.init(arch=ti.metal)

TiVectorList = ti.types.ndarray(dtype=ti.f16, ndim=2) # each particle has a position vector
TiParticleStateMatrix = ti.types.ndarray(dtype=ti.f16, ndim=3) # each particle has a list containing a position vector and a velocity vector

TiPixelArray = ti.types.ndarray(dtype=ti.f16, ndim=3) # each x, y has a vec3

DT = np.float16(0.5)
 
@dataclass
class Particle:
    p: list[float]
    v: list[float]

    def __init__(self, p: list[float], v: list[float]=[0.0, 0.0]):
        self.p = p
        self.v = v

@ti.kernel
def increment_fluid(particle_states: TiParticleStateMatrix, sceneWidth: ti.i16, sceneHeight: ti.i16):
    for i in range(particle_states.shape[0]):
        # particle index, p/v, x/y
        particle_states[i, 0, 0] = particle_states[i, 0, 0] + ti.float16(particle_states[i, 1, 0] * DT)
        particle_states[i, 0, 1] = particle_states[i, 0, 1] + ti.float16(particle_states[i, 1, 1] * DT)

        if (particle_states[i, 0, 0] < 0):
            particle_states[i, 0, 0] = 0
            particle_states[i, 1, 0] = ti.abs(particle_states[i, 1, 0])
            particle_states[i, 1, 1] *= ti.float16(0.98)

        elif (particle_states[i, 0, 0] > sceneWidth):
            particle_states[i, 0, 0] = sceneWidth
            particle_states[i, 1, 0] = -ti.abs(particle_states[i, 1, 0])
            particle_states[i, 1, 1] *= ti.float16(0.98)

        if (particle_states[i, 0, 1] < 0):
            particle_states[i, 0, 1] = 0
            particle_states[i, 1, 1] = ti.abs(particle_states[i, 1, 1])
            particle_states[i, 1, 0] *= ti.float16(0.98)

        elif (particle_states[i, 0, 1] > sceneHeight):
            particle_states[i, 0, 1] = sceneHeight
            particle_states[i, 1, 1] = -ti.abs(particle_states[i, 1, 1])
            particle_states[i, 1, 0] *= ti.float16(0.98)
        
        particle_states[i, 1, 1] -= ti.float16(5.0 * DT)

@ti.func
def comparePixels(target: TiPixelArray, x: ti.i16, y: ti.i16, value: ti.types.vector(3, ti.f16)) -> bool: #type: ignore
    o = True
    for i in range(target.shape[2]):
        if not target[x, y, i] == value[i]:
            o = False
            break
    return o

@ti.func
def assignPixel(target: TiPixelArray, x: ti.i16, y: ti.i16, value: ti.types.vector(3, ti.f16)): #type: ignore
    for i in range(target.shape[2]):
        target[x, y, i] = value[i]

@ti.kernel
def draw_ui(pixels: TiPixelArray, particle_positions: TiVectorList, part_color: ti.types.vector(3, ti.f16), part_rad: ti.i16): #type: ignore
    screen_shape = pixels.shape
    for x in range(screen_shape[0]-1):
        for y in range(screen_shape[1]-1):
            if not comparePixels(pixels, x, y, part_color):
                for part_index in range(particle_positions.shape[0]):
                    if ti.pow(particle_positions[part_index, 0] - x, 2) + ti.pow(particle_positions[part_index, 1] - y, 2) < ti.pow(part_rad, 2):
                        assignPixel(pixels, x, y, part_color)

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
        self.part_color = [0, 0, 1]
        self.particles = []
        self.scene_width = w
        self.scene_height = h
        self.t = 0
        self.gui = ti.GUI("fluid", res=(w, h))
        self.gui.fps_limit = 60
        self.pixels = np.zeros(shape=(self.scene_width, self.scene_height, 3), dtype=np.float16)

        for i in range(particle_count):
            self.particles.append(Particle(
                [
                    random.random() * w, 
                    random.random() * h
                ],
                [
                    random.random() - 5.0,
                    random.random() - 5.0
                ]
            ))

    def run(self):
        particle_states = []
        for particle in self.particles:
            particle_states.append(
                [particle.p, particle.v]
            )
        particle_states = np.array(particle_states, dtype=np.float16)
        it = 0
        while self.gui.running:
            it+=1
            if it % 300 == 0:
                print(particle_states.shape)
            self.t += DT
            increment_fluid(
                particle_states,
                self.scene_width,
                self.scene_height
            )
            self.pixels = np.ones(shape=(self.scene_width, self.scene_height, 3), dtype=np.float16)
            draw_ui(
                self.pixels,
                np.ascontiguousarray(particle_states[:, 0]),
                tm.vec3(self.part_color),
                10
            )
            self.gui.set_image(self.pixels)
            self.gui.show()

s = FluidScene(
    w=800,
    h=600,
    particle_count=30
)

s.run()