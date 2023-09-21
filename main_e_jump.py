import pygame
import neat
import neat.math_util as mu
import visualize

import os
import sys
import random
import math
import numpy as np

import pickle

from chrome_dino_neat.constants import (
    SCREEN_HEIGHT, SCREEN_WIDTH, SCREEN,
    RUNNING, JUMPING, DUCKING,
    SMALL_CACTUS, LARGE_CACTUS, BIRD,
    CLOUD, BG
)
from chrome_dino_neat.dinosaur import Dinosaur
import matplotlib.pyplot as plt

pygame.init()

FONT = pygame.font.Font('assets/Grand9K Pixel.ttf', 20)

class Cloud:
    def __init__(self):
        self.x = SCREEN_WIDTH + random.randint(800, 1000)
        self.y = random.randint(50, 100)
        self.image = CLOUD
        self.width = self.image.get_width()

    def update(self):
        self.x -= game_speed
        if self.x < -self.width:
            self.x = SCREEN_WIDTH + random.randint(800, 2500)
            self.y = random.randint(25, 250)

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.x, self.y))


class Obstacle:
    def __init__(self, image, type):
        self.image = image
        self.type = type
        self.rect = self.image[self.type].get_rect()
        self.rect.x = SCREEN_WIDTH
        self.rect.height -= 5

    def update(self):
        self.rect.x -= game_speed
        if self.rect.x < -self.rect.width:
            obstacles.pop()

    def draw(self, SCREEN):
        SCREEN.blit(self.image[self.type], self.rect)


class SmallCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 325


class LargeCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 300


class Bird(Obstacle):
    def __init__(self, image):
        self.type = 0
        super().__init__(image, self.type)
        self.rect.y = 225
        self.index = 0

    def draw(self, SCREEN):
        if self.index >= 9:
            self.index = 0
        SCREEN.blit(self.image[self.index//5], self.rect)
        self.index += 1

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

GENS = 30
generation_scores = []
def eval_genomes(genomes, config):
    global game_speed, x_pos_bg, y_pos_bg, obstacles, dinosaurs, ge, nets, points, generation_scores, pop, scores
    clock = pygame.time.Clock()
    points = 0
    generation_scores = []

    obstacles = []
    dinosaurs = []
    ge = []
    nets = []

    x_pos_bg = 0
    y_pos_bg = 380
    game_speed = 20

    for genome_id, genome in genomes:
        dinosaurs.append(Dinosaur())
        ge.append(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        genome.fitness = 0

    def score():
        global points, game_speed
        points += 1
        if points % 100 == 0:
            game_speed += 1
        text = FONT.render(f'Points:  {str(points)}', True, (0, 0, 0))
        SCREEN.blit(text, (950, 50))

    def statistics():
        global dinosaurs, game_speed, ge
        text_1 = FONT.render(f'N. Dinosaurs:  {str(len(dinosaurs))}', True, (0, 0, 0))
        text_2 = FONT.render(f'Generation:  {pop.generation+1}', True, (0, 0, 0))

        SCREEN.blit(text_1, (650, 50))
        SCREEN.blit(text_2, (400, 50))

    def background():
        global x_pos_bg, y_pos_bg
        image_width = BG.get_width()
        SCREEN.blit(BG, (x_pos_bg, y_pos_bg))
        SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
        if x_pos_bg <= -image_width:
            x_pos_bg = 0
        x_pos_bg -= game_speed

    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        SCREEN.fill((255, 255, 255))
        
        if len(dinosaurs) == 0:
            break

        if len(obstacles) == 0:
            rand_int = random.randint(0, 2)
            if rand_int == 0:
                obstacles.append(SmallCactus(SMALL_CACTUS))
            elif rand_int == 1:
                obstacles.append(LargeCactus(LARGE_CACTUS))
            else:
                obstacles.append(Bird(BIRD))
                
        obstacle = obstacles[0]
        for i, dinosaur in enumerate(dinosaurs):
            output = nets[i].activate(((obstacle.rect.x - (dinosaur.rect.x + dinosaur.rect.width)) / SCREEN_WIDTH,
                                       obstacle.rect.y / SCREEN_HEIGHT,
                                       obstacle.rect.height / SCREEN_HEIGHT))
            output = np.clip(0.5 * output[0] + 0.5, 0, 1)
            if output > 0.5 and dinosaur.dino_run:
                dinosaur.n_jumps += 1
                dinosaur.dino_jump = True
                dinosaur.dino_run = False
            dinosaur.update2()
            dinosaur.draw(SCREEN)

        for obstacle in obstacles:
            obstacle.draw(SCREEN)
            obstacle.update()
            for i, dinosaur in enumerate(dinosaurs):
                ge[i].fitness = points * (1 - min(dinosaur.n_jumps / 120, 1))
                if dinosaur.rect.colliderect(obstacle.rect):
                    generation_scores.append(points)
                    dinosaurs.pop(i)
                    ge.pop(i)
                    nets.pop(i)
                
        statistics()
        score()
        background()
        
        clock.tick(30)
        pygame.display.update()


# Setup the NEAT Neural Network
def run(config_path, iteration):
    global pop, fitnesses
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pygame.time.delay(2000)
    best = pop.run(eval_genomes, GENS)
    
    print('\nBest genome:\n{!s}'.format(best))
    
    pickle.dump(best, open(f"nets/simple_jump/{iteration}.pkl", "wb"))
    
    visualize.plot_stats(stats, ylog=False, view=False, filename=f"stats/simple_jump/{iteration}.svg")
    node_names = {-1: 'x_diff', -2: 'object_y', -3: 'object_height', 0: 'jump'}
    visualize.draw_net(config, best, node_names=node_names, filename=f"graphs/simple_jump/{iteration}")
    

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    high_scores = []
    for it in range(int(sys.argv[1])):
        run(config_path, it)
        high_scores.append(max(generation_scores))
    pickle.dump(high_scores, open(f"stats/simple_jump/high_scores.pkl", "wb"))
