import pygame
import os, sys
import random
import math

from chrome_dino_neat.constants import (
    SCREEN_HEIGHT, SCREEN_WIDTH, SCREEN,
    RUNNING, JUMPING, DUCKING,
    SMALL_CACTUS, LARGE_CACTUS, BIRD,
    CLOUD, BG
)
from chrome_dino_neat.dinosaur import Dinosaur

import neat
import pickle

pygame.init()


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
        self.rect.y = 240
        self.index = 0

    def draw(self, SCREEN):
        if self.index >= 9:
            self.index = 0
        SCREEN.blit(self.image[self.index//5], self.rect)
        self.index += 1

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def main(name):
    global game_speed, x_pos_bg, y_pos_bg, points, obstacles
    run = True
    clock = pygame.time.Clock()
    player = Dinosaur(draw_name=True)
    player.color = (0, 0, 255)
    ai = Dinosaur("ai", True)
    ai.color = (255, 0, 0)
    ai.name_offset = -25
    genome = pickle.load(open(f"nets/{name}.pkl", "rb"))
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "config.txt"
    )
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    clouds = [Cloud() for i in range(5)]
    game_speed = 20
    x_pos_bg = 0
    y_pos_bg = 380
    points = 0
    font = pygame.font.Font('assets/Grand9K Pixel.ttf', 20)
    big_font = pygame.font.Font('assets/Grand9K Pixel.ttf', 50)
    obstacles = []

    def score():
        global points, game_speed
        points += 1
        if points % 50 == 0:
            game_speed += 1

        text = font.render("Points: " + str(points), True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.center = (1000, 40)
        SCREEN.blit(text, textRect)

    def background():
        global x_pos_bg, y_pos_bg
        image_width = BG.get_width()
        SCREEN.blit(BG, (x_pos_bg, y_pos_bg))
        SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
        if x_pos_bg <= -image_width:
            SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
            x_pos_bg = 0
        x_pos_bg -= game_speed

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()

        SCREEN.fill((255, 255, 255))
        userInput = pygame.key.get_pressed()
        
        background()
        
        for c in clouds:
            c.draw(SCREEN)
            c.update()
        
        player.update(userInput)
        player.draw(SCREEN)

        score()

        if len(obstacles) == 0:
            if random.randint(0, 2) == 0:
                obstacles.append(SmallCactus(SMALL_CACTUS))
            elif random.randint(0, 2) == 1:
                obstacles.append(LargeCactus(LARGE_CACTUS))
            elif random.randint(0, 2) == 2:
                obstacles.append(Bird(BIRD))
                
        if len(obstacles) > 0:
            output = net.activate(((obstacles[0].rect.x - (ai.rect.x + ai.rect.width)) / SCREEN_WIDTH,
                                    obstacles[0].rect.y / SCREEN_HEIGHT,
                                    obstacles[0].rect.height / SCREEN_HEIGHT))
            output = sigmoid(output[0])
            if output > 0.5 and ai.dino_run:
                ai.dino_jump = True
                ai.dino_run = False
        
        ai.update2()
        ai.draw(SCREEN)

        for obstacle in obstacles:
            obstacle.draw(SCREEN)
            obstacle.update()
            if player.rect.colliderect(obstacle.rect):
                text = big_font.render("Player lost", True, (255, 0, 0))
                textRect = text.get_rect()
                textRect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2) 
                SCREEN.blit(text, textRect)
                pygame.display.update()
                pygame.time.delay(5000)
                sys.exit()
            elif ai.rect.colliderect(obstacle.rect):
                text = big_font.render("AI lost", True, (0, 0, 255))
                textRect = text.get_rect()
                textRect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
                SCREEN.blit(text, textRect)
                pygame.display.update()
                pygame.time.delay(5000)
                sys.exit()

        clock.tick(30)
        pygame.display.update()

main(sys.argv[1])
