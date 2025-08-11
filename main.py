import pygame
import random
from collections import namedtuple
from enum import Enum
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25) # font for score

class Direction(Enum):
    RIGHT = 1 
    LEFT = 2 
    UP = 3
    DOWN = 4 # just directions


Point = namedtuple('Point', 'x, y') # easy way to store points



# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20 # size of each block
SPEED = 90 # game speed



class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h)) # make the window
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock() # for controlling speed
        self.reset()


    def reset(self):
        # init the game state
        self.direction = Direction.RIGHT # always start going right

        self.head = Point(self.w / 2, self.h / 2) # start in the middle
        self.snake = [self.head,
                    Point(self.head.x - BLOCK_SIZE, self.head.y),
                    Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)] # snake body
        self.score = 0
        self.food = None
        self._place_food() # drop the first food
        self.frameIteration = 0 # how many frames since last food


    def _place_food(self):
        # randomly place food somewhere not on the snake
        x = random.randint(0, int((self.w - BLOCK_SIZE) / BLOCK_SIZE)) * BLOCK_SIZE
        y = random.randint(0, int((self.h - BLOCK_SIZE) / BLOCK_SIZE)) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()


    def play_step(self, action):
        self.frameIteration += 1 # count frames
        # check for quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        # move
        self._move(action) # updates the head
        self.snake.insert(0, self.head)

        # check if game is over
        reward = 0
        game_over = False
        if self.is_collision() or self.frameIteration > 100*len(self.snake):
            game_over = True
            reward = -10 # bad move
            return reward, game_over, self.score
        # place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10 # yum
            self._place_food()
        else:
            self.snake.pop()
        # update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # return game over and score
        return reward, game_over, self.score


    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits the boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0:
            return True
        if pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        return False


    def _update_ui(self):
        self.display.fill(BLACK) # clear screen
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)) # snake body
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12)) # snake border
        if self.food is not None:
            pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE)) # food
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0]) # draw score
        pygame.display.flip() # update


    def _move(self, action):
        # [straight, rightTurn, leftTurn]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action, [1,0,0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0,1,0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn
        elif np.array_equal(action, [0,0,1]):
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn
        self.direction = new_dir
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        self.head = Point(x, y)