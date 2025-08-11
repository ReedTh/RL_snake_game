import torch
import random
import numpy as np
from main import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from plot_result import plot
from collections import deque

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self):
        self.numOfGames = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # just a big list for memory
        self.model = Linear_QNet(11, 256, 3) # our brain
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma) # trains the brain


    def get_state(self, game):
        head = game.snake[0] # where's the head
        point_l = Point(head.x - 20, head.y) # left point
        point_r = Point(head.x + 20, head.y) # right point
        point_d = Point(head.x, head.y - 20) # down point
        point_u = Point(head.x, head.y + 20) # up point

        dir_l = game.direction == Direction.LEFT # going left?
        dir_r = game.direction == Direction.RIGHT # going right?
        dir_u = game.direction == Direction.UP # going up?
        dir_d = game.direction == Direction.DOWN # going down?

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_r and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_u and game.is_collision(point_d)),

            # Move direction
            dir_r,
            dir_d,
            dir_l,
            dir_u,

            # Food location
            game.food.x < game.head.x, # food left
            game.food.x > game.head.x, # food right
            game.food.y < game.head.y, # food down
            game.food.y > game.head.y # food up
        ]

        return np.array(state, dtype=int) # turn it into an array
    
    def remember(self, state, action, reward, next_state, gameOver):
        self.memory.append((state, action, reward, next_state, gameOver)) # just add to memory

    def train_long_memory(self):
        # train on a bunch of past moves
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, gameOvers = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, gameOvers)

    def train_short_memory(self, state, action, reward, next_state, gameOver):
        # train on just the last move
        self.trainer.train_step(state, action, reward, next_state, gameOver)

    def get_action(self, state):
        # random moves: tradeoff exploration / explotation 
        self.epsilon = 90 - self.numOfGames # less random as we play more
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2) # just do something
            final_move[int(move)] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0) # ask the brain
            move = torch.argmax(prediction).item()
            final_move[int(move)] = 1
        return final_move
    

def train(max_games=0):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        old_state = agent.get_state(game)

        # get move
        action = agent.get_action(old_state)

        # perform move and get new state
        reward, gameOver, score = game.play_step(action)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(old_state, action, reward, state_new, gameOver)

        # remember
        agent.remember(old_state, action, reward, state_new, gameOver)

        if gameOver:
            # train long memory and plot result
            game.reset()
            agent.numOfGames += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save() # new high score, save the brain

            print('Game', agent.numOfGames, 'Score', score, 'Current Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.numOfGames
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            # stop if max_games is set and reached
            if max_games > 0 and agent.numOfGames >= max_games:
                # save the final plot as an image
                from plot_result import plt
                plot(plot_scores, plot_mean_scores, save_path="results.png")
                print(f"Training finished. Plot saved as results.png")
                break

if __name__ == "__main__":
    # set max_games 
    train(max_games=500)
