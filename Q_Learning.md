# Q-Learning in RL_snake_game (Simplified)

This file explains how the Q-learning code here works.

---

## What's going on

The snake plays the game, and the code teaches it how to get better using Q-learning.

- The **game** gives us info about what is happening (the state).
- The **agent** (the code) picks what to do next (left, straight, right).
- The game tells us how good or bad that move was (reward).

We simply keep doing this over and over until the agent gets good.

---

## What is Q(s, a)?

Think of Q(s, a) as a guess: "If I do action a in state s, how good will it be in the long run?"  
We don’t know the right answers, so we start with random guesses and fix them every time the snake plays.

We use this formula to update the guess:

```
new_value = reward + gamma * max(Q(next_state, all_actions))
```

- `gamma` is a number between 0 and 1 that decides how much we care about the future.
- `max(...)` means we look at the best action in the next state.

---

## Rewards in this game

The code usually gives:
- +10 for eating food
- 0 for just moving
- -10 for dying

These numbers can be changed if you want the snake to play differently. Going to run tests to see what is best for the agent.

---

## Epsilon-greedy

This is how we decide whether to try something random or use what we already know.

- With chance `epsilon`, do something random.
- Otherwise, pick the best move based on Q-values.

At first epsilon is big (lots of random moves), and it gets smaller over time.

---

## Replay memory

We store past game steps in a list (state, action, reward, next_state, done).  
When training, we grab a bunch of random steps from this list. This helps the training be more stable.

---

## The network (model.py)

- Inputs: stuff about the game state (danger left, right, straight, food location, etc.).
- Hidden layers: a couple of layers to process the inputs.
- Outputs: 3 values, one for each possible move.

---

## Training loop (agent.py)

1. Get the starting state from the game.
2. Pick an action (random or best from model).
3. Do the action in the game.
4. Get reward and next state.
5. Save it in memory.
6. If we have enough saved steps, train the model:
   - Predict Q-values for the states.
   - Get the Q-value for the action we took.
   - Calculate the "target" value using the reward and next state.
   - Update the model so the predicted Q gets closer to the target.

Repeat until the snake dies, then start a new game.

---

## Hyperparameters (tweak these to change how it learns)

- `gamma`: how much we care about future rewards.
- `epsilon`: how often we make random moves.
- `lr`: learning rate (how big each training step is).
- `batch_size`: how many samples to train on at once.

---

## TL;DR

- Snake plays, it remembers what happened.
- Guess how good each move is.
- Slowly fix the guesses after each game step.
- Over time, snake learns what moves are better.

Thats about it!
