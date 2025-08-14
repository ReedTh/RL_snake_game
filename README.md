# Snake Game AI with Q-Learning

## Features

- Classic Snake game built using **Pygame**  
- AI agent trained with **Q-learning** and a **neural network**  
- Includes training scripts and **visualizations** of the learning process  

---

## How It Works

### Game Environment  
The Snake game is implemented using **Pygame**, which serves as the environment for the agent to interact with.

### AI Agent  
An agent observes the state of the game and learns to take actions that maximize its score over time.

### Neural Network Model  
- **Input Layer**: Encodes the current state of the game  
- **Hidden Layer**: Extracts features and patterns  
- **Output Layer**: Predicts Q-values for each possible action (left, right, straight)

### Training Algorithm  
- The agent is trained using the **Q-learning** algorithm  
- Starts with random actions to explore the environment  
- Gradually learns from experience using rewards to guide future actions  
- The neural network updates its weights to improve Q-value predictions  

---

## Training Process

1. Agent interacts with the game environment  
2. Takes actions and receives rewards or penalties  
3. Uses **Q-learning** to update its policy based on outcomes  
4. Over time, learns to make decisions that maximize game state rewards  

---

## Results

- The AI agent successfully learns to play Snake and achieves high scores  
- Training performance and score progression are visualized using **matplotlib**  

---

## Requirements

- Pygame  
- NumPy  
- Matplotlib  
- PyTorch or TensorFlow (depending on your neural network framework)

---

## Run It Yourself 

```bash
# clone the repository
git clone https://github.com/ReedTh/RL_snake_game.git

# get in the right directory
cd RL_snake_game

# install requirements
pip install -r requirements.txt

# run the training
python agent.py
