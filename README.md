# Corridor Game

A turn-based corridor game with a graphical interface, featuring both player vs player and player vs AI modes.

## Features

- Player vs Player mode
- Player vs Simple AI mode (using Minimax algorithm)
- Player vs Super AI mode (using Minimax with A* pathfinding)
- Graphical interface using Pygame
- Intuitive controls

## Requirements

- Python 3.7+
- Pygame
- NumPy

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## How to Play

1. Run the game:
```bash
python main.py
```

2. Choose a game mode from the menu:
   - Play vs Player: Two players take turns on the same computer
   - Play vs Simple AI: Play against an AI using the Minimax algorithm
   - Play vs Super AI: Play against an advanced AI using Minimax with A* pathfinding

3. Game Rules:
   - Players take turns moving their piece
   - Each move must be to an adjacent empty cell
   - The game ends when a player has no valid moves
   - The last player to make a valid move wins

## Controls

- Use the mouse to click on the desired cell to make a move
- Close the window to exit the game
