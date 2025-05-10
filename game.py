import pygame
import numpy as np
from typing import Optional, Tuple, List

class Game:
    def __init__(self):
        self.board_size = 9  # 9x9 board
        self.cell_size = 60
        self.margin = 50
        self.window_size = self.board_size * self.cell_size + 2 * self.margin
        
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Quoridor")
        
        # Game state
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)  # 0: empty, 1: player1, 2: player2
        self.horizontal_walls = np.zeros((self.board_size-1, self.board_size), dtype=int)  # 0: no wall, 1: player1 wall, 2: player2 wall
        self.vertical_walls = np.zeros((self.board_size, self.board_size-1), dtype=int)  # 0: no wall, 1: player1 wall, 2: player2 wall
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.remaining_fences = {1: 10, 2: 10}  # Each player starts with 10 fences
        self.placing_wall = False
        self.wall_orientation = 'horizontal'  # or 'vertical'
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GRAY = (128, 128, 128)
        self.LIGHT_BLUE = (200, 200, 255)
        self.LIGHT_RED = (255, 200, 200)
        self.WALL_COLOR = (139, 69, 19)  # Brown
        
        # Initialize player positions
        self.positions = {
            1: (self.board_size-1, self.board_size//2),  # Player 1 starts at bottom
            2: (0, self.board_size//2)  # Player 2 starts at top
        }
        self.board[self.positions[1]] = 1
        self.board[self.positions[2]] = 2

    def draw_board(self):
        self.screen.fill(self.WHITE)
        
        # Draw goal lines
        for j in range(self.board_size):
            pygame.draw.rect(self.screen, self.LIGHT_RED,
                           (self.margin + j * self.cell_size, self.margin,
                            self.cell_size, self.cell_size))
            pygame.draw.rect(self.screen, self.LIGHT_BLUE,
                           (self.margin + j * self.cell_size, self.margin + (self.board_size-1) * self.cell_size,
                            self.cell_size, self.cell_size))
        
        # Draw grid
        for i in range(self.board_size):
            for j in range(self.board_size):
                rect = pygame.Rect(
                    self.margin + j * self.cell_size,
                    self.margin + i * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(self.screen, self.BLACK, rect, 1)
                
                if self.board[i, j] == 1:
                    pygame.draw.circle(
                        self.screen,
                        self.RED,
                        (self.margin + j * self.cell_size + self.cell_size // 2,
                         self.margin + i * self.cell_size + self.cell_size // 2),
                        self.cell_size // 3
                    )
                elif self.board[i, j] == 2:
                    pygame.draw.circle(
                        self.screen,
                        self.BLUE,
                        (self.margin + j * self.cell_size + self.cell_size // 2,
                         self.margin + i * self.cell_size + self.cell_size // 2),
                        self.cell_size // 3
                    )
        
        # Draw walls
        for i in range(self.board_size-1):
            for j in range(self.board_size):
                if self.horizontal_walls[i, j]:
                    pygame.draw.rect(self.screen, self.WALL_COLOR,
                                   (self.margin + j * self.cell_size,
                                    self.margin + (i + 1) * self.cell_size - 5,
                                    self.cell_size * 2, 10))
        
        for i in range(self.board_size):
            for j in range(self.board_size-1):
                if self.vertical_walls[i, j]:
                    pygame.draw.rect(self.screen, self.WALL_COLOR,
                                   (self.margin + (j + 1) * self.cell_size - 5,
                                    self.margin + i * self.cell_size,
                                    10, self.cell_size * 2))
        
        # Draw remaining fences
        font = pygame.font.Font(None, 36)
        text1 = font.render(f"Fences: {self.remaining_fences[1]}", True, self.RED)
        text2 = font.render(f"Fences: {self.remaining_fences[2]}", True, self.BLUE)
        self.screen.blit(text1, (10, 10))
        self.screen.blit(text2, (self.window_size - 150, 10))
        
        # Draw current player indicator
        current_text = font.render(f"Player {self.current_player}'s turn", True, self.BLACK)
        self.screen.blit(current_text, (self.window_size//2 - 100, 10))

    def is_valid_move(self, pos: Tuple[int, int]) -> bool:
        i, j = pos
        if not (0 <= i < self.board_size and 0 <= j < self.board_size):
            return False
        if self.board[i, j] != 0:
            return False
            
        current_i, current_j = self.positions[self.current_player]
        
        # Check if move is adjacent and not blocked by a wall
        if abs(i - current_i) + abs(j - current_j) == 1:
            # Check for wall between current position and target
            if i == current_i:  # Horizontal move
                wall_pos = min(j, current_j)
                if self.vertical_walls[i, wall_pos]:
                    return False
            else:  # Vertical move
                wall_pos = min(i, current_i)
                if self.horizontal_walls[wall_pos, j]:
                    return False
            return True
            
        # Check for jump over opponent
        opponent = 3 - self.current_player
        opp_i, opp_j = self.positions[opponent]
        if (i, j) == (2 * opp_i - current_i, 2 * opp_j - current_j):
            # Check if opponent is adjacent
            if abs(opp_i - current_i) + abs(opp_j - current_j) == 1:
                # Check if there's a wall behind opponent
                if (0 <= 2 * opp_i - current_i < self.board_size and 
                    0 <= 2 * opp_j - current_j < self.board_size):
                    return True
                # Check for side jumps
                if abs(i - current_i) == 1:  # Vertical jump
                    return (self.vertical_walls[min(i, current_i), j] == 0 and
                            self.vertical_walls[min(i, current_i), j-1] == 0)
                else:  # Horizontal jump
                    return (self.horizontal_walls[i, min(j, current_j)] == 0 and
                            self.horizontal_walls[i-1, min(j, current_j)] == 0)
        return False

    def is_valid_wall_placement(self, pos: Tuple[int, int], orientation: str) -> bool:
        i, j = pos
        if orientation == 'horizontal':
            if not (0 <= i < self.board_size-1 and 0 <= j < self.board_size-1):
                return False
            if self.horizontal_walls[i, j] or self.horizontal_walls[i, j+1]:
                return False
        else:  # vertical
            if not (0 <= i < self.board_size-1 and 0 <= j < self.board_size-1):
                return False
            if self.vertical_walls[i, j] or self.vertical_walls[i+1, j]:
                return False
        
        # Check if wall placement would block all paths to goal
        # TODO: Implement path checking
        return True

    def make_move(self, pos: Tuple[int, int]) -> bool:
        if not self.is_valid_move(pos):
            return False
            
        i, j = pos
        current_i, current_j = self.positions[self.current_player]
        self.board[current_i, current_j] = 0
        self.board[i, j] = self.current_player
        self.positions[self.current_player] = (i, j)
        
        # Check for win
        if (self.current_player == 1 and i == 0) or (self.current_player == 2 and i == self.board_size-1):
            self.game_over = True
            self.winner = self.current_player
        
        self.current_player = 3 - self.current_player
        return True

    def place_wall(self, pos: Tuple[int, int], orientation: str) -> bool:
        if self.remaining_fences[self.current_player] <= 0:
            return False
            
        if not self.is_valid_wall_placement(pos, orientation):
            return False
            
        i, j = pos
        if orientation == 'horizontal':
            self.horizontal_walls[i, j] = self.current_player
            self.horizontal_walls[i, j+1] = self.current_player
        else:
            self.vertical_walls[i, j] = self.current_player
            self.vertical_walls[i+1, j] = self.current_player
            
        self.remaining_fences[self.current_player] -= 1
        self.current_player = 3 - self.current_player
        return True

    def run_pvp(self):
        while not self.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                    
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return  # Return to menu
                    elif event.key == pygame.K_SPACE:
                        self.placing_wall = not self.placing_wall
                        if self.placing_wall:
                            self.wall_orientation = 'horizontal'
                    elif event.key == pygame.K_r and self.placing_wall:
                        self.wall_orientation = 'vertical' if self.wall_orientation == 'horizontal' else 'horizontal'
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    i = (y - self.margin) // self.cell_size
                    j = (x - self.margin) // self.cell_size
                    
                    if 0 <= i < self.board_size and 0 <= j < self.board_size:
                        if self.placing_wall:
                            if self.place_wall((i, j), self.wall_orientation):
                                self.placing_wall = False
                        else:
                            self.make_move((i, j))
            
            self.draw_board()
            pygame.display.flip()
            
        # Game over screen
        font = pygame.font.Font(None, 74)
        text = font.render(f"Player {self.winner} wins!", True, self.BLACK)
        text_rect = text.get_rect(center=(self.window_size//2, self.window_size//2))
        self.screen.blit(text, text_rect)
        pygame.display.flip()
        pygame.time.wait(2000)

    def run_pve(self, ai_player):
        while not self.game_over:
            if self.current_player == 1:  # Human player
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                        
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            return  # Return to menu
                        elif event.key == pygame.K_SPACE:
                            self.placing_wall = not self.placing_wall
                            if self.placing_wall:
                                self.wall_orientation = 'horizontal'
                        elif event.key == pygame.K_r and self.placing_wall:
                            self.wall_orientation = 'vertical' if self.wall_orientation == 'horizontal' else 'horizontal'
                    
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        x, y = event.pos
                        i = (y - self.margin) // self.cell_size
                        j = (x - self.margin) // self.cell_size
                        
                        if 0 <= i < self.board_size and 0 <= j < self.board_size:
                            if self.placing_wall:
                                if self.place_wall((i, j), self.wall_orientation):
                                    self.placing_wall = False
                            else:
                                if self.make_move((i, j)):
                                    pygame.time.wait(500)  # Small delay before AI move
            else:  # AI player
                ai_move = ai_player.get_move(self.board, self.positions, self.horizontal_walls, self.vertical_walls)
                if ai_move:
                    if isinstance(ai_move, tuple) and len(ai_move) == 3:  # Wall placement
                        self.place_wall((ai_move[0], ai_move[1]), ai_move[2])
                    else:  # Pawn move
                        self.make_move(ai_move)
                    pygame.time.wait(500)  # Small delay after AI move
            
            self.draw_board()
            pygame.display.flip()
            
        # Game over screen
        font = pygame.font.Font(None, 74)
        if self.winner == 1:
            text = font.render("You win!", True, self.BLACK)
        else:
            text = font.render("AI wins!", True, self.BLACK)
        text_rect = text.get_rect(center=(self.window_size//2, self.window_size//2))
        self.screen.blit(text, text_rect)
        pygame.display.flip()
        pygame.time.wait(2000) 