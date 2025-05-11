import pygame
import numpy as np
from typing import Optional, Tuple, List

class Game:
    def __init__(self):
        self.board_size = 9  # 9x9 board
        self.cell_size = 60
        self.wall_size = 20  # Size of wall placement buttons
        self.margin = 50
        self.window_size = self.board_size * self.cell_size + (self.board_size - 1) * self.wall_size + 2 * self.margin
        
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
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GRAY = (128, 128, 128)
        self.LIGHT_BLUE = (200, 200, 255)
        self.LIGHT_RED = (255, 200, 200)
        self.WALL_COLOR = (139, 69, 19)  # Brown
        self.WALL_BUTTON_COLOR = (200, 200, 200)  # Light gray for wall buttons
        self.WALL_BUTTON_HOVER = (180, 180, 180)  # Darker gray for hover
        
        # Initialize player positions
        self.positions = {
            1: (self.board_size-1, self.board_size//2),  # Player 1 starts at bottom
            2: (0, self.board_size//2)  # Player 2 starts at top
        }
        self.board[self.positions[1]] = 1
        self.board[self.positions[2]] = 2

    def get_cell_rect(self, i: int, j: int) -> pygame.Rect:
        x = self.margin + j * (self.cell_size + self.wall_size)
        y = self.margin + i * (self.cell_size + self.wall_size)
        return pygame.Rect(x, y, self.cell_size, self.cell_size)

    def get_horizontal_wall_rect(self, i: int, j: int) -> pygame.Rect:
        x = self.margin + j * (self.cell_size + self.wall_size)
        y = self.margin + (i + 1) * self.cell_size + i * self.wall_size
        return pygame.Rect(x, y, self.cell_size * 2 + self.wall_size, self.wall_size)

    def get_vertical_wall_rect(self, i: int, j: int) -> pygame.Rect:
        x = self.margin + (j + 1) * self.cell_size + j * self.wall_size
        y = self.margin + i * (self.cell_size + self.wall_size)
        return pygame.Rect(x, y, self.wall_size, self.cell_size * 2 + self.wall_size)

    def draw_board(self):
        self.screen.fill(self.WHITE)
        
        # Draw goal lines
        for j in range(self.board_size):
            rect = self.get_cell_rect(0, j)
            pygame.draw.rect(self.screen, self.LIGHT_RED, rect)
            rect = self.get_cell_rect(self.board_size-1, j)
            pygame.draw.rect(self.screen, self.LIGHT_BLUE, rect)
        
        # Draw grid and wall buttons
        for i in range(self.board_size):
            for j in range(self.board_size):
                # Draw cell
                cell_rect = self.get_cell_rect(i, j)
                pygame.draw.rect(self.screen, self.BLACK, cell_rect, 1)
                
                # Draw pawn
                if self.board[i, j] == 1:
                    pygame.draw.circle(
                        self.screen,
                        self.RED,
                        (cell_rect.centerx, cell_rect.centery),
                        self.cell_size // 3
                    )
                elif self.board[i, j] == 2:
                    pygame.draw.circle(
                        self.screen,
                        self.BLUE,
                        (cell_rect.centerx, cell_rect.centery),
                        self.cell_size // 3
                    )
                
                # Draw wall buttons
                if i < self.board_size - 1:  # Horizontal wall buttons
                    wall_rect = self.get_horizontal_wall_rect(i, j)
                    color = self.WALL_BUTTON_HOVER if wall_rect.collidepoint(pygame.mouse.get_pos()) else self.WALL_BUTTON_COLOR
                    pygame.draw.rect(self.screen, color, wall_rect)
                    if self.horizontal_walls[i, j]:
                        pygame.draw.rect(self.screen, self.WALL_COLOR, wall_rect)
                
                if j < self.board_size - 1:  # Vertical wall buttons
                    wall_rect = self.get_vertical_wall_rect(i, j)
                    color = self.WALL_BUTTON_HOVER if wall_rect.collidepoint(pygame.mouse.get_pos()) else self.WALL_BUTTON_COLOR
                    pygame.draw.rect(self.screen, color, wall_rect)
                    if self.vertical_walls[i, j]:
                        pygame.draw.rect(self.screen, self.WALL_COLOR, wall_rect)
        
        # Draw remaining fences
        font = pygame.font.Font(None, 36)
        text1 = font.render(f"Fences: {self.remaining_fences[1]}", True, self.RED)
        text2 = font.render(f"Fences: {self.remaining_fences[2]}", True, self.BLUE)
        self.screen.blit(text1, (10, 10))
        self.screen.blit(text2, (self.window_size - 150, 10))
        
        # Draw current player indicator
        current_text = font.render(f"Player {self.current_player}'s turn", True, self.BLACK)
        self.screen.blit(current_text, (self.window_size//2 - 100, 10))

    def get_cell_from_pos(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        x, y = pos
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.get_cell_rect(i, j).collidepoint(x, y):
                    return (i, j)
        return None

    def get_wall_from_pos(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int, str]]:
        x, y = pos
        # Check horizontal walls
        for i in range(self.board_size - 1):
            for j in range(self.board_size):
                if self.get_horizontal_wall_rect(i, j).collidepoint(x, y):
                    return (i, j, 'horizontal')
        
        # Check vertical walls
        for i in range(self.board_size):
            for j in range(self.board_size - 1):
                if self.get_vertical_wall_rect(i, j).collidepoint(x, y):
                    return (i, j, 'vertical')
        return None

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

    def draw_menu(self):
        self.screen.fill(self.WHITE)
        font = pygame.font.Font(None, 74)
        title = font.render("Quoridor", True, self.BLACK)
        title_rect = title.get_rect(center=(self.window_size//2, self.window_size//3))
        
        font = pygame.font.Font(None, 48)
        pvp_text = font.render("Player vs Player", True, self.BLACK)
        pve_text = font.render("Player vs AI", True, self.BLACK)
        
        pvp_rect = pvp_text.get_rect(center=(self.window_size//2, self.window_size//2))
        pve_rect = pve_text.get_rect(center=(self.window_size//2, self.window_size//2 + 80))
        
        # Draw buttons
        pygame.draw.rect(self.screen, self.LIGHT_BLUE, pvp_rect.inflate(40, 20))
        pygame.draw.rect(self.screen, self.LIGHT_RED, pve_rect.inflate(40, 20))
        
        # Draw text
        self.screen.blit(title, title_rect)
        self.screen.blit(pvp_text, pvp_rect)
        self.screen.blit(pve_text, pve_rect)
        
        pygame.display.flip()
        
        return pvp_rect, pve_rect

    def run_menu(self):
        while True:
            pvp_rect, pve_rect = self.draw_menu()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                    
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if pvp_rect.collidepoint(event.pos):
                        return "pvp"
                    elif pve_rect.collidepoint(event.pos):
                        return "pve"
                        
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return None

    def run_pvp(self):
        while not self.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                    
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return  # Return to menu
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Check for wall placement
                    wall_pos = self.get_wall_from_pos(event.pos)
                    if wall_pos:
                        i, j, orientation = wall_pos
                        if self.place_wall((i, j), orientation):
                            continue
                    
                    # Check for pawn move
                    cell_pos = self.get_cell_from_pos(event.pos)
                    if cell_pos:
                        self.make_move(cell_pos)
            
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
                    
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        # Check for wall placement
                        wall_pos = self.get_wall_from_pos(event.pos)
                        if wall_pos:
                            i, j, orientation = wall_pos
                            if self.place_wall((i, j), orientation):
                                pygame.time.wait(500)  # Small delay before AI move
                                continue
                        
                        # Check for pawn move
                        cell_pos = self.get_cell_from_pos(event.pos)
                        if cell_pos:
                            if self.make_move(cell_pos):
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

if __name__ == "__main__":
    pygame.init()
    game = Game()
    
    while True:
        mode = game.run_menu()
        if mode is None:
            break
            
        if mode == "pvp":
            game.run_pvp()
        else:  # pve
            # TODO: Implement AI player
            print("AI mode not implemented yet")
            continue
            
        # Reset game state for new game
        game = Game()
    
    pygame.quit() 