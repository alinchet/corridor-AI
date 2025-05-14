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
        self.PATH_COLOR = (255, 200, 200)  # Light red for AI path
        
        # Menu button
        self.menu_button = pygame.Rect(10, self.window_size - 40, 100, 30)
        
        # Initialize player positions
        self.positions = {
            1: (self.board_size-1, self.board_size//2),  # Player 1 starts at bottom
            2: (0, self.board_size//2)  # Player 2 starts at top
        }
        self.board[self.positions[1]] = 1
        self.board[self.positions[2]] = 2
        
        # AI path
        self.ai_path = None

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

    def draw_ai_path(self):
        '''
        Draw the AI's current path in light red.
        '''
        if not self.ai_path:
            return
            
        for pos in self.ai_path:
            cell_rect = self.get_cell_rect(pos[0], pos[1])
            pygame.draw.rect(self.screen, self.PATH_COLOR, cell_rect)
            pygame.draw.rect(self.screen, self.BLACK, cell_rect, 1)  # Draw border

    def draw_board(self):
        self.screen.fill(self.WHITE)
        
        # Draw goal lines
        for j in range(self.board_size):
            rect = self.get_cell_rect(0, j)
            pygame.draw.rect(self.screen, self.LIGHT_RED, rect)
            rect = self.get_cell_rect(self.board_size-1, j)
            pygame.draw.rect(self.screen, self.LIGHT_BLUE, rect)
        
        # Draw AI path first (so it appears behind pawns)
        self.draw_ai_path()
        
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
        
        # Draw menu button
        pygame.draw.rect(self.screen, self.GRAY, self.menu_button)
        menu_text = font.render("Menu", True, self.BLACK)
        text_rect = menu_text.get_rect(center=self.menu_button.center)
        self.screen.blit(menu_text, text_rect)

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
        opponent = 3 - self.current_player
        opp_i, opp_j = self.positions[opponent]

        # 1. Saut direct par-dessus l'adversaire
        if (i, j) == (2 * opp_i - current_i, 2 * opp_j - current_j):
            # L'adversaire doit être adjacent
            if abs(opp_i - current_i) + abs(opp_j - current_j) == 1:
                # Vérification du mur entre l'adversaire et la destination du saut
                if abs(i - opp_i) == 1:  # Saut vertical
                    if self.horizontal_walls[opp_i, j] != 0 or self.horizontal_walls[min(current_i, opp_i), j] != 0:
                        return False
                else:  # Saut horizontal
                    if self.vertical_walls[i, opp_j] != 0 or self.vertical_walls[i, min(current_j, opp_j)] != 0:
                        return False
                return True

        # 2. Mouvement adjacent (normal)
        if abs(i - current_i) + abs(j - current_j) == 1:
            # Si la case voisine est occupée par l'adversaire, il faut vérifier s'il y a un mur entre eux
            if (i, j) == (opp_i, opp_j):
                # Mouvement vers l'adversaire interdit s'il y a un mur entre les deux
                if abs(i - current_i) == 1:  # Mouvement vertical
                    if self.horizontal_walls[min(i, current_i), j] != 0:
                        return False
                else:  # Mouvement horizontal
                    if self.vertical_walls[i, min(j, current_j)] != 0:
                        return False
                # Si pas de mur, le saut direct sera traité au prochain tour
                return False  # On ne peut pas juste aller sur la case de l'adversaire
            # Mouvement normal (pas sur l'adversaire)
            if i == current_i:  # Mouvement horizontal
                wall_pos = min(j, current_j)
                if self.vertical_walls[i, wall_pos] != 0:
                    return False
            else:  # Mouvement vertical
                wall_pos = min(i, current_i)
                if self.horizontal_walls[wall_pos, j] != 0:
                    return False
            return True

        # 3. Side jumps (saut latéral si saut direct bloqué)
        if abs(current_i - opp_i) + abs(current_j - opp_j) == 1:
            # Si le saut direct est bloqué par un mur, on autorise le side jump si pas de mur latéral
            if (abs(i - opp_i) == 1 and abs(j - opp_j) == 1 and (i == current_i or j == current_j)):
                # Saut latéral vertical
                if i == opp_i and abs(j - opp_j) == 1:
                    if self.vertical_walls[i, min(j, opp_j)] == 0:
                        return True
                # Saut latéral horizontal
                if j == opp_j and abs(i - opp_i) == 1:
                    if self.horizontal_walls[min(i, opp_i), j] == 0:
                        return True

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
        
        # Temporarily place the wall
        if orientation == 'horizontal':
            self.horizontal_walls[i, j] = self.current_player
            self.horizontal_walls[i, j+1] = self.current_player
        else:
            self.vertical_walls[i, j] = self.current_player
            self.vertical_walls[i+1, j] = self.current_player
        
        # Check if both players still have a path to their goal
        player1_has_path = self._has_path_to_goal(1, self.positions[1], 0)  # Player 1's goal is top row
        player2_has_path = self._has_path_to_goal(2, self.positions[2], self.board_size-1)  # Player 2's goal is bottom row
        
        # Remove the temporary wall
        if orientation == 'horizontal':
            self.horizontal_walls[i, j] = 0
            self.horizontal_walls[i, j+1] = 0
        else:
            self.vertical_walls[i, j] = 0
            self.vertical_walls[i+1, j] = 0
        
        return player1_has_path and player2_has_path

    def _has_path_to_goal(self, player: int, start_pos: Tuple[int, int], goal_row: int) -> bool:
        '''
        Check if there is a path from the current position to the goal row using BFS.

                Parameters:
                        player (int): Player number (1 or 2)
                        start_pos (Tuple[int, int]): Starting position
                        goal_row (int): Goal row (0 for player 1, board_size-1 for player 2)

                Returns:
                        bool: True if a path exists, False otherwise
        '''
        visited = set()
        queue = [start_pos]
        visited.add(start_pos)
        
        while queue:
            current = queue.pop(0)
            i, j = current
            
            if i == goal_row:  # Reached goal row
                return True
            
            # Check all possible moves
            moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
            for di, dj in moves:
                ni, nj = i + di, j + dj
                if (0 <= ni < self.board_size and 
                    0 <= nj < self.board_size and 
                    (ni, nj) not in visited):
                    
                    # Check if move is valid (no wall blocking)
                    if di == 0:  # Horizontal move
                        wall_pos = min(j, nj)
                        if self.vertical_walls[i, wall_pos] == 0:
                            queue.append((ni, nj))
                            visited.add((ni, nj))
                    else:  # Vertical move
                        wall_pos = min(i, ni)
                        if self.horizontal_walls[wall_pos, j] == 0:
                            queue.append((ni, nj))
                            visited.add((ni, nj))
        
        return False

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

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False  # Return to menu
                
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Check menu button
                if self.menu_button.collidepoint(event.pos):
                    return False  # Return to menu
                
                # Check for wall placement
                wall_pos = self.get_wall_from_pos(event.pos)
                if wall_pos:
                    i, j, orientation = wall_pos
                    if self.place_wall((i, j), orientation):
                        return True
                
                # Check for pawn move
                cell_pos = self.get_cell_from_pos(event.pos)
                if cell_pos:
                    if self.make_move(cell_pos):
                        return True
        
        return True

    def run_pvp(self):
        while not self.game_over:
            if not self.handle_events():
                return
            
            self.draw_board()
            pygame.display.flip()
            
        # Game over screen
        font = pygame.font.Font(None, 74)
        text = font.render(f"Player {self.winner} wins!", True, self.BLACK)
        text_rect = text.get_rect(center=(self.window_size//2, self.window_size//2))
        self.screen.blit(text, text_rect)
        pygame.display.flip()
        
        # Wait for click or ESC
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return
                if event.type == pygame.MOUSEBUTTONDOWN:
                    return

    def run_pve(self, ai_player):
        while not self.game_over:
            if self.current_player == 1:  # Human player
                if not self.handle_events():
                    return
            else:  # AI player
                # Get AI's current path before making a move
                self.ai_path = ai_player.get_current_path()
                
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
        
        # Wait for click or ESC
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return
                if event.type == pygame.MOUSEBUTTONDOWN:
                    return 