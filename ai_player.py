import numpy as np 
import heapq
from typing import Tuple, List, Dict, Optional, Union
import copy

class MinimaxAI:
    def __init__(self, max_depth=2, wall_radius=2):
        self.max_depth = max_depth
        self.wall_radius = wall_radius
        self.current_path = None
        self.timeout = 10  

    def get_move(
        self,
        board: np.ndarray,
        positions: Dict[int, Tuple[int, int]],
        horizontal_walls: np.ndarray,
        vertical_walls: np.ndarray
    ) -> Optional[Union[Tuple[int, int], Tuple[int, int, str]]]:
        """Determine the best move (either a pawn move or a wall placement) for the AI (player 2) using minimax with alpha-beta pruning.

        Parameters:
            board (np.ndarray) – A 2D numpy array representing the board (0: empty, 1: player 1, 2: player 2).
            positions (Dict[int, Tuple[int, int]]) – A dictionary mapping player numbers (1, 2) to their current (i,j) positions.
            horizontal_walls (np.ndarray) – A 2D numpy array (board_size-1, board_size) representing horizontal walls (0: none, 1: player 1, 2: player 2).
            vertical_walls (np.ndarray) – A 2D numpy array (board_size, board_size-1) representing vertical walls (0: none, 1: player 1, 2: player 2).

        Returns:
            Optional[Union[Tuple[int, int], Tuple[int, int, str]]] – The chosen move. If a pawn move, returns a tuple (i,j) (new cell coordinates); if a wall placement, returns a tuple (i,j,ori) (i,j: wall's top-left cell, ori: "horizontal" or "vertical"). Returns None if no move is found (e.g. timeout or all moves are losing).

        Notes:
            – Recalculates the A* path (self.current_path) for UI display.
            – Calls _minimax (with alpha-beta pruning) on a copy of the game state for each possible move.
            – Sorts possible moves (pawn moves first, then wall placements) to improve pruning.
            – Prints (via print) the number of possible moves, the chosen move (and its score), and the A* path.
        """
        

        import time
        start_time = time.time()
        
        # Recalcule le chemin A* pour l'affichage UI
        self.current_path = self.A_star(board, positions[2], horizontal_walls, vertical_walls, 2)
        
        # Set up minmax variable for the best move
        best_score = float('-inf')
        best_move = None
        alpha = float('-inf')
        beta = float('inf')
        
        # Retrieving all possible moves
        possible_moves = self._get_all_possible_moves(board, positions, horizontal_walls, vertical_walls)
        print("Coups possibles (IA) :", len(possible_moves))
        
        # Sort the moves so that pawn moves (i.e. tuples (i,j)) come first, and wall placements (tuples (i,j,ori)) come later.
        # (This is an optimization to improve alpha-beta pruning, as pawn moves are often "good" moves.)
        possible_moves.sort(key=lambda x: not (isinstance(x, tuple) and len(x) == 3))
        
        for move in possible_moves:
            # Check if the time is up
            if time.time() - start_time > self.timeout:
                print("Timeout reached, returning best move found so far")
                break
            # Create a copy of the game state
            new_board = board.copy()
            new_horizontal_walls = horizontal_walls.copy()
            new_vertical_walls = vertical_walls.copy()
            new_positions = positions.copy()
            new_remaining_fences = {1: 10 - np.count_nonzero(horizontal_walls) - np.count_nonzero(vertical_walls), 
                                   2: 10 - np.count_nonzero(horizontal_walls) - np.count_nonzero(vertical_walls)}
            
            # Apply the move to that copy
            if isinstance(move, tuple) and len(move) == 3:  # Wall placement
                i, j, ori = move
                if ori == 'horizontal':
                    new_horizontal_walls[i, j] = 2
                    new_horizontal_walls[i, j+1] = 2
                else:
                    new_vertical_walls[i, j] = 2
                    new_vertical_walls[i+1, j] = 2
                new_remaining_fences[2] -= 1
            else:  # Pawn move
                i, j = move
                ci, cj = new_positions[2]
                new_board[ci, cj] = 0
                new_board[i, j] = 2
                new_positions[2] = (i, j)
            
            # Call minimax recursively
            score = self._minimax(
                new_board, 
                new_positions, 
                new_horizontal_walls, 
                new_vertical_walls,
                new_remaining_fences,
                1,  # Passe au joueur adverse (1)
                self.max_depth - 1, 
                alpha, 
                beta, 
                False  # Min node (for the opponent)
            )
            
            if score > best_score:
                best_score = score
                best_move = move
        
        
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break  # Alpha-beta pruning
        # Add a check for the case where all moves are losing
        if best_move is None and possible_moves:
            # If all moves lead to a loss, take the first available move
            print("Tous les coups sont perdants, prend le premier disponible")
            best_move = possible_moves[0]

        print("Coup choisi (IA) :", best_move, "Score :", best_score)


        print("Chemin A* :", self.current_path)
        

        return best_move

    def _minimax(
        self, 
        board: np.ndarray,
        positions: Dict[int, Tuple[int, int]],
        horizontal_walls: np.ndarray,
        vertical_walls: np.ndarray,
        remaining_fences: Dict[int, int],
        player: int,  # 1 ou 2
        depth: int,
        alpha: float,
        beta: float,
        is_maximizing: bool
    ) -> float:
        """Recursive implementation of the minimax algorithm (with alpha-beta pruning) to evaluate a game state.

        Parameters:
            board (np.ndarray) – A 2D numpy array representing the board (0: empty, 1: player 1, 2: player 2).
            positions (Dict[int, Tuple[int, int]]) – A dictionary mapping player numbers (1, 2) to their current (i,j) positions.
            horizontal_walls (np.ndarray) – A 2D numpy array (board_size-1, board_size) representing horizontal walls (0: none, 1: player 1, 2: player 2).
            vertical_walls (np.ndarray) – A 2D numpy array (board_size, board_size-1) representing vertical walls (0: none, 1: player 1, 2: player 2).
            remaining_fences (Dict[int, int]) – A dictionary mapping player numbers (1, 2) to their remaining fence counts.
            player (int) – The player (1 or 2) whose turn it is (i.e. the player for whom we are evaluating the state).
            depth (int) – The remaining depth (or "ply") for minimax recursion. If 0, the state is evaluated using _evaluate.
            alpha (float) – The alpha value (lower bound) for alpha-beta pruning.
            beta (float) – The beta value (upper bound) for alpha-beta pruning.
            is_maximizing (bool) – A flag indicating whether the current node is a maximizing (True) or minimizing (False) node.

        Returns:
            float – The minimax score (evaluated via _evaluate if depth is 0, or recursively via minimax) for the given state. A higher score is favorable for player 2 (the AI).

        Notes:
            – If a player (1 or 2) has already won (i.e. reached his goal row), returns –inf (if player 1 won) or +inf (if player 2 won).
            – Possible moves (pawn moves and wall placements) are generated (via _get_all_possible_moves) and filtered (via _filter_wall_moves) before recursion.
            – For a maximizing node (is_maximizing=True), the maximum score among all child states is returned; for a minimizing node (is_maximizing=False), the minimum score is returned.
            – Alpha-beta pruning is applied (i.e. if beta <= alpha, recursion stops).
        """
        # Check if a player has won
        player1_won = positions[1][0] == 0
        player2_won = positions[2][0] == board.shape[0] - 1
        
        if player1_won:
            return float('-inf')  # The opponent has won, the worst score possible
        elif player2_won:
            return float('inf')   # The AI has won, the best score possible
        
        # If the max depth is reached, evaluate the position
        if depth == 0:
            return self._evaluate(board, positions, horizontal_walls, vertical_walls, remaining_fences)
        
        # Generate all possible moves
        possible_moves = self._get_all_possible_moves(board, positions, horizontal_walls, vertical_walls, player)
        print("Coups possibles (IA) dans minmax:", len(possible_moves))
        
        # Optimization: limit the number of walls considered
        possible_moves = self._filter_wall_moves(possible_moves, positions, player)
        print("Coups possibles (IA) dans minmax après filtrage:", len(possible_moves))
        
        # Sort the moves to improve alpha-beta pruning
        possible_moves.sort(key=lambda x: not (isinstance(x, tuple) and len(x) == 3))
        
        if is_maximizing:  # Max node (for the AI)
            max_eval = float('-inf')
            for move in possible_moves:
                # Create a copy of the game state
                new_board = board.copy()
                new_horizontal_walls = horizontal_walls.copy()
                new_vertical_walls = vertical_walls.copy()
                new_positions = positions.copy()
                new_remaining_fences = remaining_fences.copy()
                
                # Apply the move
                if isinstance(move, tuple) and len(move) == 3:  # Wall placement
                    i, j, ori = move
                    if ori == 'horizontal':
                        new_horizontal_walls[i, j] = player
                        new_horizontal_walls[i, j+1] = player
                    else:
                        new_vertical_walls[i, j] = player
                        new_vertical_walls[i+1, j] = player
                    new_remaining_fences[player] -= 1
                else:  # Pawn move
                    i, j = move
                    ci, cj = new_positions[player]
                    new_board[ci, cj] = 0
                    new_board[i, j] = player
                    new_positions[player] = (i, j)
                
                # Call minimax recursively
                next_player = 3 - player  # 1->2, 2->1
                eval = self._minimax(
                    new_board, 
                    new_positions, 
                    new_horizontal_walls, 
                    new_vertical_walls,
                    new_remaining_fences,
                    next_player,
                    depth - 1, 
                    alpha, 
                    beta, 
                    False
                )
                
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Alpha-beta pruning
            
            return max_eval
        else:  # Min node (for the opponent)
            min_eval = float('inf')
            for move in possible_moves:
                # Create a copy of the game state
                new_board = board.copy()
                new_horizontal_walls = horizontal_walls.copy()
                new_vertical_walls = vertical_walls.copy()
                new_positions = positions.copy()
                new_remaining_fences = remaining_fences.copy()
                
                # Apply the move
                if isinstance(move, tuple) and len(move) == 3:  # Wall placement
                    i, j, ori = move
                    if ori == 'horizontal':
                        new_horizontal_walls[i, j] = player
                        new_horizontal_walls[i, j+1] = player
                    else:
                        new_vertical_walls[i, j] = player
                        new_vertical_walls[i+1, j] = player
                    new_remaining_fences[player] -= 1
                else:  # Pawn move
                    i, j = move
                    ci, cj = new_positions[player]
                    new_board[ci, cj] = 0
                    new_board[i, j] = player
                    new_positions[player] = (i, j)
                
                # Appelle minimax récursivement
                next_player = 3 - player  # 1->2, 2->1
                eval = self._minimax(
                    new_board, 
                    new_positions, 
                    new_horizontal_walls, 
                    new_vertical_walls,
                    new_remaining_fences,
                    next_player,
                    depth - 1, 
                    alpha, 
                    beta, 
                    True
                )
                
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Élagage alpha-beta
            
            return min_eval

    def _filter_wall_moves(self, moves, positions, player):
        """Filter the list of possible moves (pawn moves and wall placements) so that only wall placements near the players (within a radius) are retained.

        Parameters:
            moves (List) – A list of possible moves (each move is either a tuple (i,j) (pawn move) or a tuple (i,j,ori) (wall placement)).
            positions (Dict[int, Tuple[int, int]]) – A dictionary mapping player numbers (1, 2) to their current (i,j) positions.
            player (int) – The player (1 or 2) for whom the moves are being filtered.

        Returns:
            List – A filtered list of moves. Pawn moves (i.e. tuples (i,j)) are always retained; wall placements (tuples (i,j,ori)) are retained only if the wall's top-left cell (i,j) is within a radius (self.wall_radius) of any player's position.

        Notes:
            – This function is an optimization to reduce the number of wall placements evaluated (and thus speed up minimax).
        """
        filtered_moves = []
        
        for move in moves:
            # Keep all moves that are not wall placements
            if not isinstance(move, tuple) or len(move) != 3:
                filtered_moves.append(move)
                continue
            
            # Pour les placements de murs, ne garde que ceux proches des joueurs
            i, j, _ = move
            for p in [1, 2]:
                pi, pj = positions[p]
                if abs(i - pi) <= self.wall_radius and abs(j - pj) <= self.wall_radius:
                    filtered_moves.append(move)
                    break
        
        return filtered_moves

    def _evaluate(
        self, 
        board: np.ndarray,
        positions: Dict[int, Tuple[int, int]],
        horizontal_walls: np.ndarray,
        vertical_walls: np.ndarray,
        remaining_fences: Dict[int, int]
    ) -> float:
        """Evaluate a game state (a leaf node in minimax) by computing a score (higher is better for player 2, the AI).

        Parameters:
            board (np.ndarray) – A 2D numpy array representing the board (0: empty, 1: player 1, 2: player 2).
            positions (Dict[int, Tuple[int, int]]) – A dictionary mapping player numbers (1, 2) to their current (i,j) positions.
            horizontal_walls (np.ndarray) – A 2D numpy array (board_size-1, board_size) representing horizontal walls (0: none, 1: player 1, 2: player 2).
            vertical_walls (np.ndarray) – A 2D numpy array (board_size, board_size-1) representing vertical walls (0: none, 1: player 1, 2: player 2).
            remaining_fences (Dict[int, int]) – A dictionary mapping player numbers (1, 2) to their remaining fence counts.

        Returns:
            float – A score (higher is better for player 2) computed as follows:
                   • If player 1 (resp. player 2) is blocked (i.e. A* returns None), –inf (resp. +inf) is returned.
                   • Otherwise, the score is computed as (remaining_fences[2] – remaining_fences[1]) * fence_weight – (length of A* path for player 2) * 3 + (length of A* path for player 1).

        Notes:
            – fence_weight (a constant, e.g. 0.5) is used to weigh the fence advantage.
        """
        # Calculate the shortest paths
        ai_path = self.A_star(board, positions[2], horizontal_walls, vertical_walls, 2)
        op_path = self.A_star(board, positions[1], horizontal_walls, vertical_walls, 1)
        
        # If a player is blocked, return an extreme value
        if ai_path is None:
            return float('-inf')  # The AI is blocked
        if op_path is None:
            return float('inf')   # The opponent is blocked
        
        # Take into account the number of remaining fences
        fence_advantage = remaining_fences[2] - remaining_fences[1]
        
        # Coefficient for the walls (the walls are important but less than the distance)
        fence_weight = 0.5
        
        # Final formula: diff_paths + fence_advantage * fence_weight
        return fence_advantage * fence_weight - len(ai_path)*3 + len(op_path)

    def A_star(
        self, 
        board: np.ndarray, 
        start: Tuple[int, int],
        horizontal_walls: np.ndarray, 
        vertical_walls: np.ndarray, 
        player: int
    ) -> Optional[List[Tuple[int, int]]]:
        """Compute the shortest path (using A*) from a given start position (for a player) to his goal row (top row for player 1, bottom row for player 2).

        Parameters:
            board (np.ndarray) – A 2D numpy array representing the board (0: empty, 1: player 1, 2: player 2).
            start (Tuple[int, int]) – The (i,j) coordinates of the starting cell.
            horizontal_walls (np.ndarray) – A 2D numpy array (board_size-1, board_size) representing horizontal walls (0: none, 1: player 1, 2: player 2).
            vertical_walls (np.ndarray) – A 2D numpy array (board_size, board_size-1) representing vertical walls (0: none, 1: player 1, 2: player 2).
            player (int) – The player (1 or 2) for whom the path is computed (player 1's goal is row 0, player 2's goal is row board.shape[0] – 1).

        Returns:
            Optional[List[Tuple[int, int]]] – A list of (i,j) tuples (the cells visited along the shortest path) if a path exists; None otherwise.

        Notes:
            – Uses a priority queue (heapq) (with a heuristic (Manhattan distance) + cost so far) to explore cells.
            – Calls _get_valid_moves (which checks for walls) to determine valid neighbors.
        """
        goal_row = 0 if player == 1 else board.shape[0] - 1
        frontier = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}
        visited = set()

        while frontier:
            _, current = heapq.heappop(frontier)
            if current in visited:
                continue
            visited.add(current)

            if current[0] == goal_row:
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                return list(reversed(path))

            for move_cost, neighbor in self._get_valid_moves(board, current, horizontal_walls, vertical_walls):
                if neighbor in visited:
                    continue
                new_cost = cost_so_far[current] + move_cost
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self._heuristic(neighbor, goal_row)
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current
        return None

    def _heuristic(self, pos: Tuple[int, int], goal_row: int) -> float:
        """Compute the heuristic (Manhattan distance) from a given cell (i,j) to a goal row.

        Parameters:
            pos (Tuple[int, int]) – The (i,j) coordinates of the cell.
            goal_row (int) – The target row (e.g. 0 for player 1, board.shape[0] – 1 for player 2).

        Returns:
            float – The Manhattan distance (i.e. |i – goal_row|) from pos to the goal row.
        """
        return abs(goal_row - pos[0])

    def _get_valid_moves(
        self,
        board: np.ndarray,
        position: Tuple[int, int],
        horizontal_walls: np.ndarray,
        vertical_walls: np.ndarray
    ) -> List[Tuple[float, Tuple[int, int]]]:
        """Compute a list of valid moves (with a cost) from a given cell (i,j) on the board.
        Includes both simple moves and jumps over opponents.

        Parameters:
            board (np.ndarray) – A 2D numpy array representing the board (0: empty, 1: player 1, 2: player 2).
            position (Tuple[int, int]) – The (i,j) coordinates of the cell.
            horizontal_walls (np.ndarray) – A 2D numpy array (board_size-1, board_size) representing horizontal walls (0: none, 1: player 1, 2: player 2).
            vertical_walls (np.ndarray) – A 2D numpy array (board_size, board_size-1) representing vertical walls (0: none, 1: player 1, 2: player 2).

        Returns:
            List[Tuple[float, Tuple[int, int]]] – A list of (cost, (ni,nj)) tuples for valid moves.
            Includes both simple moves (cost=1) and jumps over opponents (cost=1).
        """
        i, j = position
        moves: List[Tuple[float, Tuple[int,int]]] = []
        
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:  # Top, Bottom, Left, Right
            ni, nj = i+di, j+dj
            
            # Check if the next cell is within bounds
            if not (0 <= ni < board.shape[0] and 0 <= nj < board.shape[1]):
                continue
            
            # Check if there's a wall blocking the move
            if di != 0:  # Vertical move
                w = min(i, ni)
                if horizontal_walls[w, j] != 0:
                    continue
            else:  # Horizontal move
                w = min(j, nj)
                if vertical_walls[i, w] != 0:
                    continue
            
            # If the next cell is empty, it's a valid simple move
            if board[ni, nj] == 0:
                moves.append((1, (ni, nj)))
                continue
            
            # If the next cell is occupied, try to jump over
            ni2, nj2 = ni + di, nj + dj
            if (0 <= ni2 < board.shape[0] and 0 <= nj2 < board.shape[1] and 
                board[ni2, nj2] == 0):
                # Check if there's a wall blocking the jump
                if di != 0:  # Vertical move
                    w = min(ni, ni2)
                    if horizontal_walls[w, nj] != 0:
                        continue
                else:  # Horizontal move
                    w = min(nj, nj2)
                    if vertical_walls[ni, w] != 0:
                        continue
                moves.append((1, (ni2, nj2)))
        
        return moves

    def _get_all_possible_moves(
        self,
        board: np.ndarray,
        positions: Dict[int, Tuple[int, int]],
        horizontal_walls: np.ndarray,
        vertical_walls: np.ndarray,
        player: int = 2  # Player for which to calculate the moves (default: the AI)
    ) -> List[Union[Tuple[int,int], Tuple[int,int,str]]]:
        """Compute a list of all possible moves (pawn moves and wall placements) for a given player (default: player 2, the AI).

        Parameters:
            board (np.ndarray) – A 2D numpy array representing the board (0: empty, 1: player 1, 2: player 2).
            positions (Dict[int, Tuple[int, int]]) – A dictionary mapping player numbers (1, 2) to their current (i,j) positions.
            horizontal_walls (np.ndarray) – A 2D numpy array (board_size-1, board_size) representing horizontal walls (0: none, 1: player 1, 2: player 2).
            vertical_walls (np.ndarray) – A 2D numpy array (board_size, board_size-1) representing vertical walls (0: none, 1: player 1, 2: player 2).
            player (int, optional) – The player (1 or 2) for whom the moves are computed. Defaults to 2 (the AI).

        Returns:
            List[Union[Tuple[int,int], Tuple[int,int,str]]] – A list of moves. Each move is either a tuple (i,j) (a pawn move to cell (i,j)) or a tuple (i,j,ori) (a wall placement (i,j,ori) (ori is "horizontal" or "vertical")).

        Notes:
            – For pawn moves, iterates over adjacent cells (and checks for walls) and, if a cell is occupied, checks if a jump (over an opponent) is possible.
            – For wall placements, iterates over (i,j) (i < board.shape[0] – 1, j < board.shape[1] – 1) and, if the wall (horizontal or vertical) is not already placed, calls _check_paths_exist (to ensure that both players still have a path) and appends the move (i,j,ori) (ori is "horizontal" or "vertical").
        """
        actions: List[Union[Tuple[int,int], Tuple[int,int,str]]] = []
        
        # Pawn moves
        current_pos = positions[player]
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:  # Top, Bottom, Left, Right
            ni, nj = current_pos[0] + di, current_pos[1] + dj
            if not (0 <= ni < board.shape[0] and 0 <= nj < board.shape[1]):
                continue
            
            # Check if there is no wall between the current position and the new position
            can_move = True
            if di != 0:  # Vertical move
                w = min(current_pos[0], ni)
                if horizontal_walls[w, current_pos[1]] != 0:
                    can_move = False
            else:  # Horizontal move
                w = min(current_pos[1], nj)
                if vertical_walls[current_pos[0], w] != 0:
                    can_move = False
            
            if not can_move: 
                continue
                
            # Check if the cell is empty
            if board[ni, nj] == 0:
                actions.append((ni, nj))
            else:  # Try to jump over an opponent if possible
                # Case occupied, try to jump over
                ni2, nj2 = ni + di, nj + dj
                if (0 <= ni2 < board.shape[0] and 0 <= nj2 < board.shape[1] and 
                    board[ni2, nj2] == 0):
                    # Check if there is no wall between the pawn and the arrival cell
                    can_jump = True
                    if di != 0:  # Vertical move
                        w = min(ni, ni2)
                        if horizontal_walls[w, nj] != 0:
                            can_jump = False
                    else:  # Horizontal move
                        w = min(nj, nj2)
                        if vertical_walls[ni, w] != 0:
                            can_jump = False
                    
                    if can_jump:
                        actions.append((ni2, nj2))
        
        # Wall placements
        # Count the number of already placed walls
        total_walls = np.count_nonzero(horizontal_walls) + np.count_nonzero(vertical_walls)
        remaining = 20 - total_walls
        player_remaining = remaining // 2  # Approximate
        
        if player_remaining > 0:
            for i in range(board.shape[0] - 1):
                for j in range(board.shape[1] - 1):
                    # Horizontal wall
                    if horizontal_walls[i, j] == 0 and horizontal_walls[i, j+1] == 0:
                        # Check if the placement is valid (does not block all paths)
                        temp_h_walls = horizontal_walls.copy()
                        temp_h_walls[i, j] = player
                        temp_h_walls[i, j+1] = player
                        
                        if (self._check_paths_exist(board, positions, temp_h_walls, vertical_walls)):
                            actions.append((i, j, 'horizontal'))
                    
                    # Vertical wall
                    if vertical_walls[i, j] == 0 and vertical_walls[i+1, j] == 0:
                        # Check if the placement is valid (does not block all paths)
                        temp_v_walls = vertical_walls.copy()
                        temp_v_walls[i, j] = player
                        temp_v_walls[i+1, j] = player
                        
                        if (self._check_paths_exist(board, positions, horizontal_walls, temp_v_walls)):
                            actions.append((i, j, 'vertical'))
        
        return actions

    def _check_paths_exist(self, board, positions, h_walls, v_walls):
        """Check (using A*) that both players (1 and 2) still have a valid path (i.e. a path exists) from their current positions to their respective goal rows.

        Parameters:
            board (np.ndarray) – A 2D numpy array representing the board (0: empty, 1: player 1, 2: player 2).
            positions (Dict[int, Tuple[int, int]]) – A dictionary mapping player numbers (1, 2) to their current (i,j) positions.
            h_walls (np.ndarray) – A 2D numpy array (board_size-1, board_size) representing horizontal walls (0: none, 1: player 1, 2: player 2).
            v_walls (np.ndarray) – A 2D numpy array (board_size, board_size-1) representing vertical walls (0: none, 1: player 1, 2: player 2).

        Returns:
            bool – True if (and only if) A* (called for player 1 and player 2) returns a non-None path (i.e. a valid path exists) for both players; False otherwise.

        Notes:
            – This function is used (e.g. in _get_all_possible_moves) to validate wall placements (so that a wall does not block all paths for a player).
        """
        # Check if player 1 has a path to the top row
        player1_path = self.A_star(board, positions[1], h_walls, v_walls, 1)
        if player1_path is None:
            return False
            
        # Check if player 2 has a path to the bottom row
        player2_path = self.A_star(board, positions[2], h_walls, v_walls, 2)
        if player2_path is None:
            return False
            
        return True

    def get_current_path(self):
        """Return the current A* path (self.current_path) computed (and stored) for the AI (player 2) (used for UI display).

        Returns:
            Optional[List[Tuple[int, int]]] – A list of (i,j) tuples (the cells visited along the shortest path) if a path exists; None otherwise.
        """
        return self.current_path