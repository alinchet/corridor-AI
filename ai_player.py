import numpy as np
from typing import Tuple, List, Optional, Dict
import heapq



class SimpleAI:
    def __init__(self, max_depth: int = 3):
        '''
        Initialize the SimpleAI player.

                Parameters:
                        max_depth (int): Maximum depth for move evaluation (default: 3)

                Returns:
                        None
        '''
        self.max_depth = max_depth
        self.last_position = None

    def get_move(self, board: np.ndarray, positions: Dict[int, Tuple[int, int]], 
                horizontal_walls: np.ndarray, vertical_walls: np.ndarray) -> Optional[Tuple[int, int]]:
        '''
        Determine the next move for the AI player.

                Parameters:
                        board (np.ndarray): The game board
                        positions (Dict[int, Tuple[int, int]]): Dictionary of player positions. e.g. {2: (0, 4)} is the position of player 2 (AI) at the start of the game.
                        horizontal_walls (np.ndarray): 2D array representing horizontal walls. 
                            - Shape: (board_size-1, board_size)
                            - Values: 0 (no wall), 1 (player 1's wall), 2 (player 2's wall)
                            - Example: horizontal_walls[i,j] represents the wall between rows i and i+1 at column j
                        vertical_walls (np.ndarray): 2D array representing vertical walls.
                            - Shape: (board_size, board_size-1)
                            - Values: 0 (no wall), 1 (player 1's wall), 2 (player 2's wall)
                            - Example: vertical_walls[i,j] represents the wall between columns j and j+1 at row i

                Returns:
                        Optional[Tuple[int, int]]: The chosen move position or None if no valid move
        '''
        
        current_pos = positions[2]
        
        # Decide whether to move pawn or place wall
        if np.random.random() < 0.3 and self._get_remaining_fences(horizontal_walls, vertical_walls) > 0:  # 30% chance to place wall
            wall_move = self._get_wall_move(board, positions, horizontal_walls, vertical_walls)
            if wall_move:
                return wall_move
        
        # Get all valid moves
        valid_moves = self._get_valid_moves(board, current_pos, horizontal_walls, vertical_walls, positions)
        if not valid_moves:
            return None
            
        # Filter out moves that would take us back to the previous position
        if self.last_position:
            valid_moves = [move for move in valid_moves if move[1] != self.last_position]
        
        # Choose move that gets closer to goal (bottom row for AI)
        best_move = None
        best_score = float('inf')
        for move in valid_moves:
            score = move[0]  # Distance to goal
            if score < best_score:
                best_score = score
                best_move = move[1]
        
        if best_move:
            self.last_position = current_pos
        return best_move

    def _get_remaining_fences(self, horizontal_walls: np.ndarray, vertical_walls: np.ndarray) -> int:
        '''
        Calculate the number of remaining walls for the AI player.

                Parameters:
                        horizontal_walls (np.ndarray): Array of horizontal walls
                        vertical_walls (np.ndarray): Array of vertical walls

                Returns:
                        int: Number of remaining walls (out of 10)
        '''
        # Count walls placed by AI (player 2)
        horizontal_count = np.sum(horizontal_walls == 2)
        vertical_count = np.sum(vertical_walls == 2)
        return 10 - (horizontal_count + vertical_count) // 2  # Divide by 2 because each wall takes 2 spaces

    def _get_valid_moves(self, board: np.ndarray, position: Tuple[int, int],
                        horizontal_walls: np.ndarray, vertical_walls: np.ndarray,
                        positions: Dict[int, Tuple[int, int]]) -> List[Tuple[float, Tuple[int, int]]]:
        '''
        Get all valid moves from the current position.

                Parameters:
                        board (np.ndarray): The game board
                        position (Tuple[int, int]): Current position
                        horizontal_walls (np.ndarray): Array of horizontal walls
                        vertical_walls (np.ndarray): Array of vertical walls
                        positions (Dict[int, Tuple[int, int]]): Dictionary of player positions

                Returns:
                        List[Tuple[float, Tuple[int, int]]]: List of valid moves with their scores
        '''
        i, j = position
        valid_moves = []
        
        # Prioritize upward movement
        moves = [(-1, 0), (0, 1), (0, -1), (1, 0)]  # Up, Right, Left, Down
        
        # Check adjacent moves
        for di, dj in moves:
            ni, nj = i + di, j + dj
            if (0 <= ni < board.shape[0] and 
                0 <= nj < board.shape[1] and 
                board[ni, nj] == 0):
                # Check for wall
                if di == 0:  # Horizontal move
                    wall_pos = min(j, nj)
                    if vertical_walls[i, wall_pos] == 0:
                        # Penalize horizontal moves heavily
                        score = ni + 5  # Large penalty for horizontal movement
                        valid_moves.append((score, (ni, nj)))
                else:  # Vertical move
                    wall_pos = min(i, ni)
                    if horizontal_walls[wall_pos, j] == 0:
                        # Prioritize upward movement
                        score = ni if di == -1 else ni + 10  # Penalize downward movement heavily
                        valid_moves.append((score, (ni, nj)))
        
        # Check jump moves
        opponent_pos = None
        for pos in positions.values():
            if pos != position:
                opponent_pos = pos
                break
        
        if opponent_pos:
            opp_i, opp_j = opponent_pos
            if abs(opp_i - i) + abs(opp_j - j) == 1:
                # Check for jump over opponent
                jump_i, jump_j = 2 * opp_i - i, 2 * opp_j - j
                if (0 <= jump_i < board.shape[0] and 
                    0 <= jump_j < board.shape[1] and 
                    board[jump_i, jump_j] == 0):
                    # Only add jump if it's upward
                    if jump_i < i:  # Only upward jumps
                        valid_moves.append((jump_i, (jump_i, jump_j)))
        
        return valid_moves

    def _get_wall_move(self, board: np.ndarray, positions: Dict[int, Tuple[int, int]],
                      horizontal_walls: np.ndarray, vertical_walls: np.ndarray) -> Optional[Tuple[int, int, str]]:
        '''
        Determine the best wall placement move.

                Parameters:
                        board (np.ndarray): The game board
                        positions (Dict[int, Tuple[int, int]]): Dictionary of player positions
                        horizontal_walls (np.ndarray): Array of horizontal walls
                        vertical_walls (np.ndarray): Array of vertical walls

                Returns:
                        Optional[Tuple[int, int, str]]: Wall placement coordinates and orientation, or None
        '''
        # Try to place wall in front of opponent
        opponent_pos = positions[1]
        opp_i, opp_j = opponent_pos
        
        # Try horizontal wall
        if opp_i > 0:
            if (horizontal_walls[opp_i-1, opp_j] == 0 and 
                horizontal_walls[opp_i-1, opp_j+1] == 0):
                return (opp_i-1, opp_j, 'horizontal')
        
        # Try vertical wall
        if opp_j > 0:
            if (vertical_walls[opp_i, opp_j-1] == 0 and 
                vertical_walls[opp_i+1, opp_j-1] == 0):
                return (opp_i, opp_j-1, 'vertical')
        
        return None

class SuperAI(SimpleAI):
    def __init__(self, max_depth: int = 4):
        '''
        Initialize the SuperAI player.

                Parameters:
                        max_depth (int): Maximum depth for move evaluation (default: 4)

                Returns:
                        None
        '''
        super().__init__(max_depth)
        self.visited = set()

    def get_move(self, board: np.ndarray, positions: Dict[int, Tuple[int, int]], 
                horizontal_walls: np.ndarray, vertical_walls: np.ndarray) -> Optional[Tuple[int, int]]:
        '''
        Determine the next move for the SuperAI player using A* pathfinding.

                Parameters:
                        board (np.ndarray): The game board
                        positions (Dict[int, Tuple[int, int]]): Dictionary of player positions
                        horizontal_walls (np.ndarray): Array of horizontal walls
                        vertical_walls (np.ndarray): Array of vertical walls

                Returns:
                        Optional[Tuple[int, int]]: The chosen move position or None if no valid move
        '''
        self.visited.clear()
        
        # Use A* to find best path to goal
        path = self._find_path_to_goal(board, positions[2], horizontal_walls, vertical_walls, positions)
        if path and len(path) > 1:
            next_move = path[1]
            self.last_position = positions[2]
            return next_move
        
        # If no path found, try to place wall
        if self._get_remaining_fences(horizontal_walls, vertical_walls) > 0:
            wall_move = self._get_wall_move(board, positions, horizontal_walls, vertical_walls)
            if wall_move:
                return wall_move
        
        # Fallback to simple move
        return super().get_move(board, positions, horizontal_walls, vertical_walls)

    def _find_path_to_goal(self, board: np.ndarray, position: Tuple[int, int],
                          horizontal_walls: np.ndarray, vertical_walls: np.ndarray,
                          positions: Dict[int, Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
        '''
        Find the optimal path to the goal using A* algorithm.

                Parameters:
                        board (np.ndarray): The game board
                        position (Tuple[int, int]): Current position
                        horizontal_walls (np.ndarray): Array of horizontal walls
                        vertical_walls (np.ndarray): Array of vertical walls
                        positions (Dict[int, Tuple[int, int]]): Dictionary of player positions

                Returns:
                        Optional[List[Tuple[int, int]]]: List of positions forming the path to goal, or None
        '''
        start = position
        goal = (board.shape[0] - 1, position[1])  # Goal is bottom row
        
        frontier = [] # Priority queue for A* algorithm
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            current = heapq.heappop(frontier)[1]
            
            if current[0] == board.shape[0] - 1:  # Reached goal (bottom row)
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path
            
            for next_pos in self._get_valid_moves(board, current, horizontal_walls, vertical_walls, positions):
                next_pos = next_pos[1]  # Get actual position from (score, pos) tuple
                new_cost = cost_so_far[current] + 1
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self._heuristic(next_pos, goal)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current
        
        return None

    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        '''
        Calculate the heuristic value for A* pathfinding.

                Parameters:
                        pos (Tuple[int, int]): Current position
                        goal (Tuple[int, int]): Goal position

                Returns:
                        float: Heuristic value (weighted Manhattan distance)
        '''
        # Manhattan distance to goal with higher weight on vertical distance

        return (goal[0] - pos[0]) * 2 + abs(pos[1] - goal[1])

    def _get_wall_move(self, board: np.ndarray, positions: Dict[int, Tuple[int, int]],
                      horizontal_walls: np.ndarray, vertical_walls: np.ndarray) -> Optional[Tuple[int, int, str]]:
        '''
        Determine the best wall placement move using opponent path prediction.

                Parameters:
                        board (np.ndarray): The game board
                        positions (Dict[int, Tuple[int, int]]): Dictionary of player positions
                        horizontal_walls (np.ndarray): Array of horizontal walls
                        vertical_walls (np.ndarray): Array of vertical walls

                Returns:
                        Optional[Tuple[int, int, str]]: Wall placement coordinates and orientation, or None
        '''
        # Find opponent's path to goal
        opponent_pos = positions[1]
        opponent_path = self._find_path_to_goal(board, opponent_pos, horizontal_walls, vertical_walls, positions)
        
        if opponent_path and len(opponent_path) > 1:
            next_move = opponent_path[1]
            # Try to block the path
            if next_move[0] == opponent_pos[0]:  # Horizontal move
                wall_pos = min(next_move[1], opponent_pos[1])
                if (vertical_walls[opponent_pos[0], wall_pos] == 0 and
                    vertical_walls[opponent_pos[0]+1, wall_pos] == 0):
                    return (opponent_pos[0], wall_pos, 'vertical')
            else:  # Vertical move
                wall_pos = min(next_move[0], opponent_pos[0])
                if (horizontal_walls[wall_pos, opponent_pos[1]] == 0 and
                    horizontal_walls[wall_pos, opponent_pos[1]+1] == 0):
                    return (wall_pos, opponent_pos[1], 'horizontal')
        
        # If no strategic wall placement found, try to place wall in front of opponent
        opp_i, opp_j = opponent_pos
        if opp_i > 0:
            if (horizontal_walls[opp_i-1, opp_j] == 0 and 
                horizontal_walls[opp_i-1, opp_j+1] == 0):
                return (opp_i-1, opp_j, 'horizontal')
        
        if opp_j > 0:
            if (vertical_walls[opp_i, opp_j-1] == 0 and 
                vertical_walls[opp_i+1, opp_j-1] == 0):
                return (opp_i, opp_j-1, 'vertical')
        
        return None 

    def minimax(self, board, positions, horizontal_walls, vertical_walls, depth, is_maximizing):
        if depth == 0 or self._is_game_over(positions):
            return self._evaluate_position(board, positions, horizontal_walls, vertical_walls)
        
        if is_maximizing:
            max_eval = float('-inf')
            for move in self._get_all_possible_moves(board, positions, horizontal_walls, vertical_walls):
                # Simuler le coup
                eval = self.minimax(board, positions, horizontal_walls, vertical_walls, depth-1, False)
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = float('inf')
            for move in self._get_all_possible_moves(board, positions, horizontal_walls, vertical_walls):
                # Simuler le coup
                eval = self.minimax(board, positions, horizontal_walls, vertical_walls, depth-1, True)
                min_eval = min(min_eval, eval)
            return min_eval 