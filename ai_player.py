import numpy as np
from typing import Tuple, List, Optional, Dict
import heapq

class AI:
    def __init__(self, max_depth: int = 4):
        '''
        Initialize the AI player.

        Parameters:
            max_depth (int): Maximum depth for minimax evaluation (default: 4)

        Returns:
            None
        '''
        self.max_depth = max_depth
        self.last_position = None
        self.visited = set()
        self.current_path = None  # Store the current A* path

    def get_move(self, board: np.ndarray, positions: Dict[int, Tuple[int, int]], 
                horizontal_walls: np.ndarray, vertical_walls: np.ndarray) -> Optional[Tuple[int, int]]:
        '''
        Determine the next move using a combination of A* and minimax simultaneously.
        A* path is used to guide minimax evaluation and improve decision making.
        '''
        self.visited.clear()
        
        # Get A* path first
        path = self.A_star(board, positions[2], horizontal_walls, vertical_walls, positions)
        self.current_path = path
        
        # Get all possible moves
        possible_moves = self._get_all_possible_moves(board, positions, horizontal_walls, vertical_walls)
        best_move = None
        best_score = float('-inf')
        
        for move in possible_moves:
            # Create temporary board state
            temp_board = board.copy()
            temp_positions = positions.copy()
            
            # Simulate the move
            if isinstance(move, tuple) and len(move) == 2:  # Pawn move
                temp_positions[2] = move
                
            # Calculate combined score using both minimax and A* path information
            minimax_score = self.minimax(temp_board, temp_positions, horizontal_walls, 
                                    vertical_walls, self.max_depth, False)
            
            # Calculate A* path score
            astar_score = self._calculate_astar_move_score(move, path)
            
            # Combine scores with weights
            combined_score = (0.6 * minimax_score) + (0.4 * astar_score)
            
            if combined_score > best_score:
                best_score = combined_score
                best_move = move
        
        if best_move:
            self.last_position = positions[2]
        
        return best_move

    def _calculate_astar_move_score(self, move: Tuple[int, int], path: Optional[List[Tuple[int, int]]]) -> float:
        '''
        Calculate a score for a move based on how well it aligns with the A* path.
        
        Parameters:
            move (Tuple[int, int]): The move to evaluate
            path (Optional[List[Tuple[int, int]]]): The A* path if one exists
            
        Returns:
            float: Score for the move based on A* path alignment
        '''
        if not path or not move:
            return 0.0
        
        # If the move is the next step in the A* path
        if len(path) > 1 and move == path[1]:
            return 1.0
        
        # If the move is in the path but not the next step
        if move in path:
            return 0.7
        
        # Calculate distance to the path
        min_distance = float('inf')
        for path_pos in path:
            distance = abs(move[0] - path_pos[0]) + abs(move[1] - path_pos[1])
            min_distance = min(min_distance, distance)
        
        # Convert distance to a score (closer to path = higher score)
        if min_distance == float('inf'):
            return 0.0
        return max(0.0, 1.0 - (min_distance * 0.2))

    def _evaluate_position(self, board: np.ndarray, positions: Dict[int, Tuple[int, int]],
                        horizontal_walls: np.ndarray, vertical_walls: np.ndarray) -> float:
        '''
        Enhanced position evaluation incorporating both traditional metrics and A* path information.
        '''
        # Get current A* path for AI player
        ai_path = self.A_star(board, positions[2], horizontal_walls, vertical_walls, positions)
        opponent_path = self.A_star(board, positions[1], horizontal_walls, vertical_walls, positions)
        
        # Base position evaluation
        ai_distance_to_goal = positions[2][0]  # Distance from AI to its goal row
        opponent_distance_to_goal = board.shape[0] - 1 - positions[1][0]  # Distance from opponent to their goal
        
        # Path length evaluation
        ai_path_length = len(ai_path) if ai_path else board.shape[0] * 2
        opponent_path_length = len(opponent_path) if opponent_path else board.shape[0] * 2
        
        # Wall control evaluation
        wall_control = self._evaluate_wall_control(horizontal_walls, vertical_walls)
        
        # Combine all factors with weights
        score = (
            -1.5 * ai_path_length +  # Shorter AI path is better
            1.0 * opponent_path_length +  # Longer opponent path is better
            -2.0 * ai_distance_to_goal +  # Closer to goal is better
            1.0 * opponent_distance_to_goal +  # Opponent further from goal is better
            0.5 * wall_control  # Wall positioning importance
        )
        
        return score

    def _evaluate_wall_control(self, horizontal_walls: np.ndarray, vertical_walls: np.ndarray) -> float:
        '''
        Evaluate the strategic value of wall placements.
        '''
        score = 0.0
        
        # Evaluate horizontal walls
        for i in range(horizontal_walls.shape[0]):
            for j in range(horizontal_walls.shape[1]):
                if horizontal_walls[i, j] == 2:  # AI's wall
                    # Walls closer to the center are worth more
                    center_distance = abs(j - horizontal_walls.shape[1]//2)
                    score += 1.0 / (center_distance + 1)
        
        # Evaluate vertical walls
        for i in range(vertical_walls.shape[0]):
            for j in range(vertical_walls.shape[1]):
                if vertical_walls[i, j] == 2:  # AI's wall
                    # Walls closer to the center are worth more
                    center_distance = abs(j - vertical_walls.shape[1]//2)
                    score += 1.0 / (center_distance + 1)
        
        return score

    def get_current_path(self) -> Optional[List[Tuple[int, int]]]:
        '''
        Get the current A* path that the AI is considering.

        Returns:
            Optional[List[Tuple[int, int]]]: The current path or None if no path is being considered
        '''
        return self.current_path

    def A_star(self, board: np.ndarray, position: Tuple[int, int],
                          horizontal_walls: np.ndarray, vertical_walls: np.ndarray,
                          positions: Dict[int, Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
        '''
        Find the optimal path to the goal using A* algorithm.
        The goal is to reach any position in the last row of the board.

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
        goal_row = board.shape[0] - 1  # Last row is the goal
        
        # Initialize data structures
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}  # Initialize with start position
        cost_so_far = {start: 0}
        visited = set()  # Keep track of visited positions
        
        while frontier:
            current = heapq.heappop(frontier)[1]
            
            # Skip if already visited
            if current in visited:
                continue
                
            visited.add(current)
            
            # Check if goal is reached (any position in the last row)
            if current[0] == goal_row:
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path
            
            # Get valid moves and explore neighbors
            for next_move in self._get_valid_moves(board, current, horizontal_walls, vertical_walls, positions):
                next_pos = next_move[1]  # Get actual position from (score, pos) tuple
                
                # Skip if already visited
                if next_pos in visited:
                    continue
                
                # Calculate new cost (base cost + movement cost)
                # Movement cost is higher for horizontal moves (score from _get_valid_moves)
                new_cost = cost_so_far[current] + next_move[0]
                
                # Update if this path is better
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self._heuristic(next_pos, goal_row)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current
        
        return None  # No path found

    def _heuristic(self, pos: Tuple[int, int], goal_row: int) -> float:
        '''
        Calculate the heuristic value for A* pathfinding.
        The heuristic estimates the cost to reach any position in the goal row.

        Parameters:
            pos (Tuple[int, int]): Current position
            goal_row (int): The row number of the goal (last row)

        Returns:
            float: Heuristic value (weighted distance to goal row)
        '''
        # Distance to goal row (vertical distance)
        vertical_distance = goal_row - pos[0]
        
        # We don't need to consider horizontal distance since any position in the goal row is valid
        # But we still add a small penalty for horizontal movement to prefer more direct paths
        horizontal_penalty = abs(pos[1] - pos[1]) * 0.1  # Small penalty for horizontal movement
        
        return vertical_distance * 2 + horizontal_penalty

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
        
        moves = [(-1, 0), (0, 1), (0, -1), (1, 0)]  # Up, Right, Left, Down
        
        for di, dj in moves:
            ni, nj = i + di, j + dj
            if (0 <= ni < board.shape[0] and 
                0 <= nj < board.shape[1] and 
                board[ni, nj] == 0):
                if di == 0:  # Horizontal move
                    wall_pos = min(j, nj)
                    if vertical_walls[i, wall_pos] == 0:
                        score = ni + 5
                        valid_moves.append((score, (ni, nj)))
                else:  # Vertical move
                    wall_pos = min(i, ni)
                    if horizontal_walls[wall_pos, j] == 0:
                        score = ni if di == -1 else ni + 10
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
                jump_i, jump_j = 2 * opp_i - i, 2 * opp_j - j
                if (0 <= jump_i < board.shape[0] and 
                    0 <= jump_j < board.shape[1] and 
                    board[jump_i, jump_j] == 0):
                    if jump_i < i:  # Only upward jumps
                        valid_moves.append((jump_i, (jump_i, jump_j)))
        
        return valid_moves

    def _get_all_possible_moves(self, board: np.ndarray, positions: Dict[int, Tuple[int, int]],
                              horizontal_walls: np.ndarray, vertical_walls: np.ndarray) -> List[Tuple[int, int]]:
        '''
        Get all possible moves including pawn moves and wall placements.

        Parameters:
            board (np.ndarray): The game board
            positions (Dict[int, Tuple[int, int]]): Dictionary of player positions
            horizontal_walls (np.ndarray): Array of horizontal walls
            vertical_walls (np.ndarray): Array of vertical walls

        Returns:
            List[Tuple[int, int]]: List of all possible moves
        '''
        moves = []
        current_pos = positions[2]
        
        # Get pawn moves
        pawn_moves = self._get_valid_moves(board, current_pos, horizontal_walls, vertical_walls, positions)
        moves.extend([move[1] for move in pawn_moves])
        
        # Get wall moves
        if self._get_remaining_fences(horizontal_walls, vertical_walls) > 0:
            wall_moves = self._get_wall_moves(board, positions, horizontal_walls, vertical_walls)
            moves.extend(wall_moves)
        
        return moves

    def _get_wall_moves(self, board: np.ndarray, positions: Dict[int, Tuple[int, int]],
                       horizontal_walls: np.ndarray, vertical_walls: np.ndarray) -> List[Tuple[int, int]]:
        '''
        Get all possible wall placement moves.

        Parameters:
            board (np.ndarray): The game board
            positions (Dict[int, Tuple[int, int]]): Dictionary of player positions
            horizontal_walls (np.ndarray): Array of horizontal walls
            vertical_walls (np.ndarray): Array of vertical walls

        Returns:
            List[Tuple[int, int]]: List of possible wall positions
        '''
        wall_moves = []
        opponent_pos = positions[1]
        opp_i, opp_j = opponent_pos
        
        # Check horizontal walls
        for i in range(board.shape[0] - 1):
            for j in range(board.shape[1] - 1):
                if (horizontal_walls[i, j] == 0 and 
                    horizontal_walls[i, j + 1] == 0):
                    wall_moves.append((i, j))
        
        # Check vertical walls
        for i in range(board.shape[0] - 1):
            for j in range(board.shape[1] - 1):
                if (vertical_walls[i, j] == 0 and 
                    vertical_walls[i + 1, j] == 0):
                    wall_moves.append((i, j))
        
        return wall_moves

    def _get_remaining_fences(self, horizontal_walls: np.ndarray, vertical_walls: np.ndarray) -> int:
        '''
        Calculate the number of remaining walls for the AI player.

        Parameters:
            horizontal_walls (np.ndarray): Array of horizontal walls
            vertical_walls (np.ndarray): Array of vertical walls

        Returns:
            int: Number of remaining walls (out of 10)
        '''
        horizontal_count = np.sum(horizontal_walls == 2)
        vertical_count = np.sum(vertical_walls == 2)
        return 10 - (horizontal_count + vertical_count) // 2

    def _is_game_over(self, positions: Dict[int, Tuple[int, int]]) -> bool:
        '''
        Check if the game is over.

        Parameters:
            positions (Dict[int, Tuple[int, int]]): Dictionary of player positions

        Returns:
            bool: True if game is over, False otherwise
        '''
        # Game is over if any player reaches their goal row
        for player_id, pos in positions.items():
            if player_id == 1 and pos[0] == 0:  # Player 1 reaches top
                return True
            if player_id == 2 and pos[0] == 8:  # Player 2 reaches bottom
                return True
        return False

    def _evaluate_position(self, board: np.ndarray, positions: Dict[int, Tuple[int, int]],
                          horizontal_walls: np.ndarray, vertical_walls: np.ndarray) -> float:
        '''
        Evaluate the current position for minimax.

        Parameters:
            board (np.ndarray): The game board
            positions (Dict[int, Tuple[int, int]]): Dictionary of player positions
            horizontal_walls (np.ndarray): Array of horizontal walls
            vertical_walls (np.ndarray): Array of vertical walls

        Returns:
            float: Position evaluation score
        '''
        ai_pos = positions[2]
        opponent_pos = positions[1]
        
        # Distance to goal for AI (lower is better)
        ai_distance = board.shape[0] - 1 - ai_pos[0]
        
        # Distance to goal for opponent (higher is better)
        opponent_distance = opponent_pos[0]
        
        # Number of remaining walls
        remaining_walls = self._get_remaining_fences(horizontal_walls, vertical_walls)
        
        # Combine factors with weights
        return (opponent_distance - ai_distance) * 2 + remaining_walls

    def minimax(self, board: np.ndarray, positions: Dict[int, Tuple[int, int]],
                horizontal_walls: np.ndarray, vertical_walls: np.ndarray,
                depth: int, is_maximizing: bool) -> float:
        '''
        Minimax algorithm implementation.

        Parameters:
            board (np.ndarray): The game board
            positions (Dict[int, Tuple[int, int]]): Dictionary of player positions
            horizontal_walls (np.ndarray): Array of horizontal walls
            vertical_walls (np.ndarray): Array of vertical walls
            depth (int): Current depth in the minimax tree
            is_maximizing (bool): Whether the current player is maximizing

        Returns:
            float: Best evaluation score
        '''
        if depth == 0 or self._is_game_over(positions):
            return self._evaluate_position(board, positions, horizontal_walls, vertical_walls)
        
        if is_maximizing:
            max_eval = float('-inf')
            for move in self._get_all_possible_moves(board, positions, horizontal_walls, vertical_walls):
                # Simulate move
                eval = self.minimax(board, positions, horizontal_walls, vertical_walls, depth-1, False)
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = float('inf')
            for move in self._get_all_possible_moves(board, positions, horizontal_walls, vertical_walls):
                # Simulate move
                eval = self.minimax(board, positions, horizontal_walls, vertical_walls, depth-1, True)
                min_eval = min(min_eval, eval)
            return min_eval 