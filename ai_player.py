import numpy as np
from typing import Tuple, List, Optional, Dict
import heapq

class SimpleAI:
    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth

    def get_move(self, board: np.ndarray, positions: Dict[int, Tuple[int, int]], 
                horizontal_walls: np.ndarray, vertical_walls: np.ndarray) -> Optional[Tuple[int, int]]:
        # Decide whether to move pawn or place wall
        if np.random.random() < 0.3:  # 30% chance to place wall
            wall_move = self._get_wall_move(board, positions, horizontal_walls, vertical_walls)
            if wall_move:
                return wall_move
        
        # Get all valid moves
        valid_moves = self._get_valid_moves(board, positions[2], horizontal_walls, vertical_walls)
        if not valid_moves:
            return None
            
        # Choose move that gets closer to goal
        best_move = None
        best_score = float('inf')
        for move in valid_moves:
            score = move[0]  # Distance to goal
            if score < best_score:
                best_score = score
                best_move = move[1]
        
        return best_move

    def _get_valid_moves(self, board: np.ndarray, position: Tuple[int, int],
                        horizontal_walls: np.ndarray, vertical_walls: np.ndarray) -> List[Tuple[float, Tuple[int, int]]]:
        i, j = position
        valid_moves = []
        
        # Check adjacent moves
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ni, nj = i + di, j + dj
            if (0 <= ni < board.shape[0] and 
                0 <= nj < board.shape[1] and 
                board[ni, nj] == 0):
                # Check for wall
                if di == 0:  # Horizontal move
                    wall_pos = min(j, nj)
                    if vertical_walls[i, wall_pos] == 0:
                        valid_moves.append((ni, (ni, nj)))
                else:  # Vertical move
                    wall_pos = min(i, ni)
                    if horizontal_walls[wall_pos, j] == 0:
                        valid_moves.append((ni, (ni, nj)))
        
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
                    valid_moves.append((jump_i, (jump_i, jump_j)))
        
        return [(abs(move[0]), move[1]) for move in valid_moves]

    def _get_wall_move(self, board: np.ndarray, positions: Dict[int, Tuple[int, int]],
                      horizontal_walls: np.ndarray, vertical_walls: np.ndarray) -> Optional[Tuple[int, int, str]]:
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
        super().__init__(max_depth)
        self.visited = set()

    def get_move(self, board: np.ndarray, positions: Dict[int, Tuple[int, int]], 
                horizontal_walls: np.ndarray, vertical_walls: np.ndarray) -> Optional[Tuple[int, int]]:
        self.visited.clear()
        
        # Use A* to find best path to goal
        path = self._find_path_to_goal(board, positions[2], horizontal_walls, vertical_walls)
        if path and len(path) > 1:
            return path[1]  # Return next move in path
        
        # If no path found, try to place wall
        wall_move = self._get_wall_move(board, positions, horizontal_walls, vertical_walls)
        if wall_move:
            return wall_move
        
        # Fallback to simple move
        return super().get_move(board, positions, horizontal_walls, vertical_walls)

    def _find_path_to_goal(self, board: np.ndarray, position: Tuple[int, int],
                          horizontal_walls: np.ndarray, vertical_walls: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        start = position
        goal = (0, position[1])  # Goal is top row
        
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            current = heapq.heappop(frontier)[1]
            
            if current[0] == 0:  # Reached goal
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path
            
            for next_pos in self._get_valid_moves(board, current, horizontal_walls, vertical_walls):
                next_pos = next_pos[1]  # Get actual position from (score, pos) tuple
                new_cost = cost_so_far[current] + 1
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self._heuristic(next_pos, goal)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current
        
        return None

    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        # Manhattan distance to goal
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def _get_wall_move(self, board: np.ndarray, positions: Dict[int, Tuple[int, int]],
                      horizontal_walls: np.ndarray, vertical_walls: np.ndarray) -> Optional[Tuple[int, int, str]]:
        # Find opponent's path to goal
        opponent_pos = positions[1]
        opponent_path = self._find_path_to_goal(board, opponent_pos, horizontal_walls, vertical_walls)
        
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
        
        return super()._get_wall_move(board, positions, horizontal_walls, vertical_walls) 