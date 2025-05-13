import random
import numpy as np
import heapq
from typing import Tuple, List, Optional, Dict, Union
import copy

class AI:
    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth
        self.current_path: Optional[List[Tuple[int, int]]] = None

    def get_move(
        self,
        board: np.ndarray,
        positions: Dict[int, Tuple[int, int]],
        horizontal_walls: np.ndarray,
        vertical_walls: np.ndarray
    ) -> Optional[Union[Tuple[int,int], Tuple[int,int,str]]]:
        # Recompute A* path for UI
        self.current_path = self.A_star(board, positions[2], horizontal_walls, vertical_walls, 2)
        opponent_path = self.A_star(board, positions[1], horizontal_walls, vertical_walls, 1)

        all_actions = self._get_all_possible_moves(board, positions, horizontal_walls, vertical_walls)

        best_score = float('-inf')
        best_action = None

        for action in all_actions:
            b2 = board.copy()
            h2 = horizontal_walls.copy()
            v2 = vertical_walls.copy()
            p2 = positions.copy()

            if isinstance(action, tuple) and len(action) == 3:
                i, j, ori = action
                if ori == 'horizontal':
                    h2[i, j] = 2; h2[i, j+1] = 2
                else:
                    v2[i, j] = 2; v2[i+1, j] = 2
            else:
                i, j = action
                ci, cj = p2[2]
                b2[ci, cj] = 0
                b2[i, j] = 2
                p2[2] = (i, j)

            score = self.evaluate(b2, p2, h2, v2)
            if score > best_score:
                best_score = score
                best_action = action
            
            if best_action is None:
                # Fallback: pick a random valid action
                if all_actions:
                    return all_actions[0]
                else:
                    return None  # No move possible: game over


        return best_action

    def evaluate(self, board, positions, h_walls, v_walls):
        ai_path = self.A_star(board, positions[2], h_walls, v_walls, 2)
        op_path = self.A_star(board, positions[1], h_walls, v_walls, 1)
        if ai_path is None: return float('-inf')
        if op_path is None: return float('inf')
        return len(op_path) - len(ai_path)

    def A_star(self, board, start, h_walls, v_walls, player):
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

            for move_cost, neighbor in self._get_valid_moves(board, current, h_walls, v_walls):
                if neighbor in visited:
                    continue
                new_cost = cost_so_far[current] + move_cost
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self._heuristic(neighbor, goal_row)
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current
        return None

    def _heuristic(self, pos: Tuple[int,int], goal_row: int) -> float:
        return abs(goal_row - pos[0])

    def _get_valid_moves(
        self,
        board: np.ndarray,
        position: Tuple[int, int],
        horizontal_walls: np.ndarray,
        vertical_walls: np.ndarray
    ) -> List[Tuple[float, Tuple[int, int]]]:
        i, j = position
        moves: List[Tuple[float, Tuple[int,int]]] = []
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni, nj = i+di, j+dj
            if not (0 <= ni < board.shape[0] and 0 <= nj < board.shape[1]):
                continue
            if board[ni, nj] != 0:
                continue
            if di != 0:
                w = min(i, ni)
                if horizontal_walls[w, j] != 0:
                    continue
            else:
                w = min(j, nj)
                if vertical_walls[i, w] != 0:
                    continue
            moves.append((1, (ni, nj)))
        return moves

    def _get_all_possible_moves(
        self,
        board: np.ndarray,
        positions: Dict[int, Tuple[int, int]],
        horizontal_walls: np.ndarray,
        vertical_walls: np.ndarray
    ) -> List[Union[Tuple[int,int], Tuple[int,int,str]]]:
        actions: List[Union[Tuple[int,int], Tuple[int,int,str]]] = []
        for _, move in self._get_valid_moves(board, positions[2], horizontal_walls, vertical_walls):
            actions.append(move)
        remaining = 20 - (np.count_nonzero(horizontal_walls) + np.count_nonzero(vertical_walls))
        if remaining > 0:
            for i in range(board.shape[0] - 1):
                for j in range(board.shape[1] - 1):
                    if horizontal_walls[i,j] == 0 and horizontal_walls[i,j+1] == 0:
                        actions.append((i,j,'horizontal'))
                    if vertical_walls[i,j] == 0 and vertical_walls[i+1,j] == 0:
                        actions.append((i,j,'vertical'))
        return actions

    def get_current_path(self):
        return self.current_path