import numpy as np 
import heapq
from typing import Tuple, List, Dict, Optional, Union
import copy

class MinimaxAI:
    def __init__(self, max_depth=2, wall_radius=3):
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
        """Détermine le meilleur coup à jouer en utilisant minimax avec alpha-beta pruning"""
        # Recalcule le chemin A* pour l'affichage UI

        import time
        start_time = time.time()
        
        # Recalcule le chemin A* pour l'affichage UI
        self.current_path = self.A_star(board, positions[2], horizontal_walls, vertical_walls, 2)
        
        # Lance minimax avec alpha-beta pruning
        best_score = float('-inf')
        best_move = None
        alpha = float('-inf')
        beta = float('inf')
        
        # Récupère tous les coups possibles
        possible_moves = self._get_all_possible_moves(board, positions, horizontal_walls, vertical_walls)
        
        # Trie d'abord les mouvements des pions (plus susceptibles d'être bons)
        possible_moves.sort(key=lambda x: not isinstance(x, tuple) or len(x) != 3)
        
        for move in possible_moves:
            # Vérifie si le temps est écoulé
            if time.time() - start_time > self.timeout:
                print("Timeout reached, returning best move found so far")
                break
            # Crée une copie de l'état du jeu
            new_board = board.copy()
            new_horizontal_walls = horizontal_walls.copy()
            new_vertical_walls = vertical_walls.copy()
            new_positions = positions.copy()
            new_remaining_fences = {1: 10 - np.count_nonzero(horizontal_walls) - np.count_nonzero(vertical_walls), 
                                   2: 10 - np.count_nonzero(horizontal_walls) - np.count_nonzero(vertical_walls)}
            
            # Applique le coup
            if isinstance(move, tuple) and len(move) == 3:  # Placement d'un mur
                i, j, ori = move
                if ori == 'horizontal':
                    new_horizontal_walls[i, j] = 2
                    new_horizontal_walls[i, j+1] = 2
                else:
                    new_vertical_walls[i, j] = 2
                    new_vertical_walls[i+1, j] = 2
                new_remaining_fences[2] -= 1
            else:  # Déplacement du pion
                i, j = move
                ci, cj = new_positions[2]
                new_board[ci, cj] = 0
                new_board[i, j] = 2
                new_positions[2] = (i, j)
            
            # Appelle minimax récursivement
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
                False  # Nœud min (pour l'adversaire)
            )
            
            if score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break  # Élagage alpha-beta
        # Ajoutons une vérification pour le cas où tous les coups sont perdants
        if best_move is None and possible_moves:
            # Si tous les coups conduisent à une défaite, prend le premier disponible
            best_move = possible_moves[0]

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
        """Implémentation récursive de l'algorithme minimax avec alpha-beta pruning"""
        # Vérifie si un joueur a gagné
        player1_won = positions[1][0] == 0
        player2_won = positions[2][0] == board.shape[0] - 1
        
        if player1_won:
            return float('-inf')  # L'adversaire a gagné, pire score possible
        elif player2_won:
            return float('inf')   # L'IA a gagné, meilleur score possible
        
        # Si profondeur max atteinte, évalue la position
        if depth == 0:
            return self._evaluate(board, positions, horizontal_walls, vertical_walls, remaining_fences)
        
        # Génère les coups possibles
        possible_moves = self._get_all_possible_moves(board, positions, horizontal_walls, vertical_walls, player)
        
        # Optimisation: limite le nombre de murs considérés
        possible_moves = self._filter_wall_moves(possible_moves, positions, player)
        
        # Trie les mouvements pour améliorer l'élagage alpha-beta
        possible_moves.sort(key=lambda x: not isinstance(x, tuple) or len(x) != 3)
        
        if is_maximizing:  # Nœud max (pour l'IA)
            max_eval = float('-inf')
            for move in possible_moves:
                # Crée une copie de l'état du jeu
                new_board = board.copy()
                new_horizontal_walls = horizontal_walls.copy()
                new_vertical_walls = vertical_walls.copy()
                new_positions = positions.copy()
                new_remaining_fences = remaining_fences.copy()
                
                # Applique le coup
                if isinstance(move, tuple) and len(move) == 3:  # Placement d'un mur
                    i, j, ori = move
                    if ori == 'horizontal':
                        new_horizontal_walls[i, j] = player
                        new_horizontal_walls[i, j+1] = player
                    else:
                        new_vertical_walls[i, j] = player
                        new_vertical_walls[i+1, j] = player
                    new_remaining_fences[player] -= 1
                else:  # Déplacement du pion
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
                    False
                )
                
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Élagage alpha-beta
            
            return max_eval
        else:  # Nœud min (pour l'adversaire)
            min_eval = float('inf')
            for move in possible_moves:
                # Crée une copie de l'état du jeu
                new_board = board.copy()
                new_horizontal_walls = horizontal_walls.copy()
                new_vertical_walls = vertical_walls.copy()
                new_positions = positions.copy()
                new_remaining_fences = remaining_fences.copy()
                
                # Applique le coup
                if isinstance(move, tuple) and len(move) == 3:  # Placement d'un mur
                    i, j, ori = move
                    if ori == 'horizontal':
                        new_horizontal_walls[i, j] = player
                        new_horizontal_walls[i, j+1] = player
                    else:
                        new_vertical_walls[i, j] = player
                        new_vertical_walls[i+1, j] = player
                    new_remaining_fences[player] -= 1
                else:  # Déplacement du pion
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
        """Filtre les mouvements de murs pour ne garder que ceux proches des joueurs"""
        filtered_moves = []
        
        for move in moves:
            # Conserve tous les mouvements qui ne sont pas des placements de murs
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
        """
        Fonction d'évaluation pour l'algorithme minimax.
        Plus le score est élevé, plus la position est favorable pour l'IA (joueur 2).
        """
        # Calcule les chemins les plus courts
        ai_path = self.A_star(board, positions[2], horizontal_walls, vertical_walls, 2)
        op_path = self.A_star(board, positions[1], horizontal_walls, vertical_walls, 1)
        
        # Si un joueur est bloqué, retourne une valeur extrême
        if ai_path is None:
            return float('-inf')  # L'IA est bloquée
        if op_path is None:
            return float('inf')   # L'adversaire est bloqué
        
        # Prend en compte le nombre de murs restants
        fence_advantage = remaining_fences[2] - remaining_fences[1]
        
        # Coefficient pour les murs (les murs sont importants mais moins que la distance)
        fence_weight = 0.5
        
        # Formule finale : diff_chemins + avantage_murs * poids_murs
        return fence_advantage * fence_weight - len(ai_path)*3 + len(op_path)

    def A_star(
        self, 
        board: np.ndarray, 
        start: Tuple[int, int],
        horizontal_walls: np.ndarray, 
        vertical_walls: np.ndarray, 
        player: int
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Implémente l'algorithme A* pour trouver le chemin le plus court.
        Retourne None si aucun chemin n'est trouvé.
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
        """Heuristique pour A* : distance de Manhattan à la ligne d'arrivée"""
        return abs(goal_row - pos[0])

    def _get_valid_moves(
        self,
        board: np.ndarray,
        position: Tuple[int, int],
        horizontal_walls: np.ndarray,
        vertical_walls: np.ndarray
    ) -> List[Tuple[float, Tuple[int, int]]]:
        """Retourne les mouvements valides depuis une position donnée"""
        i, j = position
        moves: List[Tuple[float, Tuple[int,int]]] = []
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:  # Haut, Bas, Gauche, Droite
            ni, nj = i+di, j+dj
            if not (0 <= ni < board.shape[0] and 0 <= nj < board.shape[1]):
                continue
            if board[ni, nj] != 0:
                continue
            if di != 0:  # Mouvement vertical
                w = min(i, ni)
                if horizontal_walls[w, j] != 0:
                    continue
            else:  # Mouvement horizontal
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
        vertical_walls: np.ndarray,
        player: int = 2  # Joueur pour lequel calculer les coups (par défaut l'IA)
    ) -> List[Union[Tuple[int,int], Tuple[int,int,str]]]:
        """Calcule tous les coups possibles pour un joueur donné"""
        actions: List[Union[Tuple[int,int], Tuple[int,int,str]]] = []
        
        # Mouvements du pion
        current_pos = positions[player]
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:  # Haut, Bas, Gauche, Droite
            ni, nj = current_pos[0] + di, current_pos[1] + dj
            if not (0 <= ni < board.shape[0] and 0 <= nj < board.shape[1]):
                continue
            
            # Vérifie s'il n'y a pas de mur entre la position actuelle et la nouvelle position
            can_move = True
            if di != 0:  # Mouvement vertical
                w = min(current_pos[0], ni)
                if horizontal_walls[w, current_pos[1]] != 0:
                    can_move = False
            else:  # Mouvement horizontal
                w = min(current_pos[1], nj)
                if vertical_walls[current_pos[0], w] != 0:
                    can_move = False
            
            if not can_move:
                continue
                
            # Vérifie si la case est vide
            if board[ni, nj] == 0:
                actions.append((ni, nj))
            else:  # Sauter par-dessus un joueur si possible
                # Case occupée, essaie de sauter par-dessus
                ni2, nj2 = ni + di, nj + dj
                if (0 <= ni2 < board.shape[0] and 0 <= nj2 < board.shape[1] and 
                    board[ni2, nj2] == 0):
                    # Vérifie s'il n'y a pas de mur entre le pion et la case d'arrivée
                    can_jump = True
                    if di != 0:  # Mouvement vertical
                        w = min(ni, ni2)
                        if horizontal_walls[w, nj] != 0:
                            can_jump = False
                    else:  # Mouvement horizontal
                        w = min(nj, nj2)
                        if vertical_walls[ni, w] != 0:
                            can_jump = False
                    
                    if can_jump:
                        actions.append((ni2, nj2))
        
        # Placements de murs
        # Compte le nombre de murs déjà placés
        total_walls = np.count_nonzero(horizontal_walls) + np.count_nonzero(vertical_walls)
        remaining = 20 - total_walls
        player_remaining = remaining // 2  # Approximatif
        
        if player_remaining > 0:
            for i in range(board.shape[0] - 1):
                for j in range(board.shape[1] - 1):
                    # Mur horizontal
                    if horizontal_walls[i, j] == 0 and horizontal_walls[i, j+1] == 0:
                        # Vérifie si le placement est valide (ne bloque pas tous les chemins)
                        temp_h_walls = horizontal_walls.copy()
                        temp_h_walls[i, j] = player
                        temp_h_walls[i, j+1] = player
                        
                        if (self._check_paths_exist(board, positions, temp_h_walls, vertical_walls)):
                            actions.append((i, j, 'horizontal'))
                    
                    # Mur vertical
                    if vertical_walls[i, j] == 0 and vertical_walls[i+1, j] == 0:
                        # Vérifie si le placement est valide (ne bloque pas tous les chemins)
                        temp_v_walls = vertical_walls.copy()
                        temp_v_walls[i, j] = player
                        temp_v_walls[i+1, j] = player
                        
                        if (self._check_paths_exist(board, positions, horizontal_walls, temp_v_walls)):
                            actions.append((i, j, 'vertical'))
        
        return actions

    def _check_paths_exist(self, board, positions, h_walls, v_walls):
        """Vérifie si les deux joueurs ont encore un chemin vers leur objectif"""
        # Vérifie que joueur 1 a un chemin vers la ligne du haut
        player1_path = self.A_star(board, positions[1], h_walls, v_walls, 1)
        if player1_path is None:
            return False
            
        # Vérifie que joueur 2 a un chemin vers la ligne du bas
        player2_path = self.A_star(board, positions[2], h_walls, v_walls, 2)
        if player2_path is None:
            return False
            
        return True

    def get_current_path(self):
        """Retourne le chemin actuel calculé pour l'affichage dans l'interface"""
        return self.current_path