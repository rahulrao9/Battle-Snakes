import math
import itertools
import random
from typing import Dict, List, Set, Tuple

class MCTSNode:
    __slots__ = [
        'state', 'parent', 'action_taken', 'children', 
        'visits', 'scores', 'untried_actions',
        'rave_visits', 'rave_scores', 'heuristic_scores'
    ]

    def __init__(self, state, parent=None, action_taken=None, heuristic_scores=None):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken 
        self.children: List['MCTSNode'] = []
        self.visits: int = 0
        
        # Max^n arrays for standard stats
        self.scores: Dict[str, float] = {sid: 0.0 for sid in state.snakes.keys()}
        
        # AMAF/RAVE arrays: track stats for moves played *anywhere* in the rollout
        self.rave_visits: Dict[str, int] = {sid: 0 for sid in state.snakes.keys()}
        self.rave_scores: Dict[str, float] = {sid: 0.0 for sid in state.snakes.keys()}
        
        # Dhanush's heuristic evaluation of THIS specific node's state
        self.heuristic_scores = heuristic_scores if heuristic_scores else {sid: 0.0 for sid in state.snakes.keys()}
        
        self.untried_actions = self._generate_all_joint_actions()

    def _generate_all_joint_actions(self):
        """Generates the Cartesian product of all legal moves for alive snakes."""
        alive_snakes = [sid for sid, s in self.state.snakes.items() if s.is_alive]
        
        # Dhanush's heuristic/engine will provide legal actions
        # e.g., legal_moves = {"Snake1": ["up", "left"], "Snake2": ["down"]}
        legal_moves_per_snake = {sid: self.state.get_legal_actions(sid) for sid in alive_snakes}
        
        # Prepare lists for itertools.product
        snake_ids = list(legal_moves_per_snake.keys())
        moves_lists = [legal_moves_per_snake[sid] for sid in snake_ids]
        
        joint_actions = []
        for combo in itertools.product(*moves_lists):
            # Map the generated combination back to the snake IDs
            action_dict = {snake_ids[i]: move for i, move in enumerate(combo)}
            joint_actions.append(action_dict)
            
        return joint_actions

    def expand(self):
        """Pops an action, steps the state, calculates heuristic, creates child."""
        action = self.untried_actions.pop()
        next_state = self.state.step(action) 
        
        # Calculate Progressive Bias early: Get Dhanush's heuristic for the new state
        # Assume evaluate_state returns a dict of scores like: {"Snake1": 0.8, "Snake2": 0.4}
        h_scores = Heuristic.evaluate_state(next_state) 
        
        child_node = MCTSNode(state=next_state, parent=self, action_taken=action, heuristic_scores=h_scores)
        self.children.append(child_node)
        return child_node

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0
    
    def best_child(self, my_snake_id: str, c_param: float = 1.414, rave_equiv: int = 1000, pb_weight: float = 10.0):
        """
        Calculates the ultimate UCB score blending Standard MCTS, RAVE, and Progressive Bias.
        """
        best_score = float('-inf')
        best_nodes = []

        for child in self.children:
            n_i = child.visits
            
            # 1. Standard Exploitation
            std_exploit = (child.scores[my_snake_id] / n_i) if n_i > 0 else 0.0
            
            # 2. RAVE Exploitation
            r_n_i = child.rave_visits[my_snake_id]
            rave_exploit = (child.rave_scores[my_snake_id] / r_n_i) if r_n_i > 0 else std_exploit
            
            # 3. Calculate Beta (RAVE weight). As standard visits increase, beta approaches 0.
            beta = r_n_i / (n_i + r_n_i + rave_equiv + 1e-6)
            
            # Blended Exploitation
            exploitation = (1 - beta) * std_exploit + beta * rave_exploit
            
            # 4. Standard Exploration (UCB1)
            exploration = c_param * math.sqrt(math.log(self.visits + 1) / (n_i + 1e-6))
            
            # 5. Progressive Bias (Inject Dhanush's heuristic, decays as node is visited)
            h_score = child.heuristic_scores.get(my_snake_id, 0.0)
            progressive_bias = (pb_weight * h_score) / (n_i + 1)
            
            # The Final Equation
            ucb_value = exploitation + exploration + progressive_bias
            
            if ucb_value > best_score:
                best_score = ucb_value
                best_nodes = [child]
            elif ucb_value == best_score:
                best_nodes.append(child)
                
        return random.choice(best_nodes)
    
    def simulate(self, depth_limit: int = 15) -> Tuple[dict, dict]:
        """
        Rolls out the game and tracks ALL actions taken for RAVE (AMAF).
        """
        current_state = self.state.clone()
        depth = 0
        
        # Track every move each snake makes during this specific rollout
        rollout_actions: Dict[str, Set[str]] = {sid: set() for sid in current_state.snakes.keys()}
        
        while not self._is_terminal_state(current_state) and depth < depth_limit:
            alive_snakes = [sid for sid, s in current_state.snakes.items() if s.is_alive]
            joint_action = {}
            
            for sid in alive_snakes:
                legal_moves = current_state.get_legal_actions(sid)
                move = random.choice(legal_moves) if legal_moves else "up"
                joint_action[sid] = move
                
                # Add move to AMAF tracker
                rollout_actions[sid].add(move)
                
            current_state = current_state.step(joint_action)
            depth += 1
            
        scores = self._calculate_scores(current_state)
        return scores, rollout_actions

    def _is_terminal_state(self, state) -> bool:
        alive_count = sum(1 for s in state.snakes.values() if s.is_alive)
        return alive_count <= 1 or state.turn >= 300 # 

    def _calculate_scores(self, state) -> dict:
        """
        Calculates the final score for each snake based on 80% survival and 20% length[cite: 21].
        If non-terminal, this should call Dhanush's Heuristic function.
        """
        # TODO: Dhanush will provide the actual evaluate_state(state) function.
        # For now, we return a basic placeholder calculation.
        scores = {}
        total_turns_possible = 300
        
        for sid, snake in state.snakes.items():
            if snake.is_alive:
                survival_score = 1.0  # Max survival score
            else:
                survival_score = state.turn / total_turns_possible
                
            # Normalize length (assuming max reasonable length is around 25)
            length_score = min(1.0, snake.length / 25.0) 
            
            # 80% weightage to survival, 20% to length [cite: 21]
            scores[sid] = (0.8 * survival_score) + (0.2 * length_score) 
            
        return scores

    def backpropagate(self, scores: dict, rollout_actions: dict):
        """
        Updates standard stats and RAVE stats simultaneously.
        """
        self.visits += 1
        for sid, score in scores.items():
            self.scores[sid] += score
            
            # RAVE Update: If the action that led to this node was played 
            # ANYWHERE in the rollout by this snake, update the RAVE stats.
            if self.action_taken and sid in self.action_taken:
                if self.action_taken[sid] in rollout_actions.get(sid, set()):
                    self.rave_visits[sid] += 1
                    self.rave_scores[sid] += score
            
        if self.parent:
            self.parent.backpropagate(scores, rollout_actions)