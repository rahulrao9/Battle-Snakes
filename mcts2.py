import time
import math
import random
from typing import Dict, List, Set, Tuple
from collections import deque
from server import run_server

MOVES = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}

# ==========================================
# Forward Model (Physics Engine)
# ==========================================
class Snake:
    __slots__ = ['id', 'body', 'health', 'is_alive']
    def __init__(self, sid: str, body: deque, health: int, is_alive: bool = True):
        self.id = sid
        self.body = body  
        self.health = health
        self.is_alive = is_alive

    @property
    def head(self) -> Tuple[int, int]:
        return self.body[0]

    @property
    def length(self) -> int:
        return len(self.body)

    def clone(self):
        return Snake(self.id, deque(self.body), self.health, self.is_alive)

class GameState:
    __slots__ = ['board_width', 'board_height', 'turn', 'snakes', 'food', 'hazards']
    
    def __init__(self, width: int, height: int, turn: int, snakes: Dict[str, Snake], food: Set[Tuple[int, int]], hazards: Set[Tuple[int, int]]):
        self.board_width = width
        self.board_height = height
        self.turn = turn
        self.snakes = snakes
        self.food = food
        self.hazards = hazards

    @classmethod
    def from_json(cls, game_state: dict):
        width = game_state['board']['width']
        height = game_state['board']['height']
        turn = game_state['turn']
        food = {(f['x'], f['y']) for f in game_state['board']['food']}
        hazards = {(h['x'], h['y']) for h in game_state['board']['hazards']}
        
        snakes = {}
        for s in game_state['board']['snakes']:
            body = deque((pt['x'], pt['y']) for pt in s['body'])
            snakes[s['id']] = Snake(s['id'], body, s['health'])
            
        return cls(width, height, turn, snakes, food, hazards)

    def clone(self):
        cloned_snakes = {sid: s.clone() for sid, s in self.snakes.items()}
        return GameState(self.board_width, self.board_height, self.turn, cloned_snakes, set(self.food), self.hazards)

    def get_safe_moves(self, sid: str) -> List[str]:
        """Fast categorization of safe moves."""
        snake = self.snakes[sid]
        if not snake.is_alive: return []
        
        obs = set()
        for s in self.snakes.values():
            if not s.is_alive: continue
            body_list = list(s.body)
            obs.update(body_list[:-1] if s.health < 100 and len(body_list) > 1 else body_list)
            
        danger_zone = set()
        for opp in self.snakes.values():
            if opp.id == sid or not opp.is_alive: continue
            if opp.length >= snake.length:
                for dx, dy in MOVES.values():
                    danger_zone.add((opp.head[0]+dx, opp.head[1]+dy))

        safe, risky = [], []
        head = snake.head
        for m, (dx, dy) in MOVES.items():
            nx, ny = head[0]+dx, head[1]+dy
            if 0 <= nx < self.board_width and 0 <= ny < self.board_height:
                if (nx, ny) not in obs:
                    if (nx, ny) in danger_zone:
                        risky.append(m)
                    else:
                        safe.append(m)
                        
        return safe if safe else risky

    def get_guided_move(self, sid: str) -> str:
        """
        ASSIGNMENT REQUIREMENT: Heuristic-Guided Simulations.
        Actively paths to food if health is low, preventing starvation.
        """
        safe_moves = self.get_safe_moves(sid)
        if not safe_moves: return "up"
        
        snake = self.snakes[sid]
        head = snake.head
        
        # 1. Immediate Food Grab
        if self.food:
            for m in safe_moves:
                nx, ny = head[0] + MOVES[m][0], head[1] + MOVES[m][1]
                if (nx, ny) in self.food:
                    return m
                    
        # 2. Path to Food if hungry
        if (snake.health < 60 or snake.length < 10) and self.food:
            best_m = safe_moves[0]
            best_dist = float('inf')
            for m in safe_moves:
                nx, ny = head[0] + MOVES[m][0], head[1] + MOVES[m][1]
                min_dist = min(abs(nx - fx) + abs(ny - fy) for fx, fy in self.food)
                if min_dist < best_dist:
                    best_dist = min_dist
                    best_m = m
            return best_m
            
        # 3. Otherwise, random safe move
        return random.choice(safe_moves)

    def step(self, joint_action: Dict[str, str]):
        next_state = self.clone()
        next_state.turn += 1
        
        hazard_stack = min(4, max(0, (next_state.turn - 1) // 25)) if next_state.turn >= 26 else 0
        hazard_damage = 14 * hazard_stack
        eaten_food = set()
        
        for sid, action in joint_action.items():
            snake = next_state.snakes.get(sid)
            if not snake or not snake.is_alive: continue
                
            dx, dy = MOVES[action]
            new_head = (snake.head[0] + dx, snake.head[1] + dy)
            
            if new_head in next_state.food:
                eaten_food.add(new_head)
                snake.health = 100
            else:
                snake.health -= 1
                snake.body.pop() 
                
            if new_head in next_state.hazards:
                snake.health -= (1 + hazard_damage)
                
            snake.body.appendleft(new_head) 
            
        next_state.food -= eaten_food
        
        body_counts = {}
        for s in next_state.snakes.values():
            if not s.is_alive: continue
            for pt in s.body:
                body_counts[pt] = body_counts.get(pt, 0) + 1
                
        dead_this_turn = set()
        for sid, snake in next_state.snakes.items():
            if not snake.is_alive: continue
            head = snake.head
            
            if not (0 <= head[0] < next_state.board_width and 0 <= head[1] < next_state.board_height) or snake.health <= 0:
                dead_this_turn.add(sid)
                continue
                
            if body_counts[head] > 1:
                collided = [s for s in next_state.snakes.values() if s.is_alive and s.head == head]
                if len(collided) > 1:
                    max_len = max(s.length for s in collided)
                    longest = [s for s in collided if s.length == max_len]
                    if len(longest) > 1 or snake not in longest:
                        dead_this_turn.add(sid)
                else:
                    dead_this_turn.add(sid)

        for sid in dead_this_turn:
            next_state.snakes[sid].is_alive = False
            
        return next_state


# ==========================================
# Decoupled RAVE MCTS Architecture
# ==========================================
class MCTSNode:
    __slots__ = ['state', 'my_id', 'parent', 'action_taken', 'children', 'visits', 'score', 'untried_actions', 'rave_visits', 'rave_score']

    def __init__(self, state: GameState, my_id: str, parent=None, action_taken=None):
        self.state = state
        self.my_id = my_id
        self.parent = parent
        self.action_taken = action_taken 
        self.children = []
        
        # Standard Stats
        self.visits = 0
        self.score = 0.0
        
        # RAVE Stats (AMAF)
        self.rave_visits = 0
        self.rave_score = 0.0
        
        # Decoupled Tree: Branching factor is max 4 (only our moves)
        self.untried_actions = self.state.get_safe_moves(self.my_id)
        random.shuffle(self.untried_actions)

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0
    
    def expand(self):
        my_action = self.untried_actions.pop()
        
        # Decoupled Transition: We pick our move, opponents use guided heuristic
        joint_action = {self.my_id: my_action}
        for sid, snake in self.state.snakes.items():
            if sid != self.my_id and snake.is_alive:
                joint_action[sid] = self.state.get_guided_move(sid)
                
        next_state = self.state.step(joint_action) 
        child_node = MCTSNode(state=next_state, my_id=self.my_id, parent=self, action_taken=my_action)
        self.children.append(child_node)
        return child_node

    def best_child(self, c_param: float = 1.414, rave_equiv: int = 150):
        """Calculates UCB incorporating RAVE logic."""
        best_ucb = float('-inf')
        best_nodes = []

        for child in self.children:
            if child.visits == 0: continue
            
            # Standard Exploitation
            std_exploit = child.score / child.visits
            
            # RAVE Exploitation
            rave_exploit = (child.rave_score / child.rave_visits) if child.rave_visits > 0 else std_exploit
            
            # Beta Weighting (Blends RAVE into Standard as visits increase)
            beta = child.rave_visits / (child.visits + child.rave_visits + rave_equiv + 1e-6)
            exploitation = (1 - beta) * std_exploit + beta * rave_exploit
            
            exploration = c_param * math.sqrt(math.log(self.visits) / child.visits)
            
            ucb_value = exploitation + exploration
            
            if ucb_value > best_ucb:
                best_ucb = ucb_value
                best_nodes = [child]
            elif ucb_value == best_ucb:
                best_nodes.append(child)
                
        return random.choice(best_nodes) if best_nodes else self.children[0]
    
    def simulate(self, depth_limit: int = 12) -> Tuple[float, Set[str]]:
        """
        Runs a Heuristic-Guided Rollout and tracks AMAF actions for RAVE.
        """
        current_state = self.state.clone()
        depth = 0
        
        # AMAF Tracker: Records the first time we use a move in the rollout
        rollout_moves_played = set()
        
        while depth < depth_limit:
            alive = [sid for sid, s in current_state.snakes.items() if s.is_alive]
            if len(alive) <= 1 or self.my_id not in alive:
                break 
                
            joint_action = {}
            for sid in alive:
                move = current_state.get_guided_move(sid)
                joint_action[sid] = move
                if sid == self.my_id:
                    rollout_moves_played.add(move)
                    
            current_state = current_state.step(joint_action)
            depth += 1
            
        me = current_state.snakes.get(self.my_id)
        if not me or not me.is_alive: 
            return 0.0, rollout_moves_played
            
        # Binary survival + growth reward
        return 1.0 + (0.2 * (me.length / 25.0)), rollout_moves_played

    def backpropagate(self, score: float, rollout_moves: Set[str]):
        self.visits += 1
        self.score += score
        
        # RAVE Update: If the action leading to this node was played ANYWHERE in the rollout
        if self.action_taken and self.action_taken in rollout_moves:
            self.rave_visits += 1
            self.rave_score += score
            
        if self.parent:
            self.parent.backpropagate(score, rollout_moves)

class MCTSAgent:
    def __init__(self, my_id: str, time_limit_ms: int = 800):
        self.my_id = my_id
        self.time_limit = time_limit_ms / 1000.0 

    def search(self, root_state: GameState) -> str:
        start_time = time.time()
        root = MCTSNode(state=root_state, my_id=self.my_id)

        iterations = 0
        while time.time() - start_time < self.time_limit:
            node = self._select(root)
            
            if node.state.snakes.get(self.my_id, Snake("x", [], 0, False)).is_alive and not node.is_fully_expanded():
                node = node.expand()
                
            score, rollout_moves = node.simulate(depth_limit=12)
            
            # Include the node's own action in the rollout tracking for RAVE
            if node.action_taken:
                rollout_moves.add(node.action_taken)
                
            node.backpropagate(score, rollout_moves)
            iterations += 1

        print(f"Decoupled RAVE Iterations: {iterations} | Time: {time.time() - start_time:.3f}s")
        return self._best_move(root)

    def _select(self, node: MCTSNode) -> MCTSNode:
        while node.state.snakes.get(self.my_id, Snake("x", [], 0, False)).is_alive and node.is_fully_expanded():
            if not node.children: break
            node = node.best_child()
        return node

    def _best_move(self, root: MCTSNode) -> str:
        if not root.children: 
            safe_moves = root.state.get_safe_moves(self.my_id)
            return safe_moves[0] if safe_moves else "up"
            
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action_taken

# ==========================================
# Battlesnake Endpoints
# ==========================================
def info() -> Dict:
    return {"apiversion": "1", "author": "MGAIA_MCTS_Decoupled", "color": "#7724e3", "head": "default", "tail": "default"}

def start(game_state: Dict): pass
def end(game_state: Dict): pass

def move(game_state: Dict) -> Dict:
    my_id = game_state['you']['id']
    root_state = GameState.from_json(game_state) 
    
    agent = MCTSAgent(my_id=my_id)
    best_move = agent.search(root_state)
    
    print(f"Turn {game_state['turn']} | MCTS Decided: {best_move}")
    return {"move": best_move}

if __name__ == "__main__":
    run_server({"info": info, "start": start, "move": move, "end": end})