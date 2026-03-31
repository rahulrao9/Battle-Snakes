import time
import math
import random
from typing import Dict, List, Set, Tuple
from collections import deque
from server import run_server
import os
# Import your Phase 1 State Evaluator
from heuristic_agent import evaluate_state as heavy_voronoi_eval

MOVES = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}
C_PARAM = float(os.environ.get("MCTS_C_PARAM", "1.414"))
DEPTH_LIMIT = int(os.environ.get("MCTS_DEPTH_LIMIT", "15"))
PB_WEIGHT = float(os.environ.get("MCTS_PB_WEIGHT", "5.0"))
def get_neighbors(x, y, width, height):
    return [(nx, ny) for nx, ny in [(x, y+1), (x, y-1), (x-1, y), (x+1, y)] 
            if 0 <= nx < width and 0 <= ny < height]

# ==========================================
# Forward Model (Fast Physics Engine)
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

    def flood_fill_space(self, start_pos: Tuple[int, int], my_length: int) -> int:
        """Calculates exact open space to prevent coiling into dead ends."""
        obs = set()
        for s in self.snakes.values():
            if not s.is_alive: continue
            body_list = list(s.body)
            obs.update(body_list[:-1] if s.health < 100 and len(body_list) > 1 else body_list)
            
        visited = set()
        queue = deque([start_pos])
        count = 0
        
        while queue and count < my_length:
            curr = queue.popleft()
            if curr not in visited and curr not in obs:
                visited.add(curr)
                count += 1
                for nx, ny in get_neighbors(curr[0], curr[1], self.board_width, self.board_height):
                    if (nx, ny) not in visited and (nx, ny) not in obs:
                        queue.append((nx, ny))
        return count

    def get_root_moves(self, sid: str) -> List[str]:
        """
        THE FIX: Soft-Filter Flood Fill. 
        Only runs at the root node. Guarantees we never enter a death spiral if a safe path exists.
        """
        snake = self.snakes.get(sid)
        if not snake or not snake.is_alive: return ["up"]
        
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

        moves_space = {}
        for m, (dx, dy) in MOVES.items():
            nx, ny = snake.head[0]+dx, snake.head[1]+dy
            if 0 <= nx < self.board_width and 0 <= ny < self.board_height:
                target = (nx, ny)
                if target not in obs:
                    # Check double our length to be absolutely sure the path is open
                    space = self.flood_fill_space(target, snake.length * 2) 
                    
                    # Softly penalize head-to-head danger zones by cutting their perceived space in half
                    if target in danger_zone:
                        space = space // 2
                        
                    moves_space[m] = space

        if not moves_space:
            return ["up"] # Completely wrapped.

        max_space = max(moves_space.values())
        if max_space < snake.length:
            # Trapped! Return the path that delays death the longest.
            return [m for m, sp in moves_space.items() if sp == max_space]
        else:
            # Return all paths that offer enough room to survive.
            return [m for m, sp in moves_space.items() if sp >= snake.length]

    def get_safe_moves(self, sid: str) -> List[str]:
        """Fast categorization for deep tree nodes (No Flood Fill here to maintain speed)."""
        snake = self.snakes[sid]
        if not snake.is_alive: return []
        
        obs = set()
        for s in self.snakes.values():
            if not s.is_alive: continue
            body_list = list(s.body)
            obs.update(body_list[:-1] if s.health < 100 and len(body_list) > 1 else body_list)
            
        safe, risky = [], []
        for m, (dx, dy) in MOVES.items():
            nx, ny = snake.head[0]+dx, snake.head[1]+dy
            if 0 <= nx < self.board_width and 0 <= ny < self.board_height:
                if (nx, ny) not in obs:
                    safe.append(m)
        return safe if safe else ["up"]

    def get_guided_move(self, sid: str) -> str:
        """Heuristic Guide: hazard avoidance + starvation prevention."""
        safe_moves = self.get_safe_moves(sid)
        if not safe_moves: return "up"
        
        snake = self.snakes[sid]
        head = snake.head
        
        non_hazard = [m for m in safe_moves if (head[0]+MOVES[m][0], head[1]+MOVES[m][1]) not in self.hazards]
        choices = non_hazard if non_hazard and snake.health > 20 else safe_moves
        if not choices: return "up"
        
        if self.food:
            for m in choices:
                if (head[0]+MOVES[m][0], head[1]+MOVES[m][1]) in self.food:
                    return m
                    
        if (snake.health < 60 or snake.length < 10) and self.food:
            best_m = choices[0]
            best_dist = float('inf')
            for m in choices:
                nx, ny = head[0]+MOVES[m][0], head[1]+MOVES[m][1]
                min_dist = min(abs(nx - fx) + abs(ny - fy) for fx, fy in self.food)
                if min_dist < best_dist:
                    best_dist = min_dist
                    best_m = m
            return best_m
            
        return random.choice(choices)

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
# Optimized MCTS Architecture
# ==========================================
class MCTSNode:
    __slots__ = ['state', 'my_id', 'parent', 'action_taken', 'children', 'visits', 'score', 'untried_actions', 'pb_score']

    def __init__(self, state: GameState, my_id: str, parent=None, action_taken=None, pb_score=0.0):
        self.state = state
        self.my_id = my_id
        self.parent = parent
        self.action_taken = action_taken 
        self.children = []
        
        self.visits = 0
        self.score = 0.0
        self.pb_score = pb_score # Progressive Bias score from Voronoi heuristic
        
        # ROOT Node gets the smart Flood Fill. Deep nodes get fast basic checks.
        if self.parent is None:
            self.untried_actions = self.state.get_root_moves(self.my_id)
        else:
            self.untried_actions = self.state.get_safe_moves(self.my_id)
            
        random.shuffle(self.untried_actions)

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0
    
    def expand(self):
        if not self.untried_actions: return self
            
        my_action = self.untried_actions.pop()
        
        # Decoupled Search: We pick a move, assume opponents use logic
        joint_action = {self.my_id: my_action}
        for sid, snake in self.state.snakes.items():
            if sid != self.my_id and snake.is_alive:
                joint_action[sid] = self.state.get_guided_move(sid)
                
        next_state = self.state.step(joint_action) 
        
        # Heavy Voronoi Progressive Bias ONLY on immediate children of the root
        h_score = 0.0
        if self.parent is None: 
            scores_dict = heavy_voronoi_eval(next_state)
            h_score = scores_dict.get(self.my_id, 0.0)
            
        child_node = MCTSNode(state=next_state, my_id=self.my_id, parent=self, action_taken=my_action, pb_score=h_score)
        self.children.append(child_node)
        return child_node

    def best_child(self, c_param: float = C_PARAM, pb_weight: float = PB_WEIGHT):
        best_ucb = float('-inf')
        best_nodes = []

        for child in self.children:
            if child.visits == 0: continue
            
            exploitation = child.score / child.visits
            exploration = c_param * math.sqrt(math.log(self.visits) / child.visits)
            progressive_bias = (pb_weight * child.pb_score) / (child.visits + 1)
            
            ucb_value = exploitation + exploration + progressive_bias
            
            if ucb_value > best_ucb:
                best_ucb = ucb_value
                best_nodes = [child]
            elif ucb_value == best_ucb:
                best_nodes.append(child)
                
        return random.choice(best_nodes) if best_nodes else self.children[0]
    
    def simulate(self, depth_limit: int = DEPTH_LIMIT) -> float:
        current_state = self.state.clone()
        depth = 0
        
        while depth < depth_limit:
            alive = [sid for sid, s in current_state.snakes.items() if s.is_alive]
            if len(alive) <= 1 or self.my_id not in alive:
                break 
                
            joint_action = {sid: current_state.get_guided_move(sid) for sid in alive}
            current_state = current_state.step(joint_action)
            depth += 1
            
        me = current_state.snakes.get(self.my_id)
        if not me or not me.is_alive: 
            return (depth / depth_limit) * 0.5 # Penalty for dying early
            
        return 1.0 + (0.2 * (me.length / 25.0))

    def backpropagate(self, score: float):
        self.visits += 1
        self.score += score
        if self.parent:
            self.parent.backpropagate(score)

class MCTSAgent:
    # 650ms ensures we are perfectly safe under 1000ms while doing massive iterations
    def __init__(self, my_id: str, time_limit_ms: int = 650):
        self.my_id = my_id
        self.time_limit = time_limit_ms / 1000.0 

    def search(self, root_state: GameState) -> str:
        start_time = time.time()
        root = MCTSNode(state=root_state, my_id=self.my_id)
        
        if not root.untried_actions and not root.children:
            return "up" # Absolute worst-case scenario fallback

        iterations = 0
        while time.time() - start_time < self.time_limit:
            node = self._select(root)
            
            if node.state.snakes.get(self.my_id, Snake("x", [], 0, False)).is_alive and not node.is_fully_expanded():
                node = node.expand()
                
            score = node.simulate(depth_limit=15)
            node.backpropagate(score)
            iterations += 1

        print(f"MCTS-V10 Iterations: {iterations} | Time: {time.time() - start_time:.3f}s")
        return self._best_move(root)

    def _select(self, node: MCTSNode) -> MCTSNode:
        while node.state.snakes.get(self.my_id, Snake("x", [], 0, False)).is_alive and node.is_fully_expanded():
            if not node.children: break
            node = node.best_child()
        return node

    def _best_move(self, root: MCTSNode) -> str:
        if not root.children: 
            safe = root.state.get_safe_moves(self.my_id)
            return safe[0] if safe else "up"
            
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action_taken

# ==========================================
# Battlesnake Endpoints
# ==========================================
def info() -> Dict:
    return {"apiversion": "1", "author": "MGAIA_MCTS_01_F2IMPROVE", "color": "#e3242b", "head": "default", "tail": "default"}

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