import time
import math
import random
from typing import Dict, List, Set, Tuple
from collections import deque
from server import run_server
#best parameters
# Best Trial Score: 0.400
C_PARAM: 2.207
DEPTH_LIMIT: 13

MOVES = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}

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

    def get_legal_moves(self, sid: str) -> List[str]:
        """Returns physically possible moves (avoids walls and visible bodies). No heuristic guidance."""
        snake = self.snakes.get(sid)
        if not snake or not snake.is_alive: return []
        
        obs = set()
        for s in self.snakes.values():
            if not s.is_alive: continue
            body_list = list(s.body)
            obs.update(body_list[:-1] if s.health < 100 and len(body_list) > 1 else body_list)
            
        moves = []
        for m, (dx, dy) in MOVES.items():
            nx, ny = snake.head[0] + dx, snake.head[1] + dy
            if 0 <= nx < self.board_width and 0 <= ny < self.board_height:
                if (nx, ny) not in obs:
                    moves.append(m)
                    
        return moves if moves else ["up"] 

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
# Vanilla MCTS Architecture
# ==========================================
class MCTSNode:
    __slots__ = ['state', 'my_id', 'parent', 'action_taken', 'children', 'visits', 'score', 'untried_actions']

    def __init__(self, state: GameState, my_id: str, parent=None, action_taken=None):
        self.state = state
        self.my_id = my_id
        self.parent = parent
        self.action_taken = action_taken 
        self.children = []
        
        self.visits = 0
        self.score = 0.0
        
        # Only check physical legal moves, no flood fill or danger zone pruning
        self.untried_actions = self.state.get_legal_moves(self.my_id)
        random.shuffle(self.untried_actions)

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0
    
    def expand(self):
        if not self.untried_actions: return self
            
        my_action = self.untried_actions.pop()
        
        # Opponents take purely random physical moves during tree transitions
        joint_action = {self.my_id: my_action}
        for sid, snake in self.state.snakes.items():
            if sid != self.my_id and snake.is_alive:
                moves = self.state.get_legal_moves(sid)
                joint_action[sid] = random.choice(moves)
                
        next_state = self.state.step(joint_action) 
        child_node = MCTSNode(state=next_state, my_id=self.my_id, parent=self, action_taken=my_action)
        self.children.append(child_node)
        return child_node

    def best_child(self, c_param: float = C_PARAM):
        best_ucb = float('-inf')
        best_nodes = []

        for child in self.children:
            if child.visits == 0: continue
            
            # Standard UCB1: Pure Exploitation + Pure Exploration
            exploitation = child.score / child.visits
            exploration = c_param * math.sqrt(math.log(self.visits) / child.visits)
            
            ucb_value = exploitation + exploration
            
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
                
            # Purely Random Rollouts
            joint_action = {}
            for sid in alive:
                moves = current_state.get_legal_moves(sid)
                joint_action[sid] = random.choice(moves)
                
            current_state = current_state.step(joint_action)
            depth += 1
            
        me = current_state.snakes.get(self.my_id)
        if not me or not me.is_alive: 
            return (depth / depth_limit) * 0.5 
            
        survival = 1.0
        health_bonus = me.health / 200.0
        length_bonus = me.length / 50.0
        return survival + health_bonus + length_bonus

    def backpropagate(self, score: float):
        self.visits += 1
        self.score += score
        if self.parent:
            self.parent.backpropagate(score)

class MCTSAgent:
    def __init__(self, my_id: str, time_limit_ms: int = 650):
        self.my_id = my_id
        self.time_limit = time_limit_ms / 1000.0 

    def search(self, root_state: GameState) -> str:
        start_time = time.time()
        
        root = MCTSNode(state=root_state, my_id=self.my_id)
        if not root.untried_actions and not root.children:
            return "up" 

        iterations = 0
        while time.time() - start_time < self.time_limit:
            node = self._select(root)
            
            if node.state.snakes.get(self.my_id, Snake("x", [], 0, False)).is_alive and not node.is_fully_expanded():
                node = node.expand()
                
            score = node.simulate(depth_limit=DEPTH_LIMIT)
            node.backpropagate(score)
                
            iterations += 1

        print(f"Vanilla MCTS Iterations: {iterations} | Time: {time.time() - start_time:.3f}s")
        return self._best_move(root)

    def _select(self, node: MCTSNode) -> MCTSNode:
        while node.state.snakes.get(self.my_id, Snake("x", [], 0, False)).is_alive and node.is_fully_expanded():
            if not node.children: break
            node = node.best_child()
        return node

    def _best_move(self, root: MCTSNode) -> str:
        if not root.children: 
            safe = root.state.get_legal_moves(self.my_id)
            return safe[0] if safe else "up"
            
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action_taken

# ==========================================
# Battlesnake Endpoints
# ==========================================
def info() -> Dict:
    return {"apiversion": "1", "author": "MGAIA_Vanilla_MCTS", "color": "#808080", "head": "default", "tail": "default"}

def start(game_state: Dict): pass
def end(game_state: Dict): pass

def move(game_state: Dict) -> Dict:
    my_id = game_state['you']['id']
    root_state = GameState.from_json(game_state) 
    
    agent = MCTSAgent(my_id=my_id)
    best_move = agent.search(root_state)
    
    print(f"Turn {game_state['turn']} | Vanilla MCTS Decided: {best_move}")
    return {"move": best_move}

if __name__ == "__main__":
    run_server({"info": info, "start": start, "move": move, "end": end})