# Welcome to
# __________         __    __  .__                               __
# \______   \_____ _/  |__/  |_|  |   ____   ______ ____ _____  |  | __ ____
#  |    |  _/\__  \\   __\   __\  | _/ __ \ /  ___//    \\__  \ |  |/ // __ \
#  |    |   \ / __ \|  |  |  | |  |_\  ___/ \___ \|   |  \/ __ \|    <\  ___/
#  |________/(______/__|  |__| |____/\_____>______>___|__(______/__|__\\_____>
# This file contains a standalone heuristic agent that can be used for Phase 1 of the Battlesnake competition.
# It includes a Voronoi-based evaluation function that can be used as a Progressive Bias in MCTS for Phase 2/3.
# The move function implements a simple heuristic that evaluates the immediate next move using the same Voronoi logic.
import random
import typing
from collections import deque

# ==========================================
# Battlesnake API Endpoints
# ==========================================

def info() -> typing.Dict:
    return {
        "apiversion": "1",
        "author": "Heuristic_Agent",
        "color": "#FFD700",
        "head": "default",
        "tail": "default",
    }

def start(game_state: typing.Dict):
    print(f"GAME START: {game_state['game']['id']}")

def end(game_state: typing.Dict):
    print(f"GAME OVER: {game_state['game']['id']}\n")

def get_neighbors(x, y, width, height):
    return [(nx, ny) for nx, ny in [(x, y+1), (x, y-1), (x-1, y), (x+1, y)] 
            if 0 <= nx < width and 0 <= ny < height]

# ==========================================
# MCTS Evaluation Module (For Phase 2/3)
# ==========================================

def evaluate_state(state) -> typing.Dict[str, float]:
    """
    Evaluates the fast MCTS GameState using the Voronoi logic.
    Returns a dictionary of normalized scores [0.0, 1.0] for MCTS Progressive Bias.
    """
    scores = {}
    board_width = state.board_width
    board_height = state.board_height
    turn = state.turn
    hazards_set = state.hazards
    food_set = state.food

    # Dynamic Hazard Math
    hazard_stack = 0
    if turn >= 26:
        hazard_stack = min(4, (turn - 1) // 25)
    hazard_damage = 14 * hazard_stack
    total_hazard_cost = 1 + hazard_damage

    # Pre-calculate obstacle lifetimes globally for this state
    obstacle_lifetimes = {}
    for s_id, snake in state.snakes.items():
        if not snake.is_alive: continue
        is_fed = snake.health == 100
        for i, pt in enumerate(reversed(snake.body)):
            lifetime = i + (1 if is_fed else 0)
            obstacle_lifetimes[pt] = max(obstacle_lifetimes.get(pt, 0), lifetime)

    # Evaluate the board from the perspective of EVERY alive snake
    for eval_id, me in state.snakes.items():
        if not me.is_alive:
            scores[eval_id] = 0.0
            continue
            
        my_head = me.head
        my_health = me.health
        my_length = me.length
        
        risky_head_zones = set()
        kill_head_zones = set()

        for s_id, snake in state.snakes.items():
            if not snake.is_alive or s_id == eval_id: continue
            opp_head = snake.head
            for nx, ny in get_neighbors(opp_head[0], opp_head[1], board_width, board_height):
                if snake.length >= my_length:
                    risky_head_zones.add((nx, ny))
                else:
                    kill_head_zones.add((nx, ny))

        # BFS Setup
        queue = deque()
        visited = {}
        
        # Add enemies to queue first (distance 0)
        for s_id, snake in state.snakes.items():
            if not snake.is_alive or s_id == eval_id: continue
            queue.append((snake.head[0], snake.head[1], s_id, 0))
            visited[snake.head] = s_id

        # Add the evaluating snake second (distance 1)
        queue.append((my_head[0], my_head[1], eval_id, 1))
        if my_head not in visited: 
            visited[my_head] = eval_id

        territory_count = 0
        closest_food_dist = 1 if my_head in food_set else float('inf')

        # Run Voronoi Expansion
        while queue:
            cx, cy, owner_id, dist = queue.popleft()
            for nx, ny in get_neighbors(cx, cy, board_width, board_height):
                neighbor = (nx, ny)
                
                # Time-Aware Obstacle clear time
                if neighbor in obstacle_lifetimes and dist <= obstacle_lifetimes[neighbor]:
                    continue
                    
                # Hazard-blindness prevention
                if neighbor in hazards_set and my_health <= total_hazard_cost * 3:
                    continue
                    
                if neighbor not in visited:
                    visited[neighbor] = owner_id
                    queue.append((nx, ny, owner_id, dist + 1))
                    if owner_id == eval_id:
                        territory_count += 1
                        if neighbor in food_set and closest_food_dist == float('inf'):
                            closest_food_dist = dist + 1

        # Calculate final raw score
        raw_score = (territory_count * 3)
        if closest_food_dist != float('inf'):
            raw_score += ((200 if my_health < 40 else 20) / closest_food_dist)
        if territory_count < my_length:
            raw_score -= (4000 - territory_count)
        if my_head in risky_head_zones:
            raw_score -= 5000
            
        # Normalize for UCB equation (assuming typical bounds of -5000 to +1500)
        scores[eval_id] = max(0.0, min(1.0, (raw_score + 5000) / 6500.0))

    return scores

# ==========================================
# Standalone Agent Move Logic (For Phase 1)
# ==========================================

def move(game_state: typing.Dict) -> typing.Dict:
    """
    Parses standard JSON to calculate the best immediate move.
    """
    board_width = game_state['board']['width']
    board_height = game_state['board']['height']
    turn = game_state['turn']
    
    me = game_state['you']
    my_id = me['id']
    my_head = (me['head']['x'], me['head']['y'])
    my_health = me['health']
    my_length = me['length']
    
    snakes = game_state['board']['snakes']
    hazards_set = {(h['x'], h['y']) for h in game_state['board']['hazards']}
    food_set = {(f['x'], f['y']) for f in game_state['board']['food']}

    hazard_stack = 0
    if turn >= 26:
        hazard_stack = min(4, (turn - 1) // 25)
    hazard_damage = 14 * hazard_stack
    total_hazard_cost = 1 + hazard_damage

    obstacle_lifetimes = {}
    risky_head_zones = set()
    kill_head_zones = set()

    for snake in snakes:
        s_id = snake['id']
        body = [(s['x'], s['y']) for s in snake['body']]
        is_fed = snake['health'] == 100
        
        for i, segment in enumerate(reversed(body)):
            lifetime = i + (1 if is_fed else 0)
            obstacle_lifetimes[segment] = max(obstacle_lifetimes.get(segment, 0), lifetime)

        if s_id != my_id:
            opp_head = (snake['head']['x'], snake['head']['y'])
            for nx, ny in get_neighbors(opp_head[0], opp_head[1], board_width, board_height):
                if snake['length'] >= my_length:
                    risky_head_zones.add((nx, ny))
                else:
                    kill_head_zones.add((nx, ny))

    def evaluate_state(target_sq):
        if target_sq[0] < 0 or target_sq[0] >= board_width or target_sq[1] < 0 or target_sq[1] >= board_height:
            return -10000
        if target_sq in obstacle_lifetimes and obstacle_lifetimes[target_sq] >= 1: 
            return -10000

        score = 0
        if target_sq in hazards_set:
            if my_health <= total_hazard_cost:
                return -10000
            score -= (50 + hazard_damage * 2) 

        if target_sq in risky_head_zones:
            score -= 5000
        elif target_sq in kill_head_zones:
            score += 150

        queue = deque()
        visited = {}
        
        for snake in snakes:
            if snake['id'] != my_id:
                o_head = (snake['head']['x'], snake['head']['y'])
                queue.append((o_head[0], o_head[1], snake['id'], 0))
                visited[o_head] = snake['id']

        queue.append((target_sq[0], target_sq[1], my_id, 1))
        if target_sq not in visited:
            visited[target_sq] = my_id

        territory_count = 0
        closest_food_dist = 1 if target_sq in food_set else float('inf')

        while queue:
            cx, cy, owner_id, dist = queue.popleft()
            
            for nx, ny in get_neighbors(cx, cy, board_width, board_height):
                neighbor = (nx, ny)
                if neighbor in obstacle_lifetimes and dist <= obstacle_lifetimes[neighbor]:
                    continue 
                if neighbor in hazards_set and my_health <= total_hazard_cost * 3:
                    continue 

                if neighbor not in visited:
                    visited[neighbor] = owner_id
                    queue.append((nx, ny, owner_id, dist + 1))
                    
                    if owner_id == my_id:
                        territory_count += 1
                        if neighbor in food_set and closest_food_dist == float('inf'):
                            closest_food_dist = dist + 1

        score += (territory_count * 3)
        if closest_food_dist != float('inf'):
            score += ((200 if my_health < 40 else 20) / closest_food_dist)
        if territory_count < my_length:
            score -= (4000 - territory_count)

        return score

    possible_moves = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}
    move_scores = {}

    for move_name, (dx, dy) in possible_moves.items():
        target = (my_head[0] + dx, my_head[1] + dy)
        move_scores[move_name] = evaluate_state(target)

    for m in move_scores:
        if move_scores[m] > -5000:
            move_scores[m] += random.uniform(0, 0.1)

    best_move = max(move_scores, key=move_scores.get)
    print(f"Heuristic Turn {turn} | MOVE: {best_move} | Health: {my_health} | Scores: {move_scores}")
    
    return {"move": best_move}

# ==========================================
# Server Execution
# ==========================================

if __name__ == "__main__":
    from server import run_server
    run_server({"info": info, "start": start, "move": move, "end": end})