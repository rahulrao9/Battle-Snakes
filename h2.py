import random
import typing
from collections import deque

def info() -> typing.Dict:
    return {
        "apiversion": "1",
        "author": "MGAIA_Voronoi_Perfected",
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

def move(game_state: typing.Dict) -> typing.Dict:
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

    # 1. Dynamic Hazard Math
    hazard_stack = 0
    if turn >= 26:
        hazard_stack = min(4, (turn - 1) // 25)
    hazard_damage = 14 * hazard_stack
    total_hazard_cost = 1 + hazard_damage

    # 2. Time-Aware Obstacles & Head-to-Head Zones
    obstacle_lifetimes = {}
    risky_head_zones = set()
    kill_head_zones = set()

    for snake in snakes:
        s_id = snake['id']
        body = [(s['x'], s['y']) for s in snake['body']]
        is_fed = snake['health'] == 100
        
        # Flawless Time-Aware obstacle calculation (Fixed Ghost Head Bug)
        for i, segment in enumerate(reversed(body)):
            lifetime = i + (1 if is_fed else 0)
            if segment in obstacle_lifetimes:
                obstacle_lifetimes[segment] = max(obstacle_lifetimes[segment], lifetime)
            else:
                obstacle_lifetimes[segment] = lifetime

        # Head-to-Head projections
        if s_id != my_id:
            opp_head = (snake['head']['x'], snake['head']['y'])
            for nx, ny in get_neighbors(opp_head[0], opp_head[1], board_width, board_height):
                if snake['length'] >= my_length:
                    risky_head_zones.add((nx, ny))
                else:
                    kill_head_zones.add((nx, ny))

    # --- Voronoi State Evaluator ---
    def evaluate_move(target_sq):
        # Fatal static checks
        if target_sq[0] < 0 or target_sq[0] >= board_width or target_sq[1] < 0 or target_sq[1] >= board_height:
            return -10000
        if target_sq in obstacle_lifetimes and obstacle_lifetimes[target_sq] >= 1: 
            return -10000

        score = 0
        
        # Hazard lethality
        if target_sq in hazards_set:
            if my_health <= total_hazard_cost:
                return -10000
            score -= (50 + hazard_damage * 2) 

        # Head-to-Head modifiers
        if target_sq in risky_head_zones:
            score -= 5000
        elif target_sq in kill_head_zones:
            score += 150

        # Multi-Source BFS Setup
        queue = deque()
        visited = {}
        
        # Fix #1: Add enemies FIRST (dist 0) so the queue maintains strict monotonic distance
        for snake in snakes:
            if snake['id'] != my_id:
                o_head = (snake['head']['x'], snake['head']['y'])
                queue.append((o_head[0], o_head[1], snake['id'], 0))
                visited[o_head] = snake['id']

        # Fix #1: Add our move SECOND (dist 1)
        queue.append((target_sq[0], target_sq[1], my_id, 1))
        if target_sq not in visited: # Don't overwrite if we are intentionally crashing a head
            visited[target_sq] = my_id

        territory_count = 0
        
        # Fix #3: Explicitly check if our target square is food to avoid division by zero later
        closest_food_dist = 1 if target_sq in food_set else float('inf')

        # Run Voronoi Expansion
        while queue:
            cx, cy, owner_id, dist = queue.popleft()
            
            for nx, ny in get_neighbors(cx, cy, board_width, board_height):
                neighbor = (nx, ny)
                
                # Check Time-Aware Obstacle clear time
                if neighbor in obstacle_lifetimes:
                    if dist <= obstacle_lifetimes[neighbor]:
                        continue # Square is still occupied when we arrive

                # Fix #4: Hazard-blindness prevention. Don't claim hazard territory if weak.
                if neighbor in hazards_set and my_health <= total_hazard_cost * 3:
                    continue 

                if neighbor not in visited:
                    visited[neighbor] = owner_id
                    queue.append((nx, ny, owner_id, dist + 1))
                    
                    if owner_id == my_id:
                        territory_count += 1
                        if neighbor in food_set and closest_food_dist == float('inf'):
                            closest_food_dist = dist + 1

        # Reward Board Control (Voronoi)
        score += (territory_count * 3)

        # Reward Uncontested Food safely
        if closest_food_dist != float('inf'):
            food_weight = 200 if my_health < 40 else 20
            score += (food_weight / closest_food_dist)

        # Space panic: If our controlled territory is smaller than our body, panic!
        if territory_count < my_length:
            score -= (4000 - territory_count)

        return score

    # --- Move Selection ---
    possible_moves = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}
    move_scores = {}

    for move_name, (dx, dy) in possible_moves.items():
        target = (my_head[0] + dx, my_head[1] + dy)
        move_scores[move_name] = evaluate_move(target)

    # Break ties safely
    for m in move_scores:
        if move_scores[m] > -5000:
            move_scores[m] += random.uniform(0, 0.1)

    best_move = max(move_scores, key=move_scores.get)
    print(f"Turn {turn} | MOVE: {best_move} | Health: {my_health} | Scores: {move_scores}")
    
    return {"move": best_move}

if __name__ == "__main__":
    from server import run_server
    run_server({"info": info, "start": start, "move": move, "end": end})