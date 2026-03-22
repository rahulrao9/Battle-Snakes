import random
import typing
import heapq

def info() -> typing.Dict:
    return {
        "apiversion": "1",
        "author": "MGAIA_Heuristic_Refined",
        "color": "#FFD700",
        "head": "default",
        "tail": "default",
    }

def start(game_state: typing.Dict):
    print(f"GAME START: {game_state['game']['id']}")

def end(game_state: typing.Dict):
    print(f"GAME OVER: {game_state['game']['id']}\n")

# --- Helper Algorithms ---

def manhattan_dist(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def get_neighbors(node, width, height):
    x, y = node
    return [(nx, ny) for nx, ny in [(x, y+1), (x, y-1), (x-1, y), (x+1, y)] 
            if 0 <= nx < width and 0 <= ny < height]

def flood_fill(start_node, obstacles, width, height, max_depth):
    """
    Calculates how much open space is accessible from a given square.
    Returns the number of open squares found, up to max_depth.
    """
    visited = set()
    queue = [start_node]
    count = 0
    
    while queue and count < max_depth:
        current = queue.pop(0)
        if current not in visited and current not in obstacles:
            visited.add(current)
            count += 1
            for neighbor in get_neighbors(current, width, height):
                if neighbor not in visited and neighbor not in obstacles:
                    queue.append(neighbor)
    return count

def a_star(start, goal, obstacles, hazards, width, height):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: manhattan_dist(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            curr = current
            length = 0
            while curr in came_from:
                length += 1
                curr = came_from[curr]
            return length

        for neighbor in get_neighbors(current, width, height):
            if neighbor in obstacles:
                continue
            step_cost = 1
            if neighbor in hazards:
                step_cost += 15 

            tentative_g_score = g_score[current] + step_cost

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + manhattan_dist(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return float('inf')

# --- Core Move Logic ---

def move(game_state: typing.Dict) -> typing.Dict:
    board_width = game_state['board']['width']
    board_height = game_state['board']['height']
    
    my_head = game_state['you']['head']
    my_id = game_state['you']['id']
    my_health = game_state['you']['health']
    my_length = game_state['you']['length']
    
    head_tup = (my_head['x'], my_head['y'])
    hazards_set = {(h['x'], h['y']) for h in game_state['board']['hazards']}
    food_tups = [(f['x'], f['y']) for f in game_state['board']['food']]

    # Build Obstacles intelligently (Tail Chasing Rule)
    obs_set = set()
    risky_head_zones = set()
    kill_head_zones = set()
    
    for snake in game_state['board']['snakes']:
        body = [(s['x'], s['y']) for s in snake['body']]
        
        # If health is 100, they just ate. Tail stays.
        # Otherwise, the tail will move, so we don't count the very last segment as an obstacle.
        if snake['health'] < 100 and len(body) > 1:
            current_obstacles = body[:-1] 
        else:
            current_obstacles = body
            
        for segment in current_obstacles:
            obs_set.add(segment)

        # Head-to-Head calculations
        if snake['id'] != my_id:
            opp_head = (snake['head']['x'], snake['head']['y'])
            opp_length = snake['length']
            for neighbor in get_neighbors(opp_head, board_width, board_height):
                if opp_length >= my_length:
                    risky_head_zones.add(neighbor)
                else:
                    kill_head_zones.add(neighbor)

    move_scores = {"up": 0, "down": 0, "left": 0, "right": 0}
    dirs = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}

    for direction, (dx, dy) in dirs.items():
        target = (head_tup[0] + dx, head_tup[1] + dy)
        
        # 1. Bounds & Static Obstacle Check
        if target[0] < 0 or target[0] >= board_width or target[1] < 0 or target[1] >= board_height:
            move_scores[direction] -= 10000 
            continue
            
        if target in obs_set:
            move_scores[direction] -= 10000 
            continue

        # 2. Dynamic Space Check (Flood Fill)
        # Check if this move leads into a dead end smaller than our own body
        open_space = flood_fill(target, obs_set, board_width, board_height, my_length)
        if open_space < my_length:
            # Trapped! Heavily penalize based on how small the space is
            move_scores[direction] -= (4000 - open_space)

        # 3. Head-to-Head Check
        if target in risky_head_zones:
            move_scores[direction] -= 5000 
        elif target in kill_head_zones:
            move_scores[direction] += 150 

        # 4. Hazards Evaluation (Lethality check)
        if target in hazards_set:
            # Assignment specific: hazards do 14 damage + 1 per turn. 
            # If we step in at 15 health or lower, we die.
            if my_health <= 15:
                move_scores[direction] -= 10000
            else:
                move_scores[direction] -= 50 

        # 5. Food Evaluation
        if food_tups:
            min_path_len = float('inf')
            for food in food_tups:
                path_len = a_star(target, food, obs_set, hazards_set, board_width, board_height)
                if path_len < min_path_len:
                    min_path_len = path_len
            
            # Exponentially increase food priority as health gets lower
            if my_health < 20:
                food_weight = 500
            elif my_health < 50:
                food_weight = 100
            else:
                food_weight = 10
            
            if min_path_len == 0:
                move_scores[direction] += food_weight 
            elif min_path_len != float('inf'):
                move_scores[direction] += (food_weight / min_path_len)

    # Break ties
    for direction in move_scores:
        if move_scores[direction] > -5000:
            move_scores[direction] += random.uniform(0, 0.1)

    best_move = max(move_scores, key=move_scores.get)
    print(f"MOVE {game_state['turn']}: {best_move} (Scores: {move_scores})")
    
    return {"move": best_move}

if __name__ == "__main__":
    from server import run_server
    run_server({"info": info, "start": start, "move": move, "end": end})