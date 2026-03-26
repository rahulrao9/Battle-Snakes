
import random
from typing import Dict, List, Set, Tuple
from collections import deque

MOVES = {"up": (0,1), "down": (0,-1), "left": (-1,0), "right": (1,0)}


class Snake:
    __slots__ = ["id","body","health","is_alive"]
    def __init__(self, sid, body, health, is_alive=True):
        self.id = sid; self.body = body; self.health = health; self.is_alive = is_alive
    @property
    def head(self): return self.body[0]
    @property
    def length(self): return len(self.body)
    def clone(self):
        return Snake(self.id, deque(self.body), self.health, self.is_alive)


class GameState:
    __slots__ = ["board_width","board_height","turn","snakes","food","hazards"]
    def __init__(self, w, h, turn, snakes, food, hazards):
        self.board_width=w; self.board_height=h; self.turn=turn
        self.snakes=snakes; self.food=food; self.hazards=hazards
    @classmethod
    def from_json(cls, g):
        w=g["board"]["width"]; h=g["board"]["height"]
        food={(f["x"],f["y"]) for f in g["board"]["food"]}
        haz={(z["x"],z["y"]) for z in g["board"]["hazards"]}
        snakes={}
        for s in g["board"]["snakes"]:
            body=deque((p["x"],p["y"]) for p in s["body"])
            snakes[s["id"]]=Snake(s["id"],body,s["health"])
        return cls(w,h,g["turn"],snakes,food,haz)
    def clone(self):
        return GameState(self.board_width,self.board_height,self.turn,
                         {k:v.clone() for k,v in self.snakes.items()},
                         set(self.food), self.hazards)
    def get_action_space(self, sid):
        snake=self.snakes.get(sid)
        if not snake or not snake.is_alive: return []
        obs=set()
        for s in self.snakes.values():
            if not s.is_alive: continue
            bl=list(s.body)
            obs.update(bl[:-1] if s.health<100 and len(bl)>1 else bl)
        danger=set()
        for o in self.snakes.values():
            if o.id!=sid and o.is_alive and o.length>=snake.length:
                for dx,dy in MOVES.values():
                    danger.add((o.head[0]+dx,o.head[1]+dy))
        hstack=min(4,max(0,(self.turn-1)//25)) if self.turn>=26 else 0
        lethal=1+(14*hstack)
        safe,risky,lethal_m=[],[],[]
        for m,(dx,dy) in MOVES.items():
            nx,ny=snake.head[0]+dx,snake.head[1]+dy
            if 0<=nx<self.board_width and 0<=ny<self.board_height:
                t=(nx,ny)
                if t not in obs:
                    if t in self.hazards and snake.health<=lethal: lethal_m.append(m)
                    elif t in danger: risky.append(m)
                    else: safe.append(m)
        return safe or risky or lethal_m or ["up"]
    def get_guided_move(self, sid):
        moves=self.get_action_space(sid)
        if not moves: return "up"
        snake=self.snakes[sid]; head=snake.head
        if (snake.health<50 or snake.length<10) and self.food:
            best=moves[0]; bd=float("inf")
            for m in moves:
                nx,ny=head[0]+MOVES[m][0],head[1]+MOVES[m][1]
                d=min(abs(nx-fx)+abs(ny-fy) for fx,fy in self.food)
                if d<bd: bd=d; best=m
            return best
        return random.choice(moves)
    def step(self, joint):
        ns=self.clone(); ns.turn+=1
        hstack=min(4,max(0,(ns.turn-1)//25)) if ns.turn>=26 else 0
        hdmg=14*hstack; eaten=set()
        for sid,action in joint.items():
            s=ns.snakes.get(sid)
            if not s or not s.is_alive: continue
            dx,dy=MOVES[action]; nh=(s.head[0]+dx,s.head[1]+dy)
            if nh in ns.food: eaten.add(nh); s.health=100
            else: s.health-=1; s.body.pop()
            if nh in ns.hazards: s.health-=(1+hdmg)
            s.body.appendleft(nh)
        ns.food-=eaten
        bc={}
        for s in ns.snakes.values():
            if s.is_alive:
                for pt in s.body: bc[pt]=bc.get(pt,0)+1
        dead=set()
        for sid,s in ns.snakes.items():
            if not s.is_alive: continue
            h=s.head
            if not(0<=h[0]<ns.board_width and 0<=h[1]<ns.board_height) or s.health<=0:
                dead.add(sid); continue
            if bc.get(h,0)>1:
                col=[x for x in ns.snakes.values() if x.is_alive and x.head==h]
                if len(col)>1:
                    ml=max(x.length for x in col)
                    lng=[x for x in col if x.length==ml]
                    if len(lng)>1 or s not in lng: dead.add(sid)
                else: dead.add(sid)
        for sid in dead: ns.snakes[sid].is_alive=False
        return ns