import csv
import os
import time
from typing import Dict, Any, List, Tuple


class BattlesnakeDirectLogger:
    def __init__(self, out_dir: str = "logs"):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.current_game_id = None
        self.turn_csv_path = None
        self.summary_csv_path = os.path.join(self.out_dir, "game_summaries.csv")

        
        self.last_seen: Dict[str, Dict[str, Any]] = {}
        self.logged_turns: set[Tuple[str, int]] = set()
        self.first_seen_turn: Dict[str, int] = {}
        self.last_alive_turn: Dict[str, int] = {}
        self.start_length: Dict[str, int] = {}
        self.max_length: Dict[str, int] = {}
        self.last_health: Dict[str, int] = {}
        self.last_length: Dict[str, int] = {}

        self._ensure_summary_file()

    def _ensure_summary_file(self) -> None:
        if not os.path.exists(self.summary_csv_path):
            with open(self.summary_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "game_id",
                        "winner",
                        "placement",
                        "snake_id",
                        "snake_name",
                        "turns_survived",
                        "start_length",
                        "max_length",
                        "final_length",
                        "final_health",
                        "alive_at_end",
                        "survival_score",
                        "growth_score",
                        "performance_score",
                    ],
                )
                writer.writeheader()

    def start_game(self, game_state: Dict[str, Any]) -> None:
        game_id = game_state["game"]["id"]
        self.current_game_id = game_id

        ts = time.strftime("%Y%m%d-%H%M%S")
        self.turn_csv_path = os.path.join(self.out_dir, f"{ts}_{game_id}_turns.csv")

        with open(self.turn_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "game_id",
                    "turn",
                    "snake_id",
                    "snake_name",
                    "health",
                    "length",
                    "alive",
                ],
            )
            writer.writeheader()

        self.last_seen = {}
        self.logged_turns = set()
        self.first_seen_turn = {}
        self.last_alive_turn = {}
        self.start_length = {}
        self.max_length = {}
        self.last_health = {}
        self.last_length = {}

    def log_turn(self, game_state: Dict[str, Any]) -> None:
        game_id = game_state["game"]["id"]
        turn = int(game_state["turn"])

        if self.current_game_id != game_id or self.turn_csv_path is None:
            self.start_game(game_state)

        if (game_id, turn) in self.logged_turns:
            return
        self.logged_turns.add((game_id, turn))

        snakes: List[Dict[str, Any]] = game_state["board"]["snakes"]
        current_ids = set()
        rows = []

        for snake in snakes:
            sid = snake["id"]
            sname = snake.get("name", sid)
            health = int(snake.get("health", 0))
            length = int(snake.get("length", len(snake.get("body", []))))

            current_ids.add(sid)

            if sid not in self.first_seen_turn:
                self.first_seen_turn[sid] = turn
                self.start_length[sid] = length

            self.last_alive_turn[sid] = turn
            self.max_length[sid] = max(self.max_length.get(sid, 0), length)
            self.last_health[sid] = health
            self.last_length[sid] = length

            self.last_seen[sid] = {
                "snake_name": sname,
                "health": health,
                "length": length,
            }

            rows.append(
                {
                    "game_id": game_id,
                    "turn": turn,
                    "snake_id": sid,
                    "snake_name": sname,
                    "health": health,
                    "length": length,
                    "alive": 1,
                }
            )

        for sid, old_data in self.last_seen.items():
            if sid not in current_ids:
                rows.append(
                    {
                        "game_id": game_id,
                        "turn": turn,
                        "snake_id": sid,
                        "snake_name": old_data["snake_name"],
                        "health": old_data["health"],
                        "length": old_data["length"],
                        "alive": 0,
                    }
                )
        for row in rows:
                print(
                    f"turn={row['turn']} "
                    f"name={row['snake_name']} "
                    f"length={row['length']} "
                    f"health={row['health']} "
                    f"alive={row['alive']}"
                )

        self._append_turn_rows(rows)

    def end_game(self, game_state: Dict[str, Any]) -> None:
        self.log_turn(game_state)
        self._write_summary(game_state)

    def _append_turn_rows(self, rows: List[Dict[str, Any]]) -> None:
        with open(self.turn_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "game_id",
                    "turn",
                    "snake_id",
                    "snake_name",
                    "health",
                    "length",
                    "alive",
                ],
            )
            writer.writerows(rows)

    def _write_summary(self, game_state: Dict[str, Any]) -> None:
        game_id = game_state["game"]["id"]
        final_turn = int(game_state["turn"])
        total_game_turns = final_turn + 1

        all_ids = list(self.last_seen.keys())
        alive_at_end = {snake["id"] for snake in game_state["board"]["snakes"]}

        winner_name = ""
        if len(alive_at_end) == 1:
            winner_id = next(iter(alive_at_end))
            winner_name = self.last_seen[winner_id]["snake_name"]

        
        ranked_ids = sorted(
            all_ids,
            key=lambda sid: (
                -(1 if sid in alive_at_end else 0),
                -self.last_alive_turn.get(sid, -1),
                -self.max_length.get(sid, 0),
                -self.last_health.get(sid, 0),
                self.last_seen[sid]["snake_name"],
            ),
        )

        summary_rows = []
        for placement, sid in enumerate(ranked_ids, start=1):
            turns_survived = self.last_alive_turn.get(sid, -1) - self.first_seen_turn.get(sid, 0) + 1
            start_len = self.start_length.get(sid, 0)
            final_len = self.last_length.get(sid, 0)

            survival_score = turns_survived / total_game_turns if total_game_turns > 0 else 0.0
            growth_score = min(1.0, max(0, final_len - start_len) / 5.0)
            performance_score = 0.8 * survival_score + 0.2 * growth_score

            summary_rows.append(
                {
                    "game_id": game_id,
                    "winner": winner_name,
                    "placement": placement,
                    "snake_id": sid,
                    "snake_name": self.last_seen[sid]["snake_name"],
                    "turns_survived": turns_survived,
                    "start_length": start_len,
                    "max_length": self.max_length.get(sid, 0),
                    "final_length": final_len,
                    "final_health": self.last_health.get(sid, 0),
                    "alive_at_end": 1 if sid in alive_at_end else 0,
                    "survival_score": round(survival_score, 4),
                    "growth_score": round(growth_score, 4),
                    "performance_score": round(performance_score, 4),
                }
            )

        with open(self.summary_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "game_id",
                    "winner",
                    "placement",
                    "snake_id",
                    "snake_name",
                    "turns_survived",
                    "start_length",
                    "max_length",
                    "final_length",
                    "final_health",
                    "alive_at_end",
                    "survival_score",
                    "growth_score",
                    "performance_score",
                ],
            )
            writer.writerows(summary_rows)

        print(f"\nGame {game_id} summary")
        for row in summary_rows:
            print(
                f"place={row['placement']} "
                f"name={row['snake_name']} "
                f"winner={row['winner']} "
                f"turns_survived={row['turns_survived']} "
                f"final_length={row['final_length']} "
                f"score={row['performance_score']} "
                f"alive_at_end={row['alive_at_end']}"
            )