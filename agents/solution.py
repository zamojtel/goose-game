import json
import time

from openai import OpenAI
from agents.base import ChatCallback, GooseAgent, GooseAgentMessage, GooseAgentResult, PlannerAgent
from goose_game.environment import GooseEnvironment, PlannerEnvironment


class GooseAgentImpl(GooseAgent):
    def __init__(self, client: OpenAI, used_model: str, env: GooseEnvironment, append_to_chat: ChatCallback) -> None:
        super().__init__(client, used_model, env, append_to_chat)
        self._append_to_chat(f"Initialized {env.goose_id}.")

    def on_call(self, message: GooseAgentMessage) -> GooseAgentResult:
        text = message.description.lower().strip()

        if text == "observe":
            return GooseAgentResult(output=self._env.describe_state())

        self._append_to_chat(f"Planner message: {message.description}")

        if "up" in text:
            action, arg = "move", "up"
        elif "down" in text:
            action, arg = "move", "down"
        elif "left" in text:
            action, arg = "move", "left"
        elif "right" in text:
            action, arg = "move", "right"
        elif "honk" in text:
            action, arg = "honk", 1
        else:
            return GooseAgentResult(error=f"Error parsing message: {message.description}")

        if action == "move":
            success = self._env.move(arg)
            if success:
                result = f"Moved {arg} -> SUCCESS"
            else:
                result = f"Moved {arg} -> FAILED (Blocked)"
        else:
            event = self._env.honk(arg)
            result = f"Honked -> SUCCESS"

        return GooseAgentResult(output=result)


class PlannerAgentImpl(PlannerAgent):
    def __init__(self, client: OpenAI, used_model: str, env: PlannerEnvironment, agents: dict[str, GooseAgent],
                 append_to_chat: ChatCallback):
        super().__init__(client, used_model, env, agents, append_to_chat)
        self._append_to_chat(f"Initialized planner for level: {env.level_name}.")
        self.history = []

        # OSTATECZNA PAMIĘĆ AGENTA - Śledzenie odwiedzonych ścieżek
        self.last_thought = {
            "visited_1": [], "visited_2": [],
            "dud_buttons": [],
            "g1_last_cmd": None, "g2_last_cmd": None,
            "blocked_1": False, "blocked_2": False
        }

        self.known_buttons = set()
        self.known_doors = set()
        self.known_goals = set()

    def step(self) -> None:
        self._append_to_chat("Planner step executed.")

        obs_1 = self._agents["goose_1"].on_call(GooseAgentMessage(description="observe")).output
        obs_2 = self._agents["goose_2"].on_call(GooseAgentMessage(description="observe")).output

        pos_1 = pos_2 = None
        grid_map = {}
        open_doors = set()
        closed_doors = set()

        # 1. PARSOWANIE MAPY DO PAMIĘCI ABSOLUTNEJ
        for obs in [obs_1, obs_2]:
            lines = obs.strip().split('\n')
            for r, line in enumerate(lines):
                for c, char in enumerate(line):
                    if char != '?':
                        if char == 'X':
                            pos_1 = (r, c)
                        elif char == 'Y':
                            pos_2 = (r, c)
                        elif char == '@':
                            self.known_buttons.add((r, c))
                        elif char == '*':
                            self.known_goals.add((r, c))
                        elif char == '/':
                            self.known_doors.add((r, c))
                            open_doors.add((r, c))
                        elif char == '$':
                            self.known_doors.add((r, c))
                            closed_doors.add((r, c))
                        grid_map[(r, c)] = char

        # 2. NAPRAWA OKLUZJI (Przywracanie znaków spod kaczek)
        if pos_1 and pos_1 in self.known_doors: open_doors.add(pos_1); grid_map[pos_1] = '/'
        if pos_2 and pos_2 in self.known_doors: open_doors.add(pos_2); grid_map[pos_2] = '/'
        if pos_1 in closed_doors: closed_doors.remove(pos_1)
        if pos_2 in closed_doors: closed_doors.remove(pos_2)
        if pos_1 and pos_1 in self.known_buttons: grid_map[pos_1] = '@'
        if pos_2 and pos_2 in self.known_buttons: grid_map[pos_2] = '@'
        if pos_1 and pos_1 in self.known_goals: grid_map[pos_1] = '*'
        if pos_2 and pos_2 in self.known_goals: grid_map[pos_2] = '*'

        # 3. SILNIK EKSPLORACJI (DFS)
        visited_1 = self.last_thought.get("visited_1", [])
        visited_2 = self.last_thought.get("visited_2", [])

        if pos_1 and list(pos_1) not in visited_1: visited_1.append(list(pos_1))
        if pos_2 and list(pos_2) not in visited_2: visited_2.append(list(pos_2))

        def get_moves(pos, visited):
            if not pos: return []
            r, c = pos
            valid = []
            for d, dr, dc in [("up", -1, 0), ("down", 1, 0), ("left", 0, -1), ("right", 0, 1)]:
                nr, nc = r + dr, c + dc
                if grid_map.get((nr, nc), '#') not in ['#', '$']:
                    if [nr, nc] in visited:
                        valid.append(f"{d} (visited)")
                    else:
                        valid.append(d)
            return valid

        moves_1 = get_moves(pos_1, visited_1)
        moves_2 = get_moves(pos_2, visited_2)

        # 4. DETEKTOR ŚCIAN I DRZWI
        def is_blocked(pos, moves):
            if not pos: return False
            if any("(visited)" not in m for m in moves): return False
            r, c = pos
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if grid_map.get((r + dr, c + dc)) == '$': return True
            return False

        blocked_1 = is_blocked(pos_1, moves_1)
        blocked_2 = is_blocked(pos_2, moves_2)

        was_blocked_1 = self.last_thought.get("blocked_1", False)
        was_blocked_2 = self.last_thought.get("blocked_2", False)
        duds = self.last_thought.get("dud_buttons", [])

        # MAGICZNY RESET: Gdy nowa gęś utknie pod nowymi drzwiami, czyścimy historię guzików dla drugiej gęsi!
        if blocked_1 and not was_blocked_1:
            duds = []
            visited_2 = [list(pos_2)] if pos_2 else []
        if blocked_2 and not was_blocked_2:
            duds = []
            visited_1 = [list(pos_1)] if pos_1 else []

        on_btn_1 = pos_1 in self.known_buttons
        on_btn_2 = pos_2 in self.known_buttons
        g1_last_cmd = self.last_thought.get("g1_last_cmd")
        g2_last_cmd = self.last_thought.get("g2_last_cmd")

        # AUTO-DETEKCJA FAŁSZYWYCH GUZIKÓW
        if on_btn_1 and g1_last_cmd == "honk" and not open_doors and list(pos_1) not in duds:
            duds.append(list(pos_1))
        if on_btn_2 and g2_last_cmd == "honk" and not open_doors and list(pos_2) not in duds:
            duds.append(list(pos_2))

        on_dud_1 = list(pos_1) in duds if pos_1 else False
        on_dud_2 = list(pos_2) in duds if pos_2 else False
        at_goal_1 = pos_1 in self.known_goals
        at_goal_2 = pos_2 in self.known_goals

        facts = "--- GAME ENGINE FACTS ---\n"
        facts += f"Goose 1 | Blocked by door? {'YES' if blocked_1 else 'NO'} | On button? {'YES' if on_btn_1 else 'NO'} (Is DUD? {'YES' if on_dud_1 else 'NO'}) | At goal? {'YES' if at_goal_1 else 'NO'} | Moves: {moves_1}\n"
        facts += f"Goose 2 | Blocked by door? {'YES' if blocked_2 else 'NO'} | On button? {'YES' if on_btn_2 else 'NO'} (Is DUD? {'YES' if on_dud_2 else 'NO'}) | At goal? {'YES' if at_goal_2 else 'NO'} | Moves: {moves_2}\n"
        facts += f"Any open doors? {'YES' if open_doors else 'NO'}\n"

        messages = [
            {
                "role": "system",
                "content": """You are the Planner Agent coordinating Goose 1 and Goose 2.
Read the GAME ENGINE FACTS and apply these strict rules to choose the commands.

RULES (Apply in order from 1 to 6):
1. AT GOAL: If a goose is 'At goal? YES', it MUST output 'honk'.
2. BLOCKED: If a goose is 'Blocked by door? YES', it MUST output 'honk'. (Wait for the other goose to open it).
3. HOLDING BUTTON: If 'On button? YES', AND 'Any open doors? YES', AND the other goose is NOT 'Blocked by door? YES', YOU MUST 'honk' to keep the door open!
4. TESTING BUTTON: If 'On button? YES', AND 'Is DUD? NO', AND 'Any open doors? NO', output 'honk' to test it.
5. USELESS BUTTON: If 'On button? YES', but it 'Is DUD? YES', OR the other goose IS 'Blocked by door? YES' (meaning they reached a new closed door), you MUST LEAVE the button (choose a move from Moves).
6. EXPLORING: If none of the above apply, choose a move from 'Moves'. 
   - ALWAYS prefer moves WITHOUT '(visited)'. 
   - Only choose a '(visited)' move if ALL available moves are visited (dead end backtrack).

OUTPUT FORMAT (Strict JSON):
{
  "thought": {
    "g1_logic": "Rule applied for Goose 1",
    "g2_logic": "Rule applied for Goose 2"
  },
  "goose_1": "command (up, down, left, right, or honk)",
  "goose_2": "command (up, down, left, right, or honk)"
}

NOTE: Output ONLY the pure direction word for commands (e.g., 'left', not 'left (visited)').
"""
            },
            {
                "role": "user",
                "content": facts
            }
        ]

        time.sleep(6)

        try:
            response = self._client.chat.completions.create(model=self._used_model, messages=messages)
            clean_text = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
            start = clean_text.find("{")
            end = clean_text.rfind("}")
            if start != -1 and end != -1: clean_text = clean_text[start:end + 1]
            parsed_data = json.loads(clean_text)

            g1_cmd = parsed_data.get("goose_1", "honk").lower().split()[0]
            g2_cmd = parsed_data.get("goose_2", "honk").lower().split()[0]

            if g1_cmd not in ["up", "down", "left", "right", "honk"]: g1_cmd = "honk"
            if g2_cmd not in ["up", "down", "left", "right", "honk"]: g2_cmd = "honk"

            self.last_thought["visited_1"] = visited_1
            self.last_thought["visited_2"] = visited_2
            self.last_thought["dud_buttons"] = duds
            self.last_thought["blocked_1"] = blocked_1
            self.last_thought["blocked_2"] = blocked_2
            self.last_thought["g1_last_cmd"] = g1_cmd
            self.last_thought["g2_last_cmd"] = g2_cmd

        except Exception as e:
            self._append_to_chat(f"API Error: {str(e)}")
            g1_cmd, g2_cmd = "honk", "honk"

        turn_logs = []
        for goose_id, cmd in [("goose_1", g1_cmd), ("goose_2", g2_cmd)]:
            self._append_to_chat(f"Calling {goose_id}.")
            result = self._agents[goose_id].on_call(GooseAgentMessage(description=cmd))

            if result.error is not None:
                self._append_to_chat(f"{goose_id} error: {result.error}")
                turn_logs.append(f"{goose_id}: ERROR")
            else:
                self._append_to_chat(f"{goose_id} result: {result.output}")
                turn_logs.append(f"{goose_id}: {cmd}")

        self.history.append(" | ".join(turn_logs))