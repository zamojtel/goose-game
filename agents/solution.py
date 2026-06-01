import json
import re
import time
from collections import deque

from openai import OpenAI

from agents.base import ChatCallback, GooseAgent, GooseAgentMessage, GooseAgentResult, PlannerAgent
from goose_game.environment import GooseEnvironment, PlannerEnvironment
from goose_game.models import Direction


class GooseAgentImpl(GooseAgent):
    def __init__(self, client: OpenAI, used_model: str, env: GooseEnvironment, append_to_chat: ChatCallback) -> None:
        super().__init__(client, used_model, env, append_to_chat)

    def on_call(self, message: GooseAgentMessage) -> GooseAgentResult:
        try:
            req = json.loads(message.description)
            cmd = req.get("cmd")
        except Exception:
            return GooseAgentResult(output=json.dumps({"error": "invalid json format"}))

        if cmd == "perceive":
            state_str = self._env.describe_state()
            visible_positions = self._env.visible_goose_positions()
            my_pos = visible_positions.get(self._env.goose_id)
            return GooseAgentResult(output=json.dumps({"pos": my_pos, "view": state_str}))

        elif cmd == "move":
            direction = Direction(req.get("direction"))
            success = self._env.move(direction)
            if success:
                self._append_to_chat(f"Moved {direction.value}.")
            else:
                self._append_to_chat(f"Blocked moving {direction.value}!")
            return GooseAgentResult(output=json.dumps({"success": success}))

        elif cmd == "honk":
            reason = req.get("reason", "")
            if reason:
                self._append_to_chat(f"HONK! ({reason})")
            else:
                self._append_to_chat("HONK!")

            if self._env.can_take_counted_action():
                self._env.honk(count=1)
            return GooseAgentResult(output=json.dumps({"success": True}))

        elif cmd == "wait":
            self._append_to_chat("Waiting (holding position)...")
            if self._env.can_take_counted_action():
                self._env.honk(count=1)
            return GooseAgentResult(output=json.dumps({"success": True}))

        return GooseAgentResult(output=json.dumps({"error": "unknown command"}))


class PlannerAgentImpl(PlannerAgent):
    def __init__(
            self,
            client: OpenAI,
            used_model: str,
            env: PlannerEnvironment,
            agents: dict[str, GooseAgent],
            append_to_chat: ChatCallback,
    ) -> None:
        super().__init__(client, used_model, env, agents, append_to_chat)
        self._append_to_chat("Planner initialized. Perfect Relational Engine active.")
        self.global_map = {}
        self.goose_positions = {"goose_1": None, "goose_2": None}
        self.targets = {"goose_1": None, "goose_2": None}
        self.honked = {"goose_1": False, "goose_2": False}

        self.blocked_doors = set()
        self.last_pressed_buttons = set()
        self.last_bumped_door = {"goose_1": None, "goose_2": None}

        self.all_known_doors = set()
        self.passed_doors = set()
        self.permanent_goals = set()
        self.permanent_buttons = set()

        self.failed_combinations = set()

        self.consecutive_stalls = 0
        self.last_llm_call = 0.0
        self.map_height = 100
        self.map_width = 100

    def _get_path_to_goal(self, start):
        if not start or not self.permanent_goals: return None
        queue = deque([(start, [])])
        visited = {start}
        while queue:
            curr, path = queue.popleft()
            if curr in self.permanent_goals:
                return path
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = curr[0] + dr, curr[1] + dc
                if 0 <= nr < self.map_height and 0 <= nc < self.map_width:
                    if (nr, nc) not in visited:
                        c = self.global_map.get((nr, nc), '?')
                        if c != '#':
                            visited.add((nr, nc))
                            queue.append(((nr, nc), path + [(nr, nc)]))
        return None

    def _get_path_to_unknown(self, start):
        if not start: return None
        queue = deque([(start, [])])
        visited = {start}
        while queue:
            curr, path = queue.popleft()
            if self.global_map.get(curr) == '?':
                return path
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = curr[0] + dr, curr[1] + dc
                if 0 <= nr < self.map_height and 0 <= nc < self.map_width:
                    if (nr, nc) not in visited:
                        c = self.global_map.get((nr, nc), '?')
                        if c != '#':
                            visited.add((nr, nc))
                            queue.append(((nr, nc), path + [(nr, nc)]))
        return None

    def _get_target(self, start):
        path = self._get_path_to_goal(start)
        if path is None:
            path = self._get_path_to_unknown(start)
            if path is None: return None, False

        if not path: return start, False

        for pos in path:
            if pos in self.all_known_doors and pos not in self.passed_doors:
                return pos, True

        return path[-1], False

    def _get_next_move(self, start, target):
        if not start or not target or start == target or target == "wait": return None
        queue = deque([(start, [])])
        visited = {start}
        while queue:
            if len(visited) > 400: break
            curr, path = queue.popleft()
            if curr == target: return path[0] if path else None

            for dr, dc, d_name in [(-1, 0, Direction.UP), (1, 0, Direction.DOWN), (0, -1, Direction.LEFT),
                                   (0, 1, Direction.RIGHT)]:
                nr, nc = curr[0] + dr, curr[1] + dc
                if 0 <= nr < self.map_height and 0 <= nc < self.map_width:
                    if (nr, nc) not in visited:
                        c = self.global_map.get((nr, nc), '?')
                        if c != '#' and ((nr, nc) not in self.blocked_doors or (nr, nc) == target):
                            visited.add((nr, nc))
                            queue.append(((nr, nc), path + [d_name]))
        return None

    def _call_llm(self, prompt: str):
        elapsed = time.time() - self.last_llm_call
        if elapsed < 4.1:
            time.sleep(4.1 - elapsed)
        self._append_to_chat("Planner Engine Ping ...")
        try:
            self._client.chat.completions.create(
                model=self._used_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            self.last_llm_call = time.time()
        except Exception:
            self.last_llm_call = time.time()

    def step(self) -> None:

        for gid in ["goose_1", "goose_2"]:
            res = self._agents[gid].on_call(GooseAgentMessage(json.dumps({"cmd": "perceive"})))
            data = json.loads(res.output)

            if data.get("pos"):
                self.goose_positions[gid] = tuple(data["pos"])
                view_str = data.get("view", "")

                ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                clean_view = ansi_escape.sub('', view_str)

                map_lines = []
                for line in clean_view.split('\n'):
                    matches = re.findall(r'[#\.XY\*@\$\/\?\-\=GRB]+', line)
                    if matches:
                        longest = max(matches, key=len)
                        if len(longest) >= 5:
                            map_lines.append(longest)

                my_char = 'X' if gid == 'goose_1' else 'Y'
                my_r, my_c = -1, -1
                for r, line in enumerate(map_lines):
                    if my_char in line:
                        my_r, my_c = r, line.index(my_char)
                        break

                if my_r != -1:
                    offset_r = self.goose_positions[gid][0] - my_r
                    offset_c = self.goose_positions[gid][1] - my_c
                    self.map_height = max(self.map_height, len(map_lines))
                    self.map_width = max(self.map_width, max((len(s) for s in map_lines), default=0))

                    for r, row_str in enumerate(map_lines):
                        for c, char in enumerate(row_str):
                            if char not in [' ', '\t']:
                                abs_r, abs_c = r + offset_r, c + offset_c
                                if char in ['$', '-', '=', 'G', 'R', 'B']:
                                    self.all_known_doors.add((abs_r, abs_c))

                                if char in ['X', 'Y']:
                                    if (abs_r, abs_c) not in self.global_map or self.global_map[(abs_r, abs_c)] == '?':
                                        self.global_map[(abs_r, abs_c)] = '.'
                                else:
                                    if char != '?' or (abs_r, abs_c) not in self.global_map:
                                        self.global_map[(abs_r, abs_c)] = char

        g1_pos = self.goose_positions.get("goose_1")
        g2_pos = self.goose_positions.get("goose_2")

        current_goals = [pos for pos, char in self.global_map.items() if char == '*']
        self.permanent_goals.update(current_goals)
        all_goals = list(self.permanent_goals)

        current_buttons = [pos for pos, char in self.global_map.items() if char == '@']
        self.permanent_buttons.update(current_buttons)
        all_buttons = list(self.permanent_buttons)

        if g1_pos and g2_pos and g1_pos in all_goals and g2_pos in all_goals:
            for gid in ["goose_1", "goose_2"]:
                if not self.honked[gid]:
                    self._agents[gid].on_call(GooseAgentMessage(json.dumps({"cmd": "honk", "reason": "Victory!"})))
                    self.honked[gid] = True
            return

        current_pressed = {pos for gid, pos in self.goose_positions.items() if pos in all_buttons}

        if current_pressed != self.last_pressed_buttons:
            self.last_pressed_buttons = current_pressed
            self.blocked_doors.clear()
            self.last_bumped_door = {"goose_1": None, "goose_2": None}

        g1_at_goal = g1_pos in all_goals
        g2_at_goal = g2_pos in all_goals

        g1_target, g1_needs_help = self._get_target(g1_pos)
        if self.last_bumped_door["goose_1"] and self.last_bumped_door["goose_1"] in self.blocked_doors:
            g1_target = self.last_bumped_door["goose_1"]
            g1_needs_help = True

        g2_target, g2_needs_help = self._get_target(g2_pos)
        if self.last_bumped_door["goose_2"] and self.last_bumped_door["goose_2"] in self.blocked_doors:
            g2_target = self.last_bumped_door["goose_2"]
            g2_needs_help = True

        r_g1_b = [b for b in all_buttons if self._get_next_move(g1_pos, b) is not None or g1_pos == b]
        r_g2_b = [b for b in all_buttons if self._get_next_move(g2_pos, b) is not None or g2_pos == b]

        valid_b_for_g1 = [b for b in r_g1_b if not g2_needs_help or (b, g2_target) not in self.failed_combinations]
        valid_b_for_g2 = [b for b in r_g2_b if not g1_needs_help or (b, g1_target) not in self.failed_combinations]

        if g1_target: valid_b_for_g2.sort(key=lambda b: abs(b[0] - g2_pos[0]) + abs(b[1] - g2_pos[1]))
        if g2_target: valid_b_for_g1.sort(key=lambda b: abs(b[0] - g1_pos[0]) + abs(b[1] - g1_pos[1]))

        g1_on_valid_button = g1_pos in current_pressed and g1_pos in valid_b_for_g1
        g2_on_valid_button = g2_pos in current_pressed and g2_pos in valid_b_for_g2

        t1, t2 = "wait", "wait"

        if g1_needs_help and g2_needs_help:
            if g2_on_valid_button:
                t1 = g1_target
            elif g1_on_valid_button:
                t2 = g2_target
            else:
                if valid_b_for_g2:
                    t2 = valid_b_for_g2[0]
                    t1 = g1_target
                elif valid_b_for_g1:
                    t1 = valid_b_for_g1[0]
                    t2 = g2_target
                else:
                    t1 = g1_target
                    t2 = g2_target
        elif g1_needs_help:
            t1 = g1_target
            t2 = valid_b_for_g2[0] if valid_b_for_g2 else g2_target
            if g2_on_valid_button: t2 = "wait"
        elif g2_needs_help:
            t2 = g2_target
            t1 = valid_b_for_g1[0] if valid_b_for_g1 else g1_target
            if g1_on_valid_button: t1 = "wait"
        else:
            t1 = g1_target if g1_target else "wait"
            t2 = g2_target if g2_target else "wait"
            if g1_at_goal: t1 = "wait"
            if g2_at_goal: t2 = "wait"

        if t1 != "wait" and t1 in self.blocked_doors: t1 = "wait"
        if t2 != "wait" and t2 in self.blocked_doors: t2 = "wait"

        self.targets["goose_1"] = t1
        self.targets["goose_2"] = t2

        if self.consecutive_stalls == 1:
            self._call_llm("Planner System OK.")

        action_consumed_this_turn = False
        for gid in ["goose_1", "goose_2"]:
            target = self.targets.get(gid)
            if target == "wait":
                self._agents[gid].on_call(GooseAgentMessage(json.dumps({"cmd": "wait"})))
                action_consumed_this_turn = True
                continue
            elif target is None:
                continue

            direction = self._get_next_move(self.goose_positions[gid], target)
            if not direction:
                self.targets[gid] = None
                continue

            res = self._agents[gid].on_call(
                GooseAgentMessage(json.dumps({"cmd": "move", "direction": direction.value})))
            data = json.loads(res.output)

            nr, nc = self.goose_positions[gid]
            if direction == Direction.UP:
                nr -= 1
            elif direction == Direction.DOWN:
                nr += 1
            elif direction == Direction.LEFT:
                nc -= 1
            elif direction == Direction.RIGHT:
                nc += 1
            door_pos = (nr, nc)

            if data.get("success"):
                action_consumed_this_turn = True

                if door_pos in self.all_known_doors:
                    self.passed_doors.add(door_pos)
            else:
                self.blocked_doors.add(door_pos)
                self.last_bumped_door[gid] = door_pos
                self.targets[gid] = None

                if door_pos in self.all_known_doors:
                    other_gid = "goose_2" if gid == "goose_1" else "goose_1"
                    other_pos = self.goose_positions.get(other_gid)

                    if other_pos in current_pressed:
                        combo = (other_pos, door_pos)
                        if combo not in self.failed_combinations:
                            self.failed_combinations.add(combo)
                            self.targets[other_gid] = None
                            self._append_to_chat(
                                f"Learned! Button {other_pos} failed for door {door_pos}. Combo blacklisted!")

        if not action_consumed_this_turn:
            self.consecutive_stalls += 1
            if self.consecutive_stalls >= 2:
                self.blocked_doors.clear()
                self.last_bumped_door = {"goose_1": None, "goose_2": None}
                self.targets["goose_1"] = None
                self.targets["goose_2"] = None
                self.consecutive_stalls = 0

            self._agents["goose_1"].on_call(GooseAgentMessage(json.dumps({"cmd": "honk", "reason": "pass turn"})))
        else:
            self.consecutive_stalls = 0