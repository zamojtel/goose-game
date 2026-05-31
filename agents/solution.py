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
            return GooseAgentResult(error=f"Error while parsing message: {message.description}")

        if action == "move":
            success = self._env.move(arg)
            if success:
                visible_geese = self._env.visible_goose_positions()
                pos = visible_geese.get(self._env.goose_id, "unknown")
                result = f"Moved {arg} -> SUCCESS (now at {pos})"
            else:
                result = f"Moved {arg} -> FAILED (Blocked by wall # or closed door $)"
        else:
            event = self._env.honk(arg)
            result = f"Honked at {event.position}"

        return GooseAgentResult(output=result)

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
        self._append_to_chat(f"Initialized planner for level: {env.level_name}.")
        self.history = []

        self.last_thought = "First turn, no previous thoughts."

    def step(self) -> None:
        self._append_to_chat("Planner step executed.")

        obs_1 = self._agents["goose_1"].on_call(GooseAgentMessage(description="observe")).output
        obs_2 = self._agents["goose_2"].on_call(GooseAgentMessage(description="observe")).output

        history_text = "\n".join(self.history[-16:]) if self.history else "First turn - no moves yet."

        messages = [
            {
                "role": "system",
                "content": """You are the Planner Agent coordinating two geese (Goose 1 and Goose 2) to solve an unknown grid-based puzzle.
                You do NOT know the level mechanics in advance. You must deduce them dynamically based on the task description, map state, and history.

                CRITICAL OUTPUT FORMAT:
                You MUST return ONLY a valid JSON object matching EXACTLY this structure. Do not add markdown or text outside the JSON.
                {
                  "thought": {
                    "door_status": "Is the door closed ($) or open (/)?",
                    "goose_1_analysis": "Where is Goose 1? What is it doing?",
                    "goose_2_analysis": "Where is Goose 2? What is it doing?",
                    "anti_loop_and_memory": "Check RECENT ACTION HISTORY and YOUR LAST THOUGHT. Which buttons (coordinates) failed? Which direction is forbidden to avoid reversing?"
                  },
                  "goose_1": "command (up, down, left, right, or honk)",
                  "goose_2": "command (up, down, left, right, or honk)"
                }

                How to understand different symbols:
                @ - button
                $ - closed door
                / - open door
                # - wall
                . - empty field
                * - target (goal)
                ? - unknown area (unexplored map in hard mode)
                X - Goose 1
                Y - Goose 2

                Game rules & Physics:
                1. OVERLAPPING (CRITICAL): The text map shows only one symbol per tile. If a goose is on a button or goal, its letter (X or Y) covers the '@' or '*'. If you cannot find 'X' or 'Y' on the map, it means that goose is safely on the goal or a button! Do not panic.
                2. ALLOWED COMMANDS: up, down, left, right, and honk.
                3. WALKABLE TILES (CRITICAL): Geese CAN walk onto empty fields (.), buttons (@), OPEN DOORS (/), goals (*), and unknown areas (?). The symbol '/' is NOT an obstacle! It is a safe floor tile. You CAN and MUST move directly into the '/' symbol to pass through the door!
                4. PHYSICS: Geese CANNOT move through walls (#) or closed doors ($).
                5. WAITING: Use "honk" to stay in place without moving.

                Universal Cooperation & Discovery Strategy (Applies to all levels):
                - EXPLORATION: In hard mode, if the goal (*) or path is not visible, move towards '?' to uncover the map.
                - SHARED VISION: In hard mode, one goose might not see the door. You MUST combine both Goose 1 and Goose 2 views to check if a door is open (/).
                - TESTING BUTTONS: If a goose is blocked by a closed door ($), the helper goose must step on buttons (@) to test them. Use 'honk' to stay on the button and observe. 
                - MEMORY: Check the action history and YOUR LAST THOUGHT! If the helper stepped on a button (check coordinates) and the door remained closed ($) in the next turn, it's the WRONG button. The helper must step off and find a DIFFERENT button. DO NOT test the same wrong button twice!
                - HOLDING & CROSSING (CRITICAL PHASE): If the door is OPEN (/), the helper has found the CORRECT button! The helper MUST output 'honk' every turn to stay on the button and keep the door open. The blocked goose MUST immediately WALK INTO the open door (move onto the '/' tile). DO NOT HESITATE!
                - RELEASING: Once the blocked goose has crossed the door, it is safe! The helper MUST leave the button and go to its own goal (*). Ignore the door closing ($) behind the safe goose.
                - NO BACKTRACKING: Once a goose walks through a door, it must NEVER walk back through it.
                - FINISHING: Both geese must reach the goal (*) and honk.

                Strategy for your 'thought' process:
                When creating your plan in the 'thought' key, you MUST strictly follow these exact steps:
                Step 1. Locate: Find X, Y, @, /, $, and *. Check BOTH views! Is there an open door (/)?
                Step 2. Status Check: Who is blocked? Who is helping? Which buttons have been tested and failed (check the history and YOUR LAST THOUGHT for coordinates)?
                Step 3. Door Action: If the door is OPEN (/), the helper MUST 'honk', and the blocked goose MUST move towards and INTO the '/' tile.
                Step 4. Surrounding Scan: For each moving goose, state what symbol is DIRECTLY adjacent: UP, DOWN, LEFT, RIGHT. (Remember: '/' is walkable!).
                Step 5. Anti-Loop: Plan the route avoiding '#' and '$'. DO NOT reverse your previous move. NEVER bounce between the same tiles.
                Step 6. Action: Select ONE valid command for Goose 1 and Goose 2 based on the JSON schema.
                """
            },
            {
                "role": "user",
                "content":
                    f"Task description: {self._env.task_description}\n\n"
                    f"--- GOOSE 1 VIEW ---\n{obs_1}\n\n"
                    f"--- GOOSE 2 VIEW ---\n{obs_2}\n\n"
                    # DODATEK 3: Przesyłamy modelowi jego własne myśli z poprzedniej tury
                    f"--- YOUR LAST THOUGHT (MEMORY) ---\n{json.dumps(self.last_thought)}\n\n"
                    f"--- RECENT ACTION HISTORY ---\n{history_text}\n"
            }
        ]

        time.sleep(8)

        try:
            response = self._client.chat.completions.create(
                model=self._used_model,
                messages=messages
            )
        except Exception as e:
            self._append_to_chat(f"Krytyczny błąd API: {str(e)}")
            self._append_to_chat("Zatrzymuję program, aby nie marnować limitu zapytań!")
            raise RuntimeError(f"Przerywam działanie workflow z powodu błędu API: {str(e)}")

        try:
            clean_text = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
            start = clean_text.find("{")
            end = clean_text.rfind("}")
            if start != -1 and end != -1:
                clean_text = clean_text[start:end + 1]
            parsed_data = json.loads(clean_text)

            if isinstance(parsed_data, dict) and "thought" in parsed_data:
                self.last_thought = parsed_data["thought"]

            if not isinstance(parsed_data, dict):
                self._append_to_chat("Warning: LLM returned invalid structure. Defaulting to honk.")
                parsed_data = {"goose_1": "honk", "goose_2": "honk"}

        except Exception:
            self._append_to_chat("Error while parsing message. Defaulting to honk.")
            parsed_data = {"goose_1": "honk", "goose_2": "honk"}

        turn_logs = []
        for goose_id, goose in sorted(self._agents.items()):
            action_text = parsed_data.get(goose_id, "honk")
            task = GooseAgentMessage(description=action_text)
            self._append_to_chat(f"Calling {goose_id}.")
            result = goose.on_call(task)

            if result.error is not None:
                self._append_to_chat(f"{goose_id} error: {result.error}")
                turn_logs.append(f"{goose_id} tried '{action_text}' -> ERROR")
            else:
                self._append_to_chat(f"{goose_id} result: {result.output}")
                turn_logs.append(f"{goose_id}: {result.output}")

        self.history.append(" | ".join(turn_logs))