from openai import OpenAI, OpenAIError
from pandas.core.methods import describe
from streamlit.string_util import clean_text

import time
from agents.base import ChatCallback, GooseAgent, GooseAgentMessage, GooseAgentResult, PlannerAgent
from goose_game.environment import GooseEnvironment, PlannerEnvironment

import json


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
                result = f"Moved {arg} -> SUCCESS"
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

    def step(self) -> None:
        self._append_to_chat("Planner step executed.")

        # Pobieramy aktualny obraz mapy PRZED zaplanowaniem ruchów
        obs_1 = self._agents["goose_1"].on_call(GooseAgentMessage(description="observe")).output
        obs_2 = self._agents["goose_2"].on_call(GooseAgentMessage(description="observe")).output

        # ZWIĘKSZAMY PAMIĘĆ do 16 ostatnich akcji, by uniknąć amnezji fałszywych przycisków!
        history_text = "\n".join(self.history[-16:]) if self.history else "First turn - no moves yet."

        messages = [
            {
                "role": "system",
                "content": """You are the Planner Agent coordinating two geese (Goose 1 and Goose 2) to solve an unknown grid-based puzzle.
                You do NOT know the level mechanics in advance. You must deduce them dynamically based on the task description, map state, and history.
                Return ONLY a valid JSON object with exactly three keys: 
                'thought' (step-by-step analysis and deduction of the current state),
                'goose_1' and 'goose_2' with string values containing their next instructions. 
                Provide only JSON, no markdown formatting (do not use ```json blocks).

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
                - MEMORY: Check the action history! If the helper stepped on a button and the door remained closed ($) in the next turn, it's the WRONG button. The helper must step off and find a DIFFERENT button. DO NOT test the same wrong button twice!
                - HOLDING & CROSSING (CRITICAL PHASE): If the door is OPEN (/), the helper has found the CORRECT button! The helper MUST output 'honk' every turn to stay on the button and keep the door open. The blocked goose MUST immediately WALK INTO the open door (move onto the '/' tile). DO NOT HESITATE!
                - RELEASING: Once the blocked goose has crossed the door, it is safe! The helper MUST leave the button and go to its own goal (*). Ignore the door closing ($) behind the safe goose.
                - NO BACKTRACKING: Once a goose walks through a door, it must NEVER walk back through it.

                Strategy for your 'thought' process:
                When creating your plan in the 'thought' key, you MUST strictly follow these exact steps:
                Step 1. Locate: Find X, Y, @, /, $, and *. Check BOTH views! Is there an open door (/)?
                Step 2. Status Check: Who is blocked? Who is helping? Which buttons have been tested and failed (check the history)?
                Step 3. Door Action: If the door is OPEN (/), the helper MUST 'honk', and the blocked goose MUST move towards and INTO the '/' tile.
                Step 4. Surrounding Scan: For each moving goose, state what symbol is DIRECTLY adjacent: UP, DOWN, LEFT, RIGHT. (Remember: '/' is walkable!).
                Step 5. Anti-Loop: Plan the route avoiding '#' and '$'. DO NOT reverse your previous move. NEVER bounce between the same tiles.
                Step 6. Action: Select ONE valid command for Goose 1 and Goose 2.
                """
            },
            {
                "role": "user",
                "content":
                    f"Task description: {self._env.task_description}\n\n"
                    f"--- GOOSE 1 VIEW ---\n{obs_1}\n\n"
                    f"--- GOOSE 2 VIEW ---\n{obs_2}\n\n"
                    f"--- RECENT ACTION HISTORY ---\n{history_text}\n"
            }
        ]

        # Bezpieczne 8 sekund opóźnienia, by uniknąć błędu 429
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

    # def step(self) -> None:
    #     self._append_to_chat("Planner step executed.")
    #     messages = [
    #         {
    #             "role" : "system",
    #             "content":
    #                 """You are the Planner Agent coordinating two geese (Goose 1 and Goose 2) to solve an unknown grid-based puzzle.
    #                 You do NOT know the level mechanics in advance. You must deduce them dynamically based on the task description, map state, and history.
    #                 Return ONLY a valid JSON object with exactly three keys:
    #                 'thought' (step-by-step analysis and deduction of the current state),
    #                 'goose_1' and 'goose_2' with string values containing their next instructions.
    #                 Provide only JSON, no markdown formatting (do not use ```json blocks).
    #
    #                 How to understand different symbols:
    #                 @ - button
    #                 $ - closed door
    #                 / - open door
    #                 # - wall
    #                 . - empty field
    #                 * - target (goal)
    #                 ? - unknown area (unexplored map in hard mode)
    #                 X - Goose 1
    #                 Y - Goose 2
    #
    #                 Game rules & Physics:
    #                 1. OVERLAPPING (CRITICAL): The text map shows only one symbol per tile. If a goose is on a button or goal, its letter (X or Y) covers the '@' or '*'. If you cannot find 'X' or 'Y' on the map, it means that goose is safely on the goal or a button! Do not panic.
    #                 2. ALLOWED COMMANDS: up, down, left, right, and honk.
    #                 3. PHYSICS: Geese CANNOT move through walls (#), closed doors ($), or outside the map boundaries.
    #                 4. WAITING: Use "honk" to stay in place without moving (e.g., to hold a button or wait for a door to open).
    #
    #                 Universal Cooperation & Discovery Strategy (Applies to all levels):
    #                 - EXPLORATION: In hard mode, the map is partially hidden (?). If the goal (*) or path is not visible, move towards '?' to explore and discover the map.
    #                 - DISCOVERY: Button-door wiring is unknown. If one goose is blocked by a closed door ($), the other goose must explore to find and step on buttons (@) to see which one opens the door.
    #                 - COOPERATION: To test a button, the helper goose stands on it and uses 'honk'. The blocked goose waits and observes. If the door opens (/), the blocked goose passes through. If not, the helper must find another button.
    #                 - RELEASING: Once the blocked goose has safely passed the door or reached the goal (*), the helper goose MUST stop holding the button and immediately proceed to its own goal. It is completely fine if the door closes ($) behind the safe goose!
    #                 - FINISHING: Both geese must reach the goal (*) and honk.
    #
    #                 Strategy for your 'thought' process:
    #                 When creating your plan in the 'thought' key, you MUST strictly follow these exact steps:
    #                 Step 1. Map Coordinates: Count the rows (top to bottom) and columns (left to right). Locate X, Y, @, /, $, and *. If X or Y is missing, they are on a button or the goal.
    #                 Step 2. Status Check: Who is blocked? Who is helping? Is the door open (/) or closed ($)?
    #                 Step 3. Surrounding Scan (CRITICAL): For each moving goose, look at the map and explicitly state what symbol is DIRECTLY adjacent to it:
    #                    - UP: [symbol]
    #                    - DOWN: [symbol]
    #                    - LEFT: [symbol]
    #                    - RIGHT: [symbol]
    #                 Step 4. Collision & Anti-Loop: You CANNOT move into '#' or '$'. If your desired direction (e.g., left towards the goal) is blocked by '#', you MUST choose a different open path ('.', '/') to walk around the wall. Do not reverse your previous move.
    #                 Step 5. Action: Select ONE valid command for Goose 1 and Goose 2.
    #                 """
    #         },
    #         {
    #             "role" : "user",
    #             "content":
    #                 f"Task description: {self._env.task_description}\n"
    #                 f"History of moves: {self.history[-6:]}\n"
    #         }
    #     ]
    #
    #     time.sleep(8)
    #
    #     try:
    #         response = self._client.chat.completions.create(
    #             model=self._used_model,
    #             messages=messages
    #         )
    #     except Exception as e:
    #         self._append_to_chat(f"Krytyczny błąd API: {str(e)}")
    #         self._append_to_chat("Zatrzymuję program, aby nie marnować limitu zapytań!")
    #         raise RuntimeError(f"Przerywam działanie workflow z powodu błędu API: {str(e)}")
    #
    #     try:
    #         clean_text = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
    #         parsed_data = json.loads(clean_text)
    #
    #         if not isinstance(parsed_data, dict):
    #             self._append_to_chat("Warning: LLM returned invalid structure. Defaulting to honk.")
    #             parsed_data = {"goose_1": "honk", "goose_2": "honk"}
    #
    #     except Exception:
    #         self._append_to_chat("Error while parsing message. Defaulting to honk.")
    #         parsed_data = {"goose_1": "honk", "goose_2": "honk"}
    #
    #     for goose_id, goose in sorted(self._agents.items()):
    #         action_text = parsed_data.get(goose_id, "honk")
    #         description = f"{goose_id} : {action_text} "
    #         task = GooseAgentMessage(description=description)
    #         self._append_to_chat(f"Calling {goose_id}.")
    #         result = goose.on_call(task)
    #
    #         if result.error is not None:
    #             self._append_to_chat(f"{goose_id} error: {result.error}")
    #         else:
    #             self._append_to_chat(f"{goose_id} result: {result.output}")
    #             self.history.append(f"{goose_id} : {result.output}")
