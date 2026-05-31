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
        self._append_to_chat(f"Planner message: {message.description}")
        text = message.description.lower()

        if "up" in text:
            action = "move"
            arg = "up"
        elif "down" in text:
            action = "move"
            arg = "down"
        elif "left" in text:
            action = "move"
            arg = "left"
        elif "right" in text:
            action = "move"
            arg = "right"
        elif "honk" in text:
            action = "honk"
            arg = 1
        else:
            return GooseAgentResult(output=f"Error while parsing message: {message.description}")

        if action == "move":
            boolean = self._env.move(arg)
            result = (
                f"Move was {'made' if boolean else 'not made'} in direction {arg}!\n"
                f"Current map view:\n"
                f"{self._env.describe_state()}\n"
            )
        else:
            event = self._env.honk(arg)
            result = (
                f"{self._env.goose_id} honked at {event.position}.\n"
                f"Current map view:\n"
                f"{self._env.describe_state()}\n"
            )

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
        messages = [
            {
                "role" : "system",
                "content":
                    """You are the Planner Agent coordinating two geese (Goose 1 and Goose 2) to solve an unknown grid-based puzzle.
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
                    3. PHYSICS: Geese CANNOT move through walls (#), closed doors ($), or outside the map boundaries.
                    4. WAITING: Use "honk" to stay in place without moving (e.g., to hold a button or wait for a door to open).
            
                    Universal Cooperation & Discovery Strategy (Applies to all levels):
                    - EXPLORATION: In hard mode, the map is partially hidden (?). If the goal (*) or path is not visible, move towards '?' to explore and discover the map.
                    - DISCOVERY: Button-door wiring is unknown. If one goose is blocked by a closed door ($), the other goose must explore to find and step on buttons (@) to see which one opens the door.
                    - COOPERATION: To test a button, the helper goose stands on it and uses 'honk'. The blocked goose waits and observes. If the door opens (/), the blocked goose passes through. If not, the helper must find another button.
                    - RELEASING: Once the blocked goose has safely passed the door or reached the goal (*), the helper goose MUST stop holding the button and immediately proceed to its own goal. It is completely fine if the door closes ($) behind the safe goose!
                    - FINISHING: Both geese must reach the goal (*) and honk.
            
                    Strategy for your 'thought' process:
                    When creating your plan in the 'thought' key, you MUST strictly follow these exact steps:
                    Step 1. Map Coordinates: Count the rows (top to bottom) and columns (left to right). Locate X, Y, @, /, $, and *. If X or Y is missing, they are on a button or the goal.
                    Step 2. Status Check: Who is blocked? Who is helping? Is the door open (/) or closed ($)?
                    Step 3. Surrounding Scan (CRITICAL): For each moving goose, look at the map and explicitly state what symbol is DIRECTLY adjacent to it:
                       - UP: [symbol]
                       - DOWN: [symbol]
                       - LEFT: [symbol]
                       - RIGHT: [symbol]
                    Step 4. Collision & Anti-Loop: You CANNOT move into '#' or '$'. If your desired direction (e.g., left towards the goal) is blocked by '#', you MUST choose a different open path ('.', '/') to walk around the wall. Do not reverse your previous move.
                    Step 5. Action: Select ONE valid command for Goose 1 and Goose 2.
                    """
            },
            {
                "role" : "user",
                "content":
                    f"Task description: {self._env.task_description}\n"
                    f"History of moves: {self.history[-6:]}\n"
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
            parsed_data = json.loads(clean_text)

            if not isinstance(parsed_data, dict):
                self._append_to_chat("Warning: LLM returned invalid structure. Defaulting to honk.")
                parsed_data = {"goose_1": "honk", "goose_2": "honk"}

        except Exception:
            self._append_to_chat("Error while parsing message. Defaulting to honk.")
            parsed_data = {"goose_1": "honk", "goose_2": "honk"}

        for goose_id, goose in sorted(self._agents.items()):
            action_text = parsed_data.get(goose_id, "honk")
            description = f"{goose_id} : {action_text} "
            task = GooseAgentMessage(description=description)
            self._append_to_chat(f"Calling {goose_id}.")
            result = goose.on_call(task)

            if result.error is not None:
                self._append_to_chat(f"{goose_id} error: {result.error}")
            else:
                self._append_to_chat(f"{goose_id} result: {result.output}")
                self.history.append(f"{goose_id} : {result.output}")
