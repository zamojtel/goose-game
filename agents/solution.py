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

        messages = [
            {
                "role" : "system",
                "content" : "Return only JSON format. The JSON must have exactly two keys: 'action' (which must be 'move' or 'honk') and 'arg' (which must be 'up', 'down', 'left', 'right' for move, or 1 for honk)."
            },
            {
                "role": "user",
                "content":
                    f"{message.description}\n"
                    +
                    f"Map\n {self._env.describe_state()}\n"
            }
        ]

        while True:
            try:
                response = self._client.chat.completions.create(
                    model=self._used_model,
                    messages=messages
                )
                break
            except Exception:
                self._append_to_chat("Reached API limit. Awaiting 10 seconds...")
                time.sleep(10)

        try:
            clean_text = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
            parsed_data = json.loads(clean_text)
        except json.decoder.JSONDecodeError:
            return GooseAgentResult(output=f"Error while parsing message")

        if parsed_data["action"] == "move":
            boolean = self._env.move(parsed_data["arg"])
            result = (
                f"Move was { 'made' if boolean else 'not made ' } in direction {parsed_data['arg']}!\n"
                f"Current map view:\n"
                f"{self._env.describe_state()}\n"
            )
        else:
            event = self._env.honk(parsed_data["arg"])
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
                "content" : """You are the planner agent coordinating two geese to solve a goose game.
                            Based on the task description and history, propose the next instruction for each goose.
                            Return only JSON object with exactly three keys: 
                            'thought' (where you analyze the board, positions of the geese, and plan the next steps step-by-step),
                            'goose_1' and 'goose_2' with string values containing their next instructions. 
                            Provide only JSON no extra text or formatting.
                            
                            How to understand different symbols:
                            @ - it's a button
                            $ - it's a closed door
                            / - it's an open door
                            # - it's a wall
                            . - it's an empty field
                            * - it's a target (goal)
                            1 - Goose 1
                            2 - Goose 2
                    
                            Game rules:
                            1. Geese can and sometimes must stand on the same tile at the same time.
                            2. The only allowed commands you can generate are: up, down, left, right, and honk.
                            3. PHYSICS: Geese CANNOT move through walls (#) or closed doors ($).
                            4. WAITING: If a goose needs to safely wait in its current position (e.g., to hold a button for the other goose to pass through a door), it MUST use the "honk" command to skip its turn without moving.
                    
                            Strategy for your 'thought' process:
                            When creating your plan in the 'thought' key, you MUST rely on the 'Task description' and strictly follow these exact steps before deciding on the final commands:
                            Step 1. Task Analysis: What is the current goal based on the 'Task description'?
                            Step 2. Locate: Find the exact positions of Goose 1, Goose 2, and the key objects (@, $, /, *) on the map.
                            Step 3. Check Surroundings: Look at the tiles directly UP, DOWN, LEFT, and RIGHT of each goose. 
                            Step 4. Obstacle Check: Ensure your planned moves do not lead into a wall (#) or a closed door ($).
                            Step 5. Action: Select exactly ONE valid command for Goose 1 and Goose 2 based on the analysis above.
                            """
            },
            {
                "role" : "user",
                "content": f"Task description: {self._env.task_description}\n"
                           f"History of moves: {self.history[-6:]}\n"
            }
        ]

        while True:
            try:
                response = self._client.chat.completions.create(
                    model=self._used_model,
                    messages=messages,
                )
                break
            except Exception:
                self._append_to_chat("Reached API limit. Awaiting 10 seconds...")
                time.sleep(10)

        try:
            clean_text = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
            parsed_data = json.loads(clean_text)
        except json.decoder.JSONDecodeError:
            self._append_to_chat("Error while parsing message")
            return

        for goose_id, goose in sorted(self._agents.items()):
            description = f"{goose_id} : {parsed_data[goose_id]} "
            task = GooseAgentMessage(description=description)
            self._append_to_chat(f"Calling {goose_id}.")
            result = goose.on_call(task)

            if result.error is not None:
                self._append_to_chat(f"{goose_id} error: {result.error}")
            else:
                self._append_to_chat(f"{goose_id} result: {result.output}")
                self.history.append(f"{goose_id} : {result.output}")
