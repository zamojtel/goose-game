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
                self._append_to_chat("Reached API limit. Awaiting 30 seconds...")
                time.sleep(30)

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
                "content" : "You are the planner agent coordinating two geese to solve a goose game. "
                            "Based on the task description and history, propose the next instruction for each goose. "
                            "Return only JSON object with two keys 'goose_1' and 'goose_2' and string values containing their next instructions. "
                            "Provide only JSON no extra text or formatting. "
            },
            {
                "role" : "user",
                "content": f"Task description: {self._env.task_description}\n"
                           f"History of moves: {self.history}\n"
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
                self._append_to_chat("Reached API limit. Awaiting 30 seconds...")
                time.sleep(30)

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
