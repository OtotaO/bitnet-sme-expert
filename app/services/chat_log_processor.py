"""Module for processing chat logs into training data for fine-tuning."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""

    role: str  # 'user' or 'assistant'
    content: str
    timestamp: float | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class TrainingExample:
    """A single training example for fine-tuning."""

    instruction: str
    input: str = ""
    output: str = ""
    system_prompt: str = ""
    metadata: dict[str, Any] | None = None


class ChatLogProcessor:
    """Processes chat logs into training examples for fine-tuning."""

    def __init__(self, system_prompt: str = ""):
        """Initialize the chat log processor.

        Args:
            system_prompt: Default system prompt to use for training examples
        """
        self.system_prompt = system_prompt

    def load_chat_logs(self, file_path: str | Path) -> list[dict[str, Any]]:
        """Load chat logs from a JSON or JSONL file."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Chat log file not found: {file_path}")

        try:
            if file_path.suffix == ".jsonl":
                with open(file_path, encoding="utf-8") as f:
                    return [json.loads(line) for line in f if line.strip()]
            else:  # Assume regular JSON
                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
                    return [data]
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing chat log file {file_path}: {e}")
            raise

    def parse_conversation(self, conversation: dict | list) -> list[ConversationTurn]:
        """Parse a conversation into a list of turns."""
        turns = []

        if isinstance(conversation, dict):
            # Try to extract conversation from common formats
            if "conversations" in conversation:
                return self.parse_conversation(conversation["conversations"])
            elif "messages" in conversation:
                return self.parse_conversation(conversation["messages"])
            elif "turns" in conversation:
                return self.parse_conversation(conversation["turns"])

            # Try to parse as a single message
            try:
                role = conversation.get("role", "").lower()
                if not role and "from" in conversation:
                    role = conversation["from"].lower()

                if role in ["user", "assistant", "system"]:
                    return [
                        ConversationTurn(
                            role=role,
                            content=conversation.get("content", ""),
                            timestamp=conversation.get("timestamp"),
                            metadata=conversation.get("metadata"),
                        )
                    ]
            except Exception as e:
                logger.warning(f"Failed to parse conversation: {e}")

        elif isinstance(conversation, list):
            for item in conversation:
                turns.extend(self.parse_conversation(item))
            return turns

        return turns

    def create_training_examples(
        self,
        conversation: list[ConversationTurn],
        max_length: int = 2048,
        include_system_prompt: bool = True,
    ) -> list[TrainingExample]:
        """Convert a conversation into training examples."""
        if not conversation:
            return []

        examples = []
        current_dialogue = []

        for turn in conversation:
            if turn.role == "system":
                if include_system_prompt:
                    self.system_prompt = turn.content
                continue

            current_dialogue.append(turn)

            # If this is an assistant turn, create a training example
            if turn.role == "assistant" and len(current_dialogue) > 1:
                # Find the last user message before this assistant turn
                user_messages = [t for t in current_dialogue[:-1] if t.role == "user"]
                if not user_messages:
                    continue

                user_turn = user_messages[-1]
                assistant_turn = turn

                example = TrainingExample(
                    instruction=user_turn.content,
                    output=assistant_turn.content,
                    system_prompt=self.system_prompt,
                    metadata={
                        "user_metadata": user_turn.metadata or {},
                        "assistant_metadata": assistant_turn.metadata or {},
                    },
                )
                examples.append(example)

        return examples

    def process_chat_logs(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        max_examples: int | None = None,
    ) -> list[dict[str, Any]]:
        """Process chat logs and optionally save to file."""
        # Load and parse chat logs
        chat_logs = self.load_chat_logs(input_path)
        all_examples = []

        for log in chat_logs:
            try:
                turns = self.parse_conversation(log)
                examples = self.create_training_examples(turns)
                all_examples.extend(examples)

                if max_examples and len(all_examples) >= max_examples:
                    all_examples = all_examples[:max_examples]
                    break

            except Exception as e:
                logger.error(f"Error processing chat log: {e}", exc_info=True)

        # Convert to dict format for JSON serialization
        result = [
            {
                "instruction": ex.instruction,
                "input": ex.input,
                "output": ex.output,
                "system_prompt": ex.system_prompt,
                "metadata": ex.metadata or {},
            }
            for ex in all_examples
        ]

        # Save to file if output path is provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_path.suffix == ".jsonl":
                with open(output_path, "w", encoding="utf-8") as f:
                    for item in result:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
            else:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved {len(result)} training examples to {output_path}")

        return result

    @classmethod
    def convert_to_alpaca_format(
        cls, examples: list[dict[str, Any]], include_system_prompt: bool = True
    ) -> list[dict[str, str]]:
        """Convert examples to Alpaca format for fine-tuning."""
        formatted = []

        for ex in examples:
            system = (ex.get("system_prompt", "") or "").strip()
            instruction = ex.get("instruction", "").strip()
            input_text = ex.get("input", "").strip()
            output = ex.get("output", "").strip()

            # Build the prompt
            prompt_parts = []
            if system and include_system_prompt:
                prompt_parts.append(f"System: {system}")

            prompt_parts.append(f"### Instruction:\n{instruction}")

            if input_text:
                prompt_parts.append(f"### Input:\n{input_text}")

            prompt = "\n\n".join(prompt_parts)

            formatted.append(
                {
                    "text": f"{prompt}\n\n### Response:\n{output}",
                    "prompt": prompt,
                    "response": output,
                    "metadata": ex.get("metadata", {}),
                }
            )

        return formatted
