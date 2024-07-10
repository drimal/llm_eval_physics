from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class MessageContent:
    type: str
    text: Optional[str] = None
    image_url: Optional[str] = None


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class MessageBuilder:
    system_prompt: str
    mcq_prompt: str
    general_prompt: str

    def build_content(self, question: str,
                      images: Optional[List[str]] = None) -> str:
        return question

    def create_messages(self, qtype: str, question: str,
                        images: Optional[List[str]] = None) -> List[ChatMessage]:
        if qtype == "mcq":
            prompt = self.mcq_prompt + "** Question **\n" + question
        else:
            prompt = self.general_prompt + "** Question **\n" + question

        content = self.build_content(prompt, images)
        messages = [
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(role="user", content=content)
        ]

        return messages
