"""NLP node using OpenAI."""

from typing import Dict, List
import logging

import openai
from rclpy.node import Node
from std_msgs.msg import String


class NlpNode(Node):
    """Node that queries an LLM."""

    def __init__(self) -> None:
        super().__init__("nlp_node")
        self.create_subscription(String, "/robot/vision/scene", self._cb, 10)
        self.pub = self.create_publisher(String, "/robot/vision/actions", 10)

    def _cb(self, msg: String) -> None:
        try:
            actions = self._call_llm(msg.data)
            self.pub.publish(String(data=str(actions)))
        except Exception as exc:  # pragma: no cover - ignore
            logging.error("LLM call failed: %s", exc)

    def _call_llm(self, scene_json: str) -> List[Dict]:
        try:
            res = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[], functions=[])
            return res  # type: ignore
        except Exception as exc:
            logging.error("OpenAI API error: %s", exc)
            raise
