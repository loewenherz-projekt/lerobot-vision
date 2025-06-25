"""ws/src/lerobot_vision/lerobot_vision/nlp_node.py"""

from __future__ import annotations

import json
import logging
from typing import Dict, List

import openai
from rclpy.node import Node
from std_msgs.msg import String

logger = logging.getLogger(__name__)


class NlpNode(Node):
    """ROS node that queries an LLM for actions."""

    def __init__(self) -> None:
        super().__init__("nlp_node")
        self.pub = self.create_publisher(String, "/robot/vision/actions", 1)
        self.create_subscription(
            String, "/robot/vision/scene", self.scene_cb, 1
        )
        logger.debug("NlpNode initialized")

    def scene_cb(self, msg: String) -> None:
        scene_json = msg.data
        actions = self._call_llm(scene_json)
        out = String()
        out.data = json.dumps(actions)
        self.pub.publish(out)

    def _call_llm(self, scene_json: str) -> List[Dict]:
        """Call OpenAI with function-calling schema."""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": scene_json}],
                functions=[
                    {"name": "action", "parameters": {"type": "object"}}
                ],
            )
            result = response["choices"][0]["message"]["content"]
            return json.loads(result) if result else []
        except Exception as exc:  # pragma: no cover - network
            logger.error("LLM call failed: %s", exc)
            raise
