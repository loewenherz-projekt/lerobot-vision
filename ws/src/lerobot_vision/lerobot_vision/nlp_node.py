# File: ws/src/lerobot_vision/lerobot_vision/nlp_node.py
"""Node responsible for turning scenes into robot actions using an LLM."""

from typing import List, Dict
import json
import logging

from rclpy.node import Node
from std_msgs.msg import String

try:
    import openai
except ImportError:  # pragma: no cover - external dependency
    openai = None

logger = logging.getLogger(__name__)


class NlpNode(Node):
    """ROS2 node to call an LLM for planning actions."""

    def __init__(self) -> None:
        super().__init__("nlp_node")
        self.pub = self.create_publisher(String, "/robot/vision/actions", 10)
        self.create_subscription(
            String, "/robot/vision/scene", self.scene_cb, 10
        )
        logger.debug("NlpNode initialized")

    def scene_cb(self, msg: String) -> None:
        scene_json = msg.data
        actions = self._call_llm(scene_json)
        out = String()
        out.data = json.dumps(actions)
        self.pub.publish(out)

    def _call_llm(self, scene_json: str) -> List[Dict]:
        """Call OpenAI ChatCompletion with function calling schema."""
        if openai is None:
            logger.error("openai package not available")
            return []
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": scene_json}],
                functions=[{"name": "plan", "parameters": {"type": "object"}}],
            )
            actions = json.loads(response["choices"][0]["message"]["content"])
            return actions
        except Exception as exc:  # pragma: no cover - network
            logger.error("LLM call failed: %s", exc)
            raise
