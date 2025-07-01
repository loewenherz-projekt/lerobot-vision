# ws/src/lerobot_vision/lerobot_vision/nlp_node.py
"""NLP node using OpenAI API."""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List

import openai
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class NlpNode(Node):
    """Node that generates actions from scene descriptions."""

    def __init__(self) -> None:
        """Initialize the NLP node."""
        super().__init__("nlp_node")
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        self.sub = self.create_subscription(
            String,
            "/robot/vision/scene",
            self._cb,
            10,
        )
        self.pub = self.create_publisher(String, "/robot/vision/actions", 10)

    def _cb(self, msg: String) -> None:
        try:  # pragma: no cover - integration
            actions = self._call_llm(
                msg.data
            )  # pragma: no cover - integration
            result = String(data=json.dumps(actions))  # pragma: no cover
            self.pub.publish(result)  # pragma: no cover
        except Exception as exc:  # pragma: no cover
            logging.error("LLM call failed: %s", exc)

    def _call_llm(self, scene_json: str) -> List[Dict]:
        """Call OpenAI LLM to plan actions.

        Args:
            scene_json: JSON string describing the scene.

        Returns:
            A list of action dictionaries returned by the language model.
        """
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4",
                functions=[{"name": "robot_action", "parameters": {}}],
                function_call="auto",
                messages=[{"role": "user", "content": scene_json}],
            )
            args = resp.choices[0].message.function_call.arguments
            return json.loads(args)
        except Exception as exc:
            logging.error("OpenAI API error: %s", exc)
            return []


def main(args: list[str] | None = None) -> None:
    """Entry point for the ``nlp_node`` executable."""
    rclpy.init(args=args)
    node = NlpNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
