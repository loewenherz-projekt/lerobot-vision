import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "ws", "src", "lerobot_vision"
    ),
)  # noqa: E402

import json  # noqa: E402
import types  # noqa: E402

# stub rclpy and std_msgs
rclpy = types.ModuleType("rclpy")
node_mod = types.ModuleType("rclpy.node")


class DummyPub:
    def __init__(self):
        self.last = None

    def publish(self, msg):
        self.last = msg


class Node:
    def __init__(self, name):
        self.pub = DummyPub()

    def create_publisher(self, *args, **kwargs):
        return self.pub

    def create_subscription(self, *args, **kwargs):
        pass


node_mod.Node = Node
rclpy.node = node_mod
sys.modules["rclpy"] = rclpy
sys.modules["rclpy.node"] = node_mod

std_msgs = types.ModuleType("std_msgs.msg")


class String:
    def __init__(self):
        self.data = ""


std_msgs.String = String
sys.modules["std_msgs.msg"] = std_msgs

openai = types.ModuleType("openai")


class ChatCompletion:
    @staticmethod
    def create(**kwargs):
        return {"choices": [{"message": {"content": '[{"foo": "bar"}]'}}]}


openai.ChatCompletion = ChatCompletion
sys.modules["openai"] = openai

from lerobot_vision.nlp_node import NlpNode  # noqa: E402


def test_call_llm():
    node = NlpNode()
    result = node._call_llm('{"scene": 1}')
    assert result == [{"foo": "bar"}]


def test_scene_cb(monkeypatch):
    node = NlpNode()
    monkeypatch.setattr(node, "_call_llm", lambda s: [1])
    msg = String()
    msg.data = "{}"
    node.scene_cb(msg)
    assert json.loads(node.pub.last.data) == [1]
