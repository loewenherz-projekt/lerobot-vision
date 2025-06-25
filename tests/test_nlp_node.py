# File: tests/test_nlp_node.py
import sys
import types
from pathlib import Path
from unittest import mock

sys.path.append(str(Path(__file__).resolve().parents[1]))  # noqa: E402
import json  # noqa: E402


def test_call_llm(monkeypatch):
    class FakeNode:
        def __init__(self, name):
            pass

        def create_publisher(self, *a, **kw):
            return mock.Mock()

        def create_subscription(self, *a, **kw):
            return mock.Mock()

    rclpy_node_module = types.SimpleNamespace(Node=FakeNode)
    rclpy_stub = types.SimpleNamespace(node=rclpy_node_module)
    monkeypatch.setitem(sys.modules, "rclpy", rclpy_stub)
    monkeypatch.setitem(sys.modules, "rclpy.node", rclpy_node_module)
    std_msgs_stub = types.SimpleNamespace(
        msg=types.SimpleNamespace(String=type("String", (), {}))
    )
    monkeypatch.setitem(sys.modules, "std_msgs", std_msgs_stub)
    monkeypatch.setitem(sys.modules, "std_msgs.msg", std_msgs_stub.msg)

    from ws.src.lerobot_vision.lerobot_vision.nlp_node import NlpNode

    node = NlpNode()
    fake_resp = {"choices": [{"message": {"content": json.dumps([{"a": 1}])}}]}
    monkeypatch.setattr(
        "ws.src.lerobot_vision.lerobot_vision.nlp_node.openai",
        mock.MagicMock(),
    )
    target = (
        "ws.src.lerobot_vision.lerobot_vision.nlp_node.openai."
        "ChatCompletion.create"
    )
    monkeypatch.setattr(target, lambda **kwargs: fake_resp)
    res = node._call_llm("{}")
    assert res == [{"a": 1}]
