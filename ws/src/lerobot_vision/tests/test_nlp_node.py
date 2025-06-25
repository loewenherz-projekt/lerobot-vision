from unittest import mock

import rclpy
from std_msgs.msg import String

from lerobot_vision.nlp_node import NlpNode


def test_call_llm(monkeypatch):
    node = NlpNode()
    fake_resp = {"choices": [{"message": {"content": [{"action": "test"}]}}]}
    monkeypatch.setattr(
        "openai.ChatCompletion.create", mock.Mock(return_value=fake_resp)
    )
    result = node._call_llm("{}")
    assert result == [{"action": "test"}]
