from unittest.mock import MagicMock

import pytest

try:
    from std_msgs.msg import String
except Exception:
    pytest.skip("ROS messages not available", allow_module_level=True)

from lerobot_vision.nlp_node import NlpNode


def test_call_llm(monkeypatch):
    node = NlpNode()
    monkeypatch.setattr(
        "openai.ChatCompletion.create", MagicMock(return_value=[{"act": "run"}])
    )
    result = node._call_llm("{}")
    assert result == [{"act": "run"}]


def test_callback(monkeypatch):
    node = NlpNode()
    pub = MagicMock()
    monkeypatch.setattr(node, "pub", pub)
    monkeypatch.setattr(node, "_call_llm", MagicMock(return_value=[{"act": "x"}]))
    node._cb(String(data="{}"))
    pub.publish.assert_called()
