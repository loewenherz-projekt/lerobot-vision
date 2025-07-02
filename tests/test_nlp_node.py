# tests/test_nlp_node.py
from unittest import mock

from lerobot_vision.nlp_node import NlpNode


def _make_node(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    return NlpNode()


def test_call_llm(monkeypatch):
    node = _make_node(monkeypatch)
    fake_resp = mock.Mock(
        choices=[
            mock.Mock(
                message=mock.Mock(
                    function_call=mock.Mock(arguments='[{"action": "test"}]')
                )
            )
        ]
    )
    monkeypatch.setattr(
        "openai.ChatCompletion.create", mock.Mock(return_value=fake_resp)
    )
    result = node._call_llm("{}")
    assert result == [{"action": "test"}]


def test_call_llm_failure(monkeypatch):
    node = _make_node(monkeypatch)
    monkeypatch.setattr(
        "openai.ChatCompletion.create",
        mock.Mock(side_effect=Exception("fail")),
    )
    result = node._call_llm("{}")
    assert result == []


def test_nlp_cb(monkeypatch):
    node = _make_node(monkeypatch)
    node._call_llm = mock.Mock(return_value=[{"foo": 1}])
    node.pub = mock.Mock()
    msg = mock.Mock(data="{}")
    node._cb(msg)
    node.pub.publish.assert_called_once()
