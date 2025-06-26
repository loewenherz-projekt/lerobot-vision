from unittest import mock

from lerobot_vision.nlp_node import NlpNode


def test_call_llm(monkeypatch):
    node = NlpNode()
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
    node = NlpNode()
    monkeypatch.setattr(
        "openai.ChatCompletion.create",
        mock.Mock(side_effect=Exception("fail")),
    )
    result = node._call_llm("{}")
    assert result == []
