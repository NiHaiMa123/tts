from __future__ import annotations

from character_tts.app.download import plan_download


def test_cached_short_circuits():
    plan = plan_download(env={}, cached=True)
    assert [a.kind for a in plan] == ["local_cache"]


def test_mirror_then_proxy_order():
    plan = plan_download(env={})
    kinds = [a.kind for a in plan]
    assert kinds == ["mirror", "proxy"]
    assert plan[0].env["HF_ENDPOINT"]
    assert plan[1].env["HTTP_PROXY"].endswith(":7897")
    assert plan[1].env["HTTPS_PROXY"].endswith(":7897")


def test_existing_mirror_reused_first():
    env = {"HF_ENDPOINT": "https://my-mirror.example", "HF_HOME": "/cache"}
    plan = plan_download(env=env)
    assert plan[0].kind == "existing_mirror"
    assert plan[1].kind == "mirror"
    assert plan[-1].kind == "proxy"


def test_no_double_proxy_when_user_already_has_it():
    env = {"HTTP_PROXY": "http://127.0.0.1:7897"}
    plan = plan_download(env=env)
    # existing_mirror only triggers on HF_ENDPOINT; proxy still last
    assert plan[-1].kind == "proxy"


def test_mirror_can_be_disabled():
    plan = plan_download(env={}, mirror_endpoint=None)
    assert [a.kind for a in plan] == ["proxy"]
