import pytest


def test_llmflux_version_flag_prints_and_exits(capsys):
    from llmflux.cli import main

    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])

    assert excinfo.value.code == 0
    out = capsys.readouterr().out.strip()
    assert out.startswith("llmflux ")


def test_llmflux_short_version_flag_prints_and_exits(capsys):
    from llmflux.cli import main

    with pytest.raises(SystemExit) as excinfo:
        main(["-V"])

    assert excinfo.value.code == 0
    out = capsys.readouterr().out.strip()
    assert out.startswith("llmflux ")

