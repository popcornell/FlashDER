import json

import numpy as np

from flashder import io
from flashder.cli import main

from conftest import random_segments


def test_cli_table_and_json(tmp_path, capsys):
    rng = np.random.default_rng(0)
    io.save_rttm({"a": random_segments(rng, 20, 3)}, tmp_path / "ref.rttm")
    io.save_seglst({"a": random_segments(rng, 20, 3, prefix="h")}, tmp_path / "hyp.json")

    assert main(["-r", str(tmp_path / "ref.rttm"), "-s", str(tmp_path / "hyp.json"), "-c", "0.25", "--per-file"]) == 0
    out = capsys.readouterr().out
    assert "OVERALL" in out and "JER" in out

    main(["-r", str(tmp_path / "ref.rttm"), "-s", str(tmp_path / "hyp.json"), "--json", "-1"])
    data = json.loads(capsys.readouterr().out)
    assert 0 < data["*** OVERALL ***"]["der"] < 10
