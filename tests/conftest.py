from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent.parent / "data" / "the-verdict.txt"


@pytest.fixture(scope="session")
def verdict_text():
    if not DATA.exists():
        pytest.skip("run scripts/fetch_data.py first")
    return DATA.read_text(encoding="utf-8")
