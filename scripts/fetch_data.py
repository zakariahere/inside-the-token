"""Download the-verdict.txt (Raschka section 2.2) into data/."""
import urllib.request
from pathlib import Path

URL = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/"
       "ch02/01_main-chapter-code/the-verdict.txt")
DEST = Path(__file__).resolve().parent.parent / "data" / "the-verdict.txt"

if __name__ == "__main__":
    DEST.parent.mkdir(exist_ok=True)
    if DEST.exists():
        print(f"already present: {DEST}")
    else:
        urllib.request.urlretrieve(URL, DEST)
        print(f"saved {DEST} ({DEST.stat().st_size} bytes)")
