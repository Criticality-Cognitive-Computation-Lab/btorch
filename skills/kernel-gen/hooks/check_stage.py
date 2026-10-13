"""Run the same portable evidence gate from an agent hook or a terminal."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_model import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(["gate", *sys.argv[1:]]))
