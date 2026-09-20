"""Validate cart topology, mass/inertia, wheel geometry and attachment offline."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/imgo2_rl/imgo2_rl/assets"))
from cart_model import read_cart_model  # noqa: E402

DEFAULT_CART = ROOT.parent / "imgo2_description/cart/cart.urdf"


def check_cart(path: Path):
    model = read_cart_model(path)
    return {**model, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=Path(os.environ.get("IMGO2_CART_URDF_PATH", str(DEFAULT_CART))).expanduser())
    args = parser.parse_args()
    try:
        print(json.dumps(check_cart(args.urdf), indent=2))
        print("PASS: cart model (offline only; no simulation)")
    except (ValueError, OSError, ET.ParseError, AttributeError, TypeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
