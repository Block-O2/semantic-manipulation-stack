"""Mechanically replace GIF placeholders in the single-file explainer."""

from __future__ import annotations

import argparse
import base64
from pathlib import Path
import re


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html", type=Path)
    parser.add_argument("--bottle", type=Path, required=True)
    parser.add_argument("--tissue", type=Path, required=True)
    parser.add_argument("--draw", type=Path, required=True)
    args = parser.parse_args()
    text = args.html.read_text()
    sources = (("BOTTLE", args.bottle), ("TISSUE", args.tissue), ("DRAW", args.draw))
    encoded = [base64.b64encode(path.read_bytes()).decode("ascii") for _, path in sources]
    placeholders = ["{{" + name + "_GIF}}" for name, _ in sources]
    if all(placeholder in text for placeholder in placeholders):
        for placeholder, payload in zip(placeholders, encoded, strict=True):
            text = text.replace(placeholder, payload)
    else:
        pattern = re.compile(r'(?<=src="data:image/gif;base64,)[A-Za-z0-9+/=]+(?=")')
        if len(pattern.findall(text)) != 3:
            raise RuntimeError("Expected three GIF placeholders or three embedded GIFs")
        payloads = iter(encoded)
        text = pattern.sub(lambda _: next(payloads), text)
    args.html.write_text(text)
    print({"html": str(args.html.resolve()), "bytes": args.html.stat().st_size})


if __name__ == "__main__":
    main()
