#!/usr/bin/env python3
"""Plan selective regeneration from hashes; never modify a source run."""
import argparse
import json

from tricompose_v11.bridge_delta import write_bridge_delta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old-run", "new-run", "output-root", "run-id"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    try:
        result = write_bridge_delta(**vars(args))
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
