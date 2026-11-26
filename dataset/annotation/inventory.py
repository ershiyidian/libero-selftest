import argparse
from pathlib import Path

from events import DEFAULT_RELEASE, write_inventory


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', type=Path, default=DEFAULT_RELEASE)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'outputs/inventory')
    args = parser.parse_args()
    write_inventory(args.release, args.output)
