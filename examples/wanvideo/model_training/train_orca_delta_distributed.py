#!/usr/bin/env python3
"""Compatibility entrypoint for existing launch commands and running supervisors.

New runs use train.py (YAML) or train_orca.py (distributed worker).
Keep exported names available to DataLoader workers spawned by older runs.
"""
from train_orca import *  # noqa: F401,F403


if __name__ == '__main__':
    main(parse_args())
