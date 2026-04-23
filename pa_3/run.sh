#!/usr/bin/env sh
set -e

python src/train.py
python src/test.py