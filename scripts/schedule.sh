#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/train.py experiment=svhn
python src/train.py experiment=cifar10
python src/train.py experiment=cifar10
