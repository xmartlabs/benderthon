#!/usr/bin/env bash

set -e

./sample.py
palutils/cmdline.py freeze --no-weights checkpoints/mnist.ckpt output/g2.pb Prediction
diff test/g.pb output/g2.pb
