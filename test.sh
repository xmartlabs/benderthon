#!/usr/bin/env bash

set -e

./sample.py
./tf_freezer.py --no-weights checkpoints/mnist.ckpt output/g2.pb Prediction
diff test/g.pb output/g2.pb
