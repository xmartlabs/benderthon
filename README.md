# palutils

Set of utilities to work easier with Palladium.

Currently there's only *tf_freezer*, but we are working on more stuff!

Works on Python 2.7.+ and 3.+, with Tensorflow 1.2+.

To install:

```bash
pip install plutils
```

## tf_freezer

Utility to convert **TensorFlow** checkpoints into minimal frozen **graphs**.

### Usage

To take the checkpoint in `checkpoint_path.ckpt`, whose output is yielded by the node named `Tanh`, and save it to `graph_with_weights.pb`:

```bash
palutils freeze checkpoint_path.ckpt graph_with_weights.pb Tanh
```

### Sample

The file `sample.py` contains a 2-hidden layer network example for MNIST dataset. If you run it, it will generate checkpoints files with prefix `checkpoints/mnist.ckpt`:

```bash
./sample.py
```

Then you can get a minimal protobuf version with the weights frozen:

```bash
palutils freeze checkpoints/mnist.ckpt output/mnist.pb Prediction
```

The generated file occupies **half** the original checkpoints (14.4MB to 7.2MB).


You can also get only the graph, which occupies just **10.8kB**:

```bash
palutils freeze --no-weights checkpoints/mnist.ckpt output/mnist_only_graph.pb Prediction
```

## License

```
Copyright 2017 Xmartlabs SRL.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
