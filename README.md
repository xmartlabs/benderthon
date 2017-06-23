# benderthon

Set of utilities to work easier with [Bender](https://github.com/xmartlabs/Bender).

Currently there's only *tf-freeze*, but we are working on more stuff!

Works on Python 2.7.+ and 3.+, with Tensorflow 1.2+.

To install:

```bash
pip install benderthon
```

## tf-freeze

Utility to convert **TensorFlow** checkpoints into minimal frozen **graphs**.

### Usage

To take the checkpoint in `checkpoint_path.ckpt`, whose output is yielded by the node named `Tanh`, and save it to `graph_with_weights.pb`:

```bash
benderthon tf-freeze checkpoint_path.ckpt graph_with_weights.pb Tanh
```

### Sample

The file `sample.py` contains a network example for MNIST dataset with 2 convolutional layers and 2 dens layers. If you run it, it will generate checkpoints files with prefix `checkpoints/mnist.ckpt`:

```bash
./sample.py
```

Then you can get a minimal protobuf version with the weights frozen:

```bash
benderthon tf-freeze checkpoints/mnist.ckpt output/mnist.pb Prediction
```

The generated file occupies **half** the original checkpoints (26MB to 13MB).


You can also get only the graph, which occupies just **13kB**:

```bash
benderthon tf-freeze --no-weights checkpoints/mnist.ckpt output/mnist_only_graph.pb Prediction
```

To save the weights in a separate path for later processing:

```bash
benderthon tf-freeze --only-weights checkpoints/mnist.ckpt weights/ Prediction
```

## caffe-freeze

You need `caffeflow` first:

```bash
pip install -e git://github.com/xmartlabs/caffeflow.git@2c8c9d45b164fece0629f27a0a201517d3918b5a#egg=caffeflow
```

It's recommended to have Caffe installed too, with its Python extensions. Note that it only supports Python 2.

## Development

This utility is under development and the API **is not stable**. So, do not heavily rely on it.

To install locally you should do ```./setup.py install```, but first have [pandoc](http://pandoc.org/) and [pypandoc](https://github.com/bebraw/pypandoc) installed.

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
