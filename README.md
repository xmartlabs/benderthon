# benderthon

Set of utilities to work easier with [Bender](https://github.com/xmartlabs/Bender).

Currently there's support for TensorFlow and Caffe, but we are working on more stuff!

Works on Python 2.7.+ and 3.+, with Tensorflow 1.2+.

To install:

```bash
pip install benderthon
```

TensorFlow is required too. The simplest way to install it is:

```bash
pip install tensorflow
```

There are other ways, see [Installing Tensorflow](https://www.tensorflow.org/install/). Benderthon does not install it
by default to let the usage of a custom installation.

## tf-freeze

Utility to convert **TensorFlow** checkpoints into minimal frozen **graphs**.

### Usage

#### From a checkpoint

To take the checkpoint `checkpoint_path.ckpt`, whose output is yielded by the node named `Tanh`, and save it to `graph_with_weights.pb`:

```bash
benderthon tf-freeze checkpoint_path.ckpt graph_with_weights.pb Tanh
```

#### From code

If you don't have a checkpoint or prefer to run it from code, this is the way to go. This is the same example as above but from code:

```python
from benderthon import tf_freeze

// …

with tf.Session() as sess:
    // …

    tf_freeze.freeze(sess, 'graph_with_weights.pb', ['Tanh'])
```

### Sample

The file `sample.py` contains a network example for MNIST dataset with 2 convolutional layers and 2 dense layers. If you run it, it will generate a minimal protobuf for with the weights frozen to run in Bender in `output/mnist.pb`:

```bash
./sample.py
```

The generated file occupies **half** the original checkpoints (26MB to 13MB).

The script will also generate checkpoints files with prefix `checkpoints/mnist.ckpt`. So you could have generated the protobuf from it:

```bash
benderthon tf-freeze checkpoints/mnist.ckpt output/mnist.pb Prediction
```

You can also get only the graph, which occupies just **13kB**:

```bash
benderthon tf-freeze --no-weights checkpoints/mnist.ckpt output/mnist_only_graph.pb Prediction
```

To save the weights in a separate path for later processing:

```bash
benderthon tf-freeze --only-weights checkpoints/mnist.ckpt weights/ Prediction
```

## caffe-freeze

This module cannot be accessed from the command line utility, it should be used from Python code, importing `benderthon.caffe_freeze`.

You need `caffeflow` package installed first:

```bash
pip install -e git://github.com/xmartlabs/caffeflow.git@4618f89#egg=caffeflow
```

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
