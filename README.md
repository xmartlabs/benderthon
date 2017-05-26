# palutils

Set of utilities to work easier with Palladium.

Currently there's only *tf_freezer*, but we are working on more stuff!

Works with Python 2.7.+ and 3.+, with Tensorflow 1.2+.

## tf_freezer

Utility to convert **TensorFlow** checkpoints into minimal frozen **graphs**.

### Usage

To take the checkpoint in `checkpoint_path.ckpt`, whose output is yielded by the node named `Tanh`, and save it to `graph_with_weights.pb`:

```bash
./tf_freezer.py checkpoint_path.ckpt graph_with_weights.pb Tanh 
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
