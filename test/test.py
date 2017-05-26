#!/usr/bin/env python
import tf_freezer

tf_freezer.freeze_from_checkpoint('/home/santiago/Escritorio/checkpoints/fns.ckpt', 'g_and_w.pb', 'Tanh')

# tf_freezer.save_graph_only_from_checkpoint('/home/santiago/Escritorio/checkpoints/fns.ckpt', 'g.pb', 'Tanh')
