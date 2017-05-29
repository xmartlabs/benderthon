#!/usr/bin/env python
import argparse

from palutils import tf_freezer


def main():
    parser = argparse.ArgumentParser(description="Set of utilities for Palladium.")
    subparsers = parser.add_subparsers()

    parser_freeze = subparsers.add_parser('freeze',
                                          help="Utility to easily convert TensorFlow checkpoints into minimal frozen "
                                               "graphs in binary protobuf format.")
    parser_freeze.add_argument('input_checkpoint', help="checkpoint path to load")
    parser_freeze.add_argument('output_file', help="file to save the binary protobuf graph")
    parser_freeze.add_argument('output_node_names', help="the name of the output nodes, comma separated")
    parser_freeze.add_argument('--no-weights', action='store_true',
                               help="indicate that the variables are not converted to consts")
    args = parser.parse_args()
    if args.no_weights:
        tf_freezer.save_graph_only_from_checkpoint(args.input_checkpoint, args.output_file, args.output_node_names)
    else:
        tf_freezer.freeze_from_checkpoint(args.input_checkpoint, args.output_file, args.output_node_names)


if __name__ == '__main__':
    main()
