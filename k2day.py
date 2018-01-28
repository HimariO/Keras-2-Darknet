#! /usr/bin/env python
"""
Reads Darknet19 config and weights and creates Keras model with TF backend.

Currently only supports layers in Darknet19 config.
"""

import argparse
import configparser
import io
import os
from collections import defaultdict
from termcolor import colored

import numpy as np
from keras import backend as K
from keras.layers import (Conv2D, GlobalAveragePooling2D, Input, Lambda,
                          MaxPooling2D)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model as plot

SWAP_LAYER = [
    (19, 20)
]

parser = argparse.ArgumentParser(
    description='Yet Another Darknet To Keras Converter.')
parser.add_argument('config_path', help='Path to Darknet cfg file.')
parser.add_argument('weights_path', help='Path to Darknet weights file.')
parser.add_argument('output_path', help='Path to output Keras model file.')
parser.add_argument(
    '-p',
    '--plot_model',
    help='Plot generated Keras model and save as image.',
    action='store_true')
parser.add_argument(
    '-flcl',
    '--fully_convolutional',
    help='Model is fully convolutional so set input shape to (None, None, 3). '
    'WARNING: This experimental option does not work properly for YOLO_v2.',
    action='store_true')


def unique_config_sections(config_file):
    """Convert all config sections to have unique names.

    Adds unique suffixes to config sections for compability with configparser.
    """
    section_counters = defaultdict(int)
    output_stream = io.StringIO()
    with open(config_file) as fin:
        for line in fin:
            if line.startswith('['):
                section = line.strip().strip('[]')
                _section = section + '_' + str(section_counters[section])
                section_counters[section] += 1
                line = line.replace(section, _section)
            output_stream.write(line)
    output_stream.seek(0)
    return output_stream


def write_dark_conv(conv_bn, F):
    assert conv_bn['conv_w'] != None
    assert conv_bn['conv_b'] != None
    assert conv_bn['bn'] != None

    total_size = len(conv_bn['conv_w']) +  len(conv_bn['conv_b'])

    F.write(conv_bn['conv_b'])
    if type(conv_bn['bn']) is bool:  # last layer not using batch normalization.
        print(colored('[Conv without BN]', color='red'))
    else:
        F.write(conv_bn['bn'])
        total_size += len(conv_bn['bn'])
    F.write(conv_bn['conv_w'])

    conv_bn['conv_w'] = None
    conv_bn['conv_b'] = None
    conv_bn['bn'] = None

    print(colored('[write]', color='green') + ' %d bytes to weight file.' % total_size)
    return conv_bn

def layer_swape(layer_pairs, layer_list):
    """
    Keras not allways order layer as you put them.
    So sometime reorder is nessary before duming weight into Darknet.weights.
    """
    for lp in layer_pairs:
        temp_l = layer_list[lp[0]]
        layer_list[lp[0]] = layer_list[lp[1]]
        layer_list[lp[1]] = temp_l
    return layer_list

# %%
def _main(args):
    config_path = os.path.expanduser(args.config_path)
    weights_path = os.path.expanduser(args.weights_path)
    assert config_path.endswith('.cfg'), '{} is not a .cfg file'.format(
        config_path)
    assert weights_path.endswith(
        '.h5'), '{} is not a .h5 file'.format(weights_path)

    output_path = os.path.expanduser(args.output_path)
    assert output_path.endswith(
        '.weights'), 'output path {} is not a .weights file'.format(output_path)
    output_root = os.path.splitext(output_path)[0]

    # Load weights and config.
    print('Loading weights.')
    weights_file = open(output_path, 'wb')

    weights_header = np.zeros(
        shape=(4, ), dtype='int32'
    )
    weights_header[1] = 1
    weights_header[3] = 32013312  # trained batch_count

    weights_file.write(weights_header.tobytes())

    print('Weights Header: ', weights_header)
    # TODO: Check transpose flag when implementing fully connected layers.
    # transpose = (weight_header[0] > 1000) or (weight_header[1] > 1000)

    print('Parsing Darknet config.')
    unique_config_file = unique_config_sections(config_path)
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read_file(unique_config_file)
    print('-' * 100)
    print(cfg_parser.sections())
    print('-' * 100)

    print('Creating Keras model.')
    yolo_model = load_model(weights_path)

    weight_decay = float(cfg_parser['net_0']['decay']) if 'net_0' in cfg_parser.sections() else 5e-4
    count = 0
    layer_count = 0

    init_dict = lambda :{'conv_w': None, 'conv_b': None, 'bn': None}
    conv_bn = init_dict()
    conv_bn_list = []

    for k_layer in yolo_model.layers:
        if k_layer.count_params() > 0:
            print(k_layer.name)
            k_weights = k_layer.get_weights()
            if 'conv' in k_layer.name:
                assert 0 < len(k_weights) <= 2

                conv_kernel = k_weights[0]
                conv_kernel = np.transpose(conv_kernel, [3, 2, 0, 1])
                # weights_file.write(conv_kernel.tobytes())
                conv_bn['conv_w'] = conv_kernel.tobytes()

                if len(k_weights) == 2:
                    conv_bias = k_weights[1]
                    conv_bn['conv_b'] = conv_bias.tobytes()
                    conv_bn['bn'] = False
                    # conv_bn = write_dark_conv(conv_bn, weights_file)
                    conv_bn_list.append(conv_bn)
                    conv_bn = init_dict()

            elif 'batch' in k_layer.name:
                assert len(k_weights) == 4
                conv_bn['conv_b'] = k_weights[1].tobytes()
                bn_weights = np.array(k_weights[0:1] + k_weights[2:])
                conv_bn['bn'] = bn_weights.tobytes()

                # conv_bn = write_dark_conv(conv_bn, weights_file)
                conv_bn_list.append(conv_bn)
                conv_bn = init_dict()

            print(colored('[load]', color='green') + ' %d bytes to weight file. from {%s} layer' % (len(k_weights), k_layer.name))
            print([i.shape for i in k_weights])
            print('-' * 100)
        else:
            print(colored('[skip]', color='yellow') + ' no params, skip {%s} layer' % k_layer.name)
            print('-' * 100)

    print('Get total %d layers with weight' % len(conv_bn_list))
    conv_bn_list = layer_swape(SWAP_LAYER, conv_bn_list)

    for layer_binary in conv_bn_list:
        write_dark_conv(layer_binary, weights_file)
    # Create and save model.

    print('Saved Keras model to {}'.format(output_path))
    # Check to see if all weights have been read.
    # remaining_weights = len(weights_file.read()) / 4
    weights_file.close()


if __name__ == '__main__':
    _main(parser.parse_args())
