import pickle
import os
import time
import tensorflow as tf
from sys import path
from os import getcwd
path.append(getcwd() + "/a-eye_segmentation/3DUnet_TF1/model")
from network import Unet_3D
import generate_h5


def configure():
    flags = tf.app.flags
    # data
    flags.DEFINE_string('raw_data_dir', 'a-eye_segmentation/3DUnet_TF1/Data/Test/', 'Name of raw data file(s)')
    flags.DEFINE_string('data_dir', 'a-eye_segmentation/3DUnet_TF1/h5_test/', 'Name of data file(s)')
    flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
    flags.DEFINE_boolean('aug_flip', True, 'Training data augmentation: flip. Extra 3 datasets.')
    flags.DEFINE_boolean('aug_rotate', True, 'Training data augmentation: rotate. Extra 9 datasets.')
    flags.DEFINE_integer('validation_id', 10, '1-10, which subject is used for validation')
    flags.DEFINE_integer('patch_size', 32, 'patch size')
    flags.DEFINE_integer('overlap_stepsize', 16, 'overlap stepsize when performing testing')
    flags.DEFINE_string('data_type', '3D', '2D data or 3D data')
    flags.DEFINE_integer('batch', 1, 'training batch size')
    flags.DEFINE_integer('channel', 2, 'channel size')
    flags.DEFINE_integer('depth', 32, 'depth size') # should be equal to patch_size
    flags.DEFINE_integer('height', 32, 'height size') # should be equal to patch_size
    flags.DEFINE_integer('width', 32, 'width size') # should be equal to patch_size
    # Debug
    flags.DEFINE_string('logdir', 'a-eye_segmentation/3DUnet_TF1/model/logdir', 'Log dir')
    flags.DEFINE_string('modeldir', 'a-eye_segmentation/3DUnet_TF1/model/modeldir', 'Model dir')
    flags.DEFINE_string('savedir', 'a-eye_segmentation/3DUnet_TF1/model/result', 'Result saving directory')
    flags.DEFINE_string('model_name', 'model', 'Model file name')
    flags.DEFINE_integer('reload_step', 0, 'Reload step to continue training')
    flags.DEFINE_integer('test_step', 25000, 'Test or predict model at this step')
    flags.DEFINE_integer('random_seed', int(time.time()), 'random seed')
    # network architecture
    flags.DEFINE_integer('network_depth', 4, 'network depth for U-Net')
    flags.DEFINE_integer('class_num', 5 , 'output class number')
    flags.DEFINE_integer('start_channel_num', 32,
                         'start number of outputs for the first conv layer')
    flags.DEFINE_string(
        'conv_name', 'conv',
        'Use which conv op in decoder: conv or ipixel_cl')
    flags.DEFINE_string(
        'deconv_name', 'deconv',
        'Use which deconv op in decoder: deconv, pixel_dcl, ipixel_dcl')
    flags.DEFINE_string(
        'action', 'concat',
        'Use how to combine feature maps in pixel_dcl and ipixel_dcl: concat or add')
	# Dense Transformer Networks
    flags.DEFINE_boolean('add_dtn', False, 'add Dense Transformer Networks or not')
    flags.DEFINE_integer('dtn_location', 2, 'The Dense Transformer Networks location')
    flags.DEFINE_integer('control_points_ratio', 2,
        'Setup the ratio of control_points comparing with the Dense transformer networks input size')    
    # fix bug of flags
    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS

def main():
    conf = configure()
    model = Unet_3D(tf.Session(), conf)
    generate_h5.build_h5_dataset(conf.raw_data_dir,conf.data_dir)
    getattr(model, 'predict')()
    return 'DONE!!'

if __name__ == '__main__':
    # configure which gpu or cpu to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tf.app.run()