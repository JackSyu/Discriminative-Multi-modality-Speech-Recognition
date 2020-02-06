from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob


class ModelConfig(object):
    def __init__(self):
        # tfrecords的目录，在train和evaluation中必须有
        self.input_file = None

        self.image_format = 'png'

        # 队列中的容量
        self.queue_capcity = 200

        # 队列最短长度
        self.shuffle_min_after_dequeue = 100

        self.num_threads = 8

        self.label_length_name = 'label_length'
        self.frames_name = 'frames'
        self.label_name = 'labels'
#2019.02.28 batch size 由2改为1
        self.batch_size = 25

        self.image_weight = 112
        self.image_height = 112
        self.image_depth = 77
        self.image_channel = 3

        self.initializer_scale = 0.08


        #beam width 由100改为5 2019.2.19 14.40

        #self.beam_width = 100
        #beam width 由5改为1 2019.2.20 9：51
        #self.beam_width = 5
        self.beam_width = 30

        self.train_tfrecord_list = glob.glob(os.path.join('/data-private/nas/lrw/av_enhancement_mix_half', '*data*'))
        self.val_tfrecord_list = glob.glob(os.path.join('/data-private/nas/lrw/av_enhancement_mix', '*data*'))

        self.embedding_size = 256
        self.hidden_units = 512
        self.num_layers = 2
        self.num_units = 512
        self.num_blocks = 6
        self.num_heads = 8


        #2019.2.15maxlen = 15改为 maxlen = 9
        #2019.02.21新加maxlen1：the maximum of source sequence
        #maxlen：the maxium of target sequence


        self.maxlen = 7376
        self.force_teaching_ratio = 0.2
        self.sinusoid = True

        self.dropout_keep_prob = 1.0
        self.dropout_prob = 0.1

        self.scale_embedding = True


class TrainingConfig(object):
    def __init__(self):
        #2019.2.16 0.001改为0.01
        #self.learning_rate = 0.01

        self.learning_rate = 0.0001
        self.learning_rate_decay = 0.5
        self.num_iteration_per_decay = 5000
        self.weight_decay = 0.00005
        self.max_gradient_norm = 5
        self.lambda_weight = 0.2
