from tensorflow.python.layers import core as core_layers
import tensorflow as tf
import numpy as np
from tcn_model import TCN
import utils as utils

def get_conv_weight(name, kshape, wd=0.0005):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape=kshape, initializer=tf.contrib.layers.xavier_initializer())
    if wd != 0:
        weight_decay = tf.nn.l2_loss(var) * wd
        tf.add_to_collection('weightdecay_losses', weight_decay)
    return var


def convS(name, l_input, in_channels, out_channels):
    return tf.nn.bias_add(tf.nn.conv3d(l_input, get_conv_weight(name=name,
                                                                kshape=[1, 3, 3, in_channels, out_channels]),
                                       strides=[1, 1, 1, 1, 1], padding='SAME'),
                          get_conv_weight(name + '_bias', [out_channels], 0))


def convT(name, l_input, in_channels, out_channels):
    return tf.nn.bias_add(tf.nn.conv3d(l_input, get_conv_weight(name=name,
                                                                kshape=[3, 1, 1, in_channels, out_channels]),
                                       strides=[1, 1, 1, 1, 1], padding='SAME'),
                          get_conv_weight(name + '_bias', [out_channels], 0))


# build the bottleneck struction of each block.
class Bottleneck():
    def __init__(self, l_input, inplanes, is_training, planes, stride=1, downsample='', n_s=0, depth_3d=13):

        self.X_input = l_input
        self.downsample = downsample
        self.planes = planes
        self.inplanes = inplanes
        self.is_training = is_training
        self.depth_3d = depth_3d
        self.ST_struc = ('A', 'B', 'C')
        self.len_ST = len(self.ST_struc)
        self.id = n_s
        self.n_s = n_s
        self.ST = list(self.ST_struc)[self.id % self.len_ST]
        self.stride_p = [1, 1, 1, 1, 1]
        if self.downsample != '':
            self.stride_p = [1, 1, 2, 2, 1]
        if n_s < self.depth_3d:
            if n_s == 0:
                self.stride_p = [1, 1, 1, 1, 1]
        else:
            if n_s == self.depth_3d:
                self.stride_p = [1, 2, 2, 2, 1]
            else:
                self.stride_p = [1, 1, 1, 1, 1]

    # P3D has three types of bottleneck sub-structions.
    def ST_A(self, name, x):
        x = convS(name + '_S', x, self.planes, self.planes)
        x = tf.layers.batch_normalization(x, training=self.is_training)
        x = tf.nn.relu(x)
        x = convT(name + '_T', x, self.planes, self.planes)
        x = tf.layers.batch_normalization(x, training=self.is_training)
        x = tf.nn.relu(x)
        return x

    def ST_B(self, name, x):
        tmp_x = convS(name + '_S', x, self.planes, self.planes)
        tmp_x = tf.layers.batch_normalization(tmp_x, training=self.is_training)
        tmp_x = tf.nn.relu(tmp_x)
        x = convT(name + '_T', x, self.planes, self.planes)
        x = tf.layers.batch_normalization(x, training=self.is_training)
        x = tf.nn.relu(x)
        return x + tmp_x

    def ST_C(self, name, x):
        x = convS(name + '_S', x, self.planes, self.planes)
        x = tf.layers.batch_normalization(x, training=self.is_training)
        x = tf.nn.relu(x)
        tmp_x = convT(name + '_T', x, self.planes, self.planes)
        tmp_x = tf.layers.batch_normalization(tmp_x, training=self.is_training)
        tmp_x = tf.nn.relu(tmp_x)
        return x + tmp_x

    def infer(self):
        residual = self.X_input
        if self.n_s < self.depth_3d:
            out = tf.nn.conv3d(self.X_input,
                               get_conv_weight('conv3_{}_1'.format(self.id), [1, 1, 1, self.inplanes, self.planes]),
                               strides=self.stride_p, padding='SAME')
            out = tf.layers.batch_normalization(out, training=self.is_training)

        else:
            param = self.stride_p
            param.pop(1)
            out = tf.nn.conv2d(self.X_input,
                               get_conv_weight('conv2_{}_1'.format(self.id), [1, 1, self.inplanes, self.planes]),
                               strides=param, padding='SAME')
            out = tf.layers.batch_normalization(out, training=self.is_training)

        out = tf.nn.relu(out)
        if self.id < self.depth_3d:
            if self.ST == 'A':
                out = self.ST_A('STA_{}_2'.format(self.id), out)
            elif self.ST == 'B':
                out = self.ST_B('STB_{}_2'.format(self.id), out)
            elif self.ST == 'C':
                out = self.ST_C('STC_{}_2'.format(self.id), out)
        else:
            out = tf.nn.conv2d(out, get_conv_weight('conv2_{}_2'.format(self.id), [3, 3, self.planes, self.planes]),
                               strides=[1, 1, 1, 1], padding='SAME')
            out = tf.layers.batch_normalization(out, training=self.is_training)
            out = tf.nn.relu(out)

        if self.n_s < self.depth_3d:
            out = tf.nn.conv3d(out, get_conv_weight('conv3_{}_3'.format(self.id),
                                                    [1, 1, 1, self.planes, self.planes * 4]),
                               strides=[1, 1, 1, 1, 1], padding='SAME')
            out = tf.layers.batch_normalization(out, training=self.is_training)
        else:
            out = tf.nn.conv2d(out, get_conv_weight('conv2_{}_3'.format(self.id),
                                                    [1, 1, self.planes, self.planes * 4]),
                               strides=[1, 1, 1, 1], padding='SAME')
            out = tf.layers.batch_normalization(out, training=self.is_training)

        if len(self.downsample) == 1:
            residual = tf.nn.conv2d(residual, get_conv_weight('dw2d_{}'.format(self.id),
                                                              [1, 1, self.inplanes, self.planes * 4]),
                                    strides=[1, 2, 2, 1], padding='SAME')
            residual = tf.layers.batch_normalization(residual, training=self.is_training)
        elif len(self.downsample) == 2:
            residual = tf.nn.conv3d(residual, get_conv_weight('dw3d_{}'.format(self.id),
                                                              [1, 1, 1, self.inplanes, self.planes * 4]),
                                    strides=self.downsample[1], padding='SAME')
            residual = tf.layers.batch_normalization(residual, training=self.is_training)
        out += residual
        out = tf.nn.relu(out)

        return out


class make_block():
    def __init__(self, _X, planes, is_training, num, inplanes, cnt, depth_3d=13, stride=1):
        self.input = _X
        self.planes = planes
        self.is_training = is_training
        self.inplanes = inplanes
        self.num = num
        self.cnt = cnt
        self.depth_3d = depth_3d
        self.stride = stride
        if self.cnt < depth_3d:
            if self.cnt == 0:
                stride_p = [1, 1, 1, 1, 1]
            else:
                stride_p = [1, 1, 2, 2, 1]
            if stride != 1 or inplanes != planes * 4:
                self.downsample = ['3d', stride_p]
        else:
            if stride != 1 or inplanes != planes * 4:
                self.downsample = ['2d']

    def infer(self):
        x = Bottleneck(self.input, self.inplanes, self.is_training, self.planes, self.stride, self.downsample,
                       n_s=self.cnt,
                       depth_3d=self.depth_3d).infer()
        self.cnt += 1
        self.inplanes = 4 * self.planes
        for i in range(1, self.num):
            x = Bottleneck(x, self.inplanes, self.is_training, self.planes, n_s=self.cnt,
                           depth_3d=self.depth_3d).infer()
            self.cnt += 1
        return x

class AE_MSR(object):

    def __init__(self, model_config, iterator, train_config):

        self.config = model_config
        self.train_config = train_config
        self.is_training = tf.placeholder_with_default(False, [])
        self.data_format = "channels_last"
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.learning_rate = tf.train.exponential_decay(train_config.learning_rate,
                                                        self.global_step, 2000, 0.5, staircase=True,
                                                        name='learning_rate')
        self.dropout_prob = tf.placeholder_with_default(0.0, [])
        self.iterator = iterator
        self.num_class = 41
        self.build_graph()

    def build_graph(self):
            self.build_inputs()
            self.build_resnet()
            self.build_concat()
            self.build_encoder()
            self.build_eval_model()
            self.build_train()
            self.merged_summary = tf.summary.merge_all('train')

    def build_inputs(self):
        with tf.name_scope('input'), tf.device('/cpu: 0'):
            self.mouth_frames, self.mix_audio_frames, self.audio_frames, self.frames_length = self.iterator.get_next()

    def build_resnet(self):
        def batch_norm_relu(inputs, is_training, data_format):
            """Performs a batch normalization followed by a ReLU."""
            inputs = tf.layers.batch_normalization(
                inputs=inputs, axis=1 if data_format == 'channels_first' else -1,
                center=True, scale=True, training=is_training, fused=True)
            inputs = tf.nn.relu(inputs)
            return inputs

        with tf.variable_scope("p3d"):
            cnt = 0
            conv1_custom = tf.nn.conv3d(self.image_seqs, get_conv_weight('firstconv1', [1, 7, 7, 1, 64]),
                                        strides=[1, 1, 2, 2, 1],
                                        padding='SAME')
            conv1_custom_bn = tf.layers.batch_normalization(conv1_custom, training=self.is_training)
            conv1_custom_bn_relu = tf.nn.relu(conv1_custom_bn)
            print('hh:', conv1_custom.shape)
            x = tf.nn.max_pool3d(conv1_custom_bn_relu, [1, 2, 3, 3, 1], strides=[1, 1, 2, 2, 1], padding='SAME')
            print(x.shape)
            b1 = make_block(x, 64, self.is_training, 3, 64, cnt)
            x = b1.infer()
            print(x.shape)
            cnt = b1.cnt

            x = tf.nn.max_pool3d(x, [1, 2, 1, 1, 1], strides=[1, 1, 1, 1, 1], padding='SAME')
            print(x.shape)
            b2 = make_block(x, 128, self.is_training, 4, 256, cnt, stride=2)
            x = b2.infer()
            print(x.shape)
            cnt = b2.cnt
            x = tf.nn.max_pool3d(x, [1, 2, 1, 1, 1], strides=[1, 1, 1, 1, 1], padding='SAME')
            print(x.shape)
            b3 = make_block(x, 256, self.is_training, 6, 512, cnt, stride=2)
            x = b3.infer()
            print(x.shape)
            cnt = b3.cnt
            x = tf.nn.max_pool3d(x, [1, 2, 1, 1, 1], strides=[1, 1, 1, 1, 1], padding='SAME')

            shape = x.shape.as_list()
            print('x', x.shape)
            x = tf.reshape(x, shape=[-1, shape[2], shape[3], shape[4]])

            x = make_block(x, 512, self.is_training, 3, 1024, cnt, stride=2).infer()
            print('x', x.shape)

            # Caution:make sure avgpool on the input which has the same shape as kernelsize has been setted padding='VALID'
            x = tf.nn.avg_pool(x, [1, 4, 4, 1], strides=[1, 1, 1, 1], padding='VALID')
            print('x', x.shape)
            #x = tf.layers.dense(x, 512, name='p3d_fc')
            self.res_out_1 = tf.reshape(x, [self.config.batch_size, -1, 2048])
            print('res_out', self.res_out_1.shape)

    def build_concat(self):

        with tf.variable_scope("enhance_temporal_conv"):
            x = self.res_out_1
            x = tf.layers.dense(x, 520, name="video_fc_0")
            x = TCN(x, 520, [520, 520, 100], 100, kernel_size=3, dropout=0.1)
            x = self.conv_Block_1D_transpose(x, 520, 2, name='video_conv_1d_3')
            x = TCN(x, 520, [520, 520, 100], 100, kernel_size=3, dropout=0.1)
            x = self.conv_Block_1D_transpose(x, 520, 2, name='video_conv_1d_3')

            x = tf.layers.dense(x, 256, name="video_fc_1")
            self.res_out_video = tf.reshape(x, [self.config.batch_size, -1, 256])
            print('res_out_video', self.res_out_video.shape)

            x = self.mix_audio_frames
            x = tf.layers.dense(x, 520, name="audio_fc_0")
            x = TCN(x, 520, [520, 520, 100], 100, kernel_size=3, dropout=0.1)

            x = tf.layers.dense(x, 256, name="audio_fc_1")
            self.res_out_audio = tf.reshape(x, [self.config.batch_size, -1, 256])
            print('res_out_audio', self.res_out_audio.shape)
            x = tf.concat([self.res_out_video, self.res_out_a``udio], 2)
            self.multi_output = tf.reshape(x, [self.config.batch_size, -1, 512])
            print('multi_output', self.multi_output.shape)
            tf.summary.histogram("res_out_video", self.multi_output, collections=['train'])


    def build_encoder(self):
        """Encoder."""
        with tf.variable_scope('enhance_encoder'):
            # Mask
            x = self.multi_output
            encoder_input_2 = self.mix_audio_frames

            backend_outputs, bi_gru_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=tf.nn.rnn_cell.MultiRNNCell(
                    [self.gru_cell(400) for _ in range(1)]),
                cell_bw=tf.nn.rnn_cell.MultiRNNCell(
                    [self.gru_cell(400) for _ in range(1)]),
                inputs=x, dtype=tf.float32)
            encoder_output_1 = tf.concat([backend_outputs[0], backend_outputs[1]],
                                         axis=-1)
            print('Bi-gru output shape', encoder_output_1.shape)
            encoder_output_1 = tf.layers.dense(encoder_output_1, 600, name="fc_1")
            encoder_output_1 = tf.layers.dense(encoder_output_1, 600, name="fc_2")
            encoder_output_1 = tf.layers.dense(encoder_output_1, 80, activation=lambda x: tf.sigmoid(x), name="fc_mask")
            self.enhanced_audio_frames = encoder_output_1 * encoder_input_2


    def encoder(self):

        with tf.variable_scope("video_encoder") as scope:
            x = self.res_out_1
            encoder_outputs, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=tf.nn.rnn_cell.MultiRNNCell(
                    [self.gru_cell(self.config.num_units) for _ in range(self.config.num_layers)]),
                cell_bw=tf.nn.rnn_cell.MultiRNNCell(
                    [self.gru_cell(self.config.num_units) for _ in range(self.config.num_layers)]),
                inputs=x, dtype=tf.float32, scope=scope)

            # self.encoder_outputs = encoder_outputs
            # self.bi_encoder_state = bi_encoder_state
            # 可以实现变长
            #     inputs=self.res_out, dtype=tf.float32, scope=scope, sequence_length=self.image_length)
            encoder_state = []
            for i in range(self.config.num_layers):
                encoder_state.append(tf.concat([bi_encoder_state[0][i], bi_encoder_state[1][i]], axis=-1))
            self.video_encoder_state = tuple(encoder_state)
            self.video_encoder_out = tf.concat(encoder_outputs, -1)
            tf.summary.histogram('video_encoder_out', self.video_encoder_out, collections=['train'])
            tf.summary.histogram('video_encoder_state', self.video_encoder_state, collections=['train'])
            #print('res_out_video', self.res_out_video.shape)

        with tf.variable_scope("audio_encoder"):
            x = self.enhanced_audio_frames
            x = tf.layers.dense(x, 512, name="audio_fc_0")
            encoder_outputs, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=tf.nn.rnn_cell.MultiRNNCell(
                    [self.gru_cell(self.config.num_units) for _ in range(self.config.num_layers)]),
                cell_bw=tf.nn.rnn_cell.MultiRNNCell(
                    [self.gru_cell(self.config.num_units) for _ in range(self.config.num_layers)]),
                inputs=x, dtype=tf.float32, scope=scope)

            encoder_state = []
            for i in range(self.config.num_layers):
                encoder_state.append(tf.concat([bi_encoder_state[0][i], bi_encoder_state[1][i]], axis=-1))
            self.audio_encoder_state = tuple(encoder_state)
            self.audio_encoder_out = tf.concat(encoder_outputs, -1)
            tf.summary.histogram('audio_encoder_out', self.encoder_out, collections=['train'])
            tf.summary.histogram('audio_encoder_state', self.encoder_state, collections=['train'])

    def decoder(self):
        """Encoder."""
        with tf.variable_scope('decoder') as scope:
            decode_embedding = tf.get_variable('decode_embedding',
                                               [41, self.config.embedding_size],
                                               tf.float32, tf.random_uniform_initializer(-1.0, 1.0))
            seq_embedding = tf.nn.embedding_lookup(decode_embedding, self.tgt_in)

            v_attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units=self.config.num_units * 2,
                memory=self.video_encoder_out)
            a_attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units=self.config.num_units * 2,
                memory=self.audio_encoder_out)

            video_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell=tf.nn.rnn_cell.MultiRNNCell(
                    [self.gru_cell(self.config.num_units * 2) for _ in range(self.config.num_layers)]),
                attention_mechanism=v_attention_mechanism,
                attention_layer_size=self.config.num_units * 2)
            audio_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell=tf.nn.rnn_cell.MultiRNNCell(
                    [self.gru_cell(self.config.num_units * 2) for _ in range(self.config.num_layers)]),
                attention_mechanism=a_attention_mechanism,
                attention_layer_size=self.config.num_units * 2)

            # print("-------",self.processed_decoder_input()[0])
            training_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                inputs=seq_embedding,
                sequence_length=self.label_length,
                embedding=decode_embedding,
                sampling_probability=self.sampling_probability,
                time_major=False,
                name='traininig_helper')

            video_training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=video_decoder_cell,
                helper=training_helper,
                initial_state=video_decoder_cell.zero_state(self.config.batch_size, tf.float32).clone(
                    cell_state=self.video_encoder_state),
                output_layer=core_layers.Dense(41, use_bias=False))

            audio_training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=audio_decoder_cell,
                helper=training_helper,
                initial_state=audio_decoder_cell.zero_state(self.config.batch_size, tf.float32).clone(
                    cell_state=self.audio_encoder_state),
                output_layer=core_layers.Dense(41, use_bias=False))

            training_decoder = tf.tf.concat([video_training_decoder, audio_training_decoder], axis=-1)
            training_decoder = tf.layers.dense(training_decoder, 41, name="decode_fc")

            self.training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,
                                                                                   impute_finished=True,
                                                                                   maximum_iterations=tf.reduce_max(
                                                                                       self.label_length),
                                                                                   scope=scope)
            with tf.variable_scope('logits'):
                self.training_logits = self.training_decoder_output.rnn_output
                # print(self.training_logits.shape)
                tf.summary.histogram('training_logits', self.training_logits, collections=['train'])
                self.sample_id = self.training_decoder_output.sample_id
                tf.summary.histogram('training_sample_id', self.sample_id, collections=['train'])


    def build_eval_model(self):

        print('begin test/val data.')

        self.loss = tf.reduce_mean(tf.reduce_sum(tf.abs(self.train_logits - self.audio_frames), [1, -1]))


    def _residual_block_first(self, x, out_channel, strides, name="unit"):
        in_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)

            # Shortcut connection
            if in_channel == out_channel:
                if strides == 1:
                    shortcut = tf.identity(x)
                else:
                    shortcut = tf.nn.max_pool(x, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')
            else:
                shortcut = self._conv(x, 1, out_channel, strides, name='shortcut')
            # Residual
            x = self._conv(x, 3, out_channel, strides, name='conv_1')
            x = self._bn(x, name='bn_1')
            x = self._relu(x, name='relu_1')
            x = self._conv(x, 3, out_channel, 1, name='conv_2')
            x = self._bn(x, name='bn_2')
            # Merge
            x = x + shortcut
            x = self._relu(x, name='relu_2')
        return x


    def _residual_block(self, x, input_q=None, output_q=None, name="unit"):
        num_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)
            # Shortcut connection
            shortcut = x
            # Residual
            x = self._conv(x, 3, num_channel, 1, input_q=input_q, output_q=output_q, name='conv_1')
            x = self._bn(x, name='bn_1')
            x = self._relu(x, name='relu_1')
            x = self._conv(x, 3, num_channel, 1, input_q=output_q, output_q=output_q, name='conv_2')
            x = self._bn(x, name='bn_2')

            x = x + shortcut
            x = self._relu(x, name='relu_2')
        return x

    def _conv(self, x, filter_size, out_channel, stride, pad="SAME", input_q=None, output_q=None, name="conv"):
        b, h, w, in_channel = x.get_shape().as_list()
        x = utils._conv(x, filter_size, out_channel, stride, pad, input_q, output_q, name)
        f = 2 * (h/stride) * (w/stride) * in_channel * out_channel * filter_size * filter_size
        w = in_channel * out_channel * filter_size * filter_size
        scope_name = tf.get_variable_scope().name + "/" + name
        #self._add_flops_weights(scope_name, f, w)
        return x

    def _fc(self, x, out_dim, input_q=None, output_q=None, name="fc"):
        b, in_dim = x.get_shape().as_list()
        x = utils._fc(x, out_dim, input_q, output_q, name)
        f = 2 * (in_dim + 1) * out_dim
        w = (in_dim + 1) * out_dim
        scope_name = tf.get_variable_scope().name + "/" + name
        #self._add_flops_weights(scope_name, f, w)
        return x

    def _bn(self, x, name="bn"):
        x = utils._bn(x, self.is_training, self.global_step, name)
        # f = 8 * self._get_data_size(x)
        # w = 4 * x.get_shape().as_list()[-1]
        # scope_name = tf.get_variable_scope().name + "/" + name
        # self._add_flops_weights(scope_name, f, w)
        return x

    def _relu(self, x, name="relu"):
        x = utils._relu(x, 0.0, name)
        # f = self._get_data_size(x)
        # scope_name = tf.get_variable_scope().name + "/" + name
        # self._add_flops_weights(scope_name, f, 0)
        return x

    def gru_cell(self, num_units, reuse=False):
        cell = tf.nn.rnn_cell.GRUCell(num_units, reuse=reuse)
        dropout_cell = tf.nn.rnn_cell.DropoutWrapper(cell, self.dropout_prob, self.dropout_prob)
        return dropout_cell

    def build_decode_for_infer(self):
        with tf.variable_scope('decoder', reuse=True) as scope:
            encoder_out_tiled = tf.contrib.seq2seq.tile_batch(self.encoder_out, self.config.beam_width)
            encoder_state_tiled = tf.contrib.seq2seq.tile_batch(self.encoder_state, self.config.beam_width)

            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.config.num_units * 2,
                                                                    memory=encoder_out_tiled)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell=tf.nn.rnn_cell.MultiRNNCell([self.gru_cell(self.config.num_units * 2, reuse=True) for _ in range(self.config.num_layers)]),
                attention_mechanism=attention_mechanism, attention_layer_size=self.config.num_units * 2)

            predicting_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=decoder_cell, embedding=tf.get_variable('decode_embedding'),
                start_tokens=tf.tile(tf.constant([40], dtype=tf.int32), [self.config.batch_size]),
                end_token=39,
                initial_state=decoder_cell.zero_state(self.config.batch_size * self.config.beam_width, tf.float32).clone(
                    cell_state=encoder_state_tiled),
                beam_width=self.config.beam_width,
                output_layer=core_layers.Dense(41, use_bias=False, _reuse=True),
                length_penalty_weight=0.0)
            self.predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=predicting_decoder,
                impute_finished=False,
                maximum_iterations=2 * tf.reduce_max(self.label_length),
                scope=scope)
            with tf.variable_scope('pre_result'):
                self.predicting_ids = self.predicting_decoder_output.predicted_ids[:, :, 0]

    def build_train(self):
        masks = tf.sequence_mask(self.label_length, tf.reduce_max(self.label_length), dtype=tf.float32)   # [?, ?] 动态的掩码
        #self.l2_losses = [self.train_config.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if'kernel' in v.name]
        self.loss = tf.contrib.seq2seq.sequence_loss(
            logits=self.training_logits, targets=self.tgt_out, weights=masks)
                    # + tf.add_n(self.l2_losses)

        # tf.summary.scalar('loss', self.loss, collections=['train'])
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            params = tf.trainable_variables()

            train_params = []
            for param in params:
                # if "conv3d" not in param.name or "resnet" not in param.name:
                train_params.append(param)

            gradients = tf.gradients(self.loss, train_params)

            # for grad in gradients:
            #     tf.summary.histogram(grad.name, grad, collections=['train'])
            clipped_gradients, grad_norm = tf.clip_by_global_norm(gradients, self.train_config.max_gradient_norm)
            # tf.summary.scalar("grad_norm", grad_norm, collections=['train'])
            # tf.summary.scalar("learning_rate", self.train_config.learning_rate, collections=['train'])
            # self.train_op = tf.train.MomentumOptimizer(self.learning_rate, 0.9).apply_gradients(
            self.train_op = tf.train.AdamOptimizer(learning_rate=0.0001).apply_gradients(
                zip(clipped_gradients, train_params), global_step=self.global_step)

    def train(self, sess,sampling_probability):
        _, loss= sess.run([self.train_op, self.loss], feed_dict={self.is_training: True,self.sampling_probability:sampling_probability,
                                                                  self.dropout_prob: self.config.dropout_keep_prob})

        return loss

    def eval(self, sess):
        pred, loss, label = sess.run([self.predicting_ids, self.loss, self.tgt_out], feed_dict={self.is_training: False})
        print("sequence shape", (self.image_seqs).shape)
        return pred, loss, label

    def merge(self, sess):
        summary = sess.run(self.merged_summary)
        return summary


if __name__ == '__main__':
    pass
