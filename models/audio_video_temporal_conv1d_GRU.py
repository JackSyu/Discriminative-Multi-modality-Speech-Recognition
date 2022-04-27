from tensorflow.python.layers import core as core_layers
import tensorflow as tf
import numpy as np
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

class SpeechEnhancement(object):

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
            conv1_custom = tf.nn.conv3d(self.mouth_frames, get_conv_weight('firstconv1', [1, 7, 7, 1, 64]),
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
            x = tf.layers.dense(x, 1536, name="video_fc_0")
            for i in range(3):
                x = self.conv_Block_1D(x, 1536, 1, name='video_conv_1d_{}'.format(i))

            x = self.conv_Block_1D_transpose(x, 1536, 2, name='video_conv_1d_3')
            for i in range(3):
                x = self.conv_Block_1D(x, 1536, 1, name='video_conv_1d_{}'.format(i+4))

            x = self.conv_Block_1D_transpose(x, 1536, 2, name='video_conv_1d_7')
            for i in range(2):
                x = self.conv_Block_1D(x, 1536, 1, name='video_conv_1d_{}'.format(i+8))

            x = tf.layers.dense(x, 256, name="video_fc_1")
            self.res_out_video = tf.reshape(x, [self.config.batch_size, -1, 256])
            print('res_out_video', self.res_out_video.shape)

            x = self.mix_audio_frames
            x = tf.layers.dense(x, 1536, name="audio_fc_0")
            for i in range(5):
                x = self.conv_Block_1D(x, 1536, 1, name='audio_conv_1d_{}'.format(i))

            x = tf.layers.dense(x, 256, name="audio_fc_1")
            self.res_out_audio = tf.reshape(x, [self.config.batch_size, -1, 256])
            print('res_out_audio', self.res_out_audio.shape)
            x = tf.concat([self.res_out_video, self.res_out_audio], 2)
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
            self.train_logits = encoder_output_1 * encoder_input_2


    def build_eval_model(self):

        print('begin test/val data.')

        self.loss = tf.reduce_mean(tf.reduce_sum(tf.abs(self.train_logits - self.audio_frames), [1, -1]))
        self.dif = tf.reduce_mean(tf.reduce_sum(tf.abs(self.mix_audio_frames - self.audio_frames), [1, -1]))

    def conv_Block_1D(self, x, out_channel, strides, pad="SAME", name='conv_1d'):
        in_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual conv_1d: %s' % scope.name)
            if in_channel == out_channel:
                if strides == 1:
                    shortcut = tf.identity(x)
                else:
                    shortcut = tf.layers.average_pooling1d(x, strides, strides, 'VALID')
            else:
                shortcut = tf.layers.separable_conv1d(x, out_channel, 5, strides=strides, padding=pad, name='shortcut')
            x = tf.layers.separable_conv1d(x, out_channel, 5, strides=strides, padding=pad, name='conv_1')
            x = tf.layers.batch_normalization(x, name='bn_1')
            x = tf.nn.relu(x, name='relu_1')

            x = x + shortcut
        return x

    def conv1d_transpose(self, inputs, filters, kernel_width, stride=4, padding='same', upsample='zeros'):
        if upsample == 'zeros':
            return tf.layers.conv2d_transpose(tf.expand_dims(inputs, axis=1),
                                              filters, (1, kernel_width),
                                              strides=(1, stride),
                                              padding='same')[:, 0]
        elif upsample == 'nn':
            batch_size = tf.shape(inputs)[0]
            _, w, nch = inputs.get_shape().as_list()

            x = inputs

            x = tf.expand_dims(x, axis=1)
            #y = tf.matmul(w, stride)
            x = tf.image.resize_nearest_neighbor(x, [1, w * stride])
            x = x[:, 0]

            return tf.layers.conv1d(x, filters, kernel_width, 1, padding='same')
        else:
            raise NotImplementedError
    def conv_Block_1D_transpose(self, x, out_channel, strides, pad="SAME", name='conv_1d_transpose'):
        in_channel = x.get_shape().as_list()[-1]
        batch = x.get_shape().as_list()[0]
        #batch = tf.shape(x)[0]
        in_width = x.get_shape().as_list()[1]
        output_shape = [batch, -1, out_channel]
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual conv_1d: %s' % scope.name)
            shortcut = self.conv1d_transpose(x, out_channel, 5, stride=strides, padding=pad, upsample='zeros')
            x = tf.layers.batch_normalization(shortcut, name='bn_1')
            x = tf.nn.relu(x, name='relu_1')
            x = x + shortcut
        return x

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
    def build_train(self):

        self.loss = tf.reduce_mean(tf.reduce_sum(tf.abs(self.train_logits - self.audio_frames), [1, -1]))
        self.dif = tf.reduce_mean(tf.reduce_sum(tf.abs(self.mix_audio_frames - self.audio_frames), [1, -1]))
        tf.summary.scalar('loss', self.loss, collections=['train'])
        self.train_op = tf.train.GradientDescentOptimizer(0.0001).minimize(self.loss, global_step=self.global_step, name='train_op')

    def train(self, sess):
        _, diff, loss = sess.run([self.train_op, self.dif, self.loss], feed_dict={self.is_training: True, self.dropout_prob: self.config.dropout_prob})
        return loss, diff

    def eval(self, sess):
        loss, dif, pred = sess.run([self.loss, self.dif, self.train_logits], feed_dict={self.is_training: False})
        return loss, dif

    def merge(self, sess):
        summary = sess.run(self.merged_summary)
        return summary


if __name__ == '__main__':
    pass
