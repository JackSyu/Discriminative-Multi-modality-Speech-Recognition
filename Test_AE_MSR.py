import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import tensorflow as tf
from models.AE-MSR_tcn_GRU import AE_MSR as Model
from input import build_dataset, Vocabulary
import datetime
from configuration import ModelConfig, TrainingConfig
import numpy as np
from statistic import cer_s


FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string('vocab_path', '/home/jack/model/w22lmc/original_english_word_count.txt', 'dictionary path')

tf.flags.DEFINE_integer('NUM_EPOCH', 100, 'epoch次数')


def main(unused_argv):

    vocab = Vocabulary(FLAGS.vocab_path)
    #vocab.id_to_word['-1'] = -1

    model_dir = 'AVSR_LAB_transformer_1word' + datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    model_name = 'ckp'
    model_config = ModelConfig()
    train_config = TrainingConfig()

    train_dataset = build_dataset(model_config.train_tfrecord_list, batch_size=model_config.batch_size, shuffle=True)
    val_dataset = build_dataset(model_config.val_tfrecord_list, batch_size=model_config.batch_size, is_training=False)
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    train_init_op = iterator.make_initializer(train_dataset)
    val_init_op = iterator.make_initializer(val_dataset)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.intra_op_parallelism_threads = 24
    config.inter_op_parallelism_threads = 24

    model = Model(model_config=model_config, iterator=iterator, train_config=train_config)

    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(max_to_keep=FLAGS.NUM_EPOCH)

    summary_writer = tf.summary.FileWriter('logs_no_mh/asr_transformer' +
                                                    datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S'),
                                                    graph=sess.graph)

    print('Model compiled')

    count = 0
    for epoch in range(FLAGS.NUM_EPOCH):

        print('Epoch %d] eval begin' % epoch)
        val_total_loss = 0
        sess.run(val_init_op)
        val_pairs = []
        i = 0
        if epoch > -1:
            # for jjjj in range(118550):
            while True:
                try:
                    out_indices, loss1, y = model.eval(sess)
                    print('pred: ', out_indices)
                    print('ground truth: ', y)
                    print('loss: ', loss1)
                    val_total_loss += loss1
                    print('\n   [%d ]' % (i))
                    for j in range(len(y)):
                        unpadded_out = None
                        if 1 in out_indices[j]:
                            idx_1 = np.where(out_indices[j] == 1)[0][0]
                            unpadded_out = out_indices[j][:idx_1]
                        else:
                            unpadded_out = out_indices[j]
                        idx_1 = np.where(y[j] == 1)[0][0]
                        unpadded_y = y[j][:idx_1]
                        predic = ''.join([vocab.id_to_word[k] for k in unpadded_out])
                        label = ''.join([vocab.id_to_word[i] for i in unpadded_y])
                        val_pairs.append((predic, label))
                    i += 1
                except:
                    break
            avg_loss = val_total_loss / max(i, 1)
            print("avg_loss", avg_loss)
            counts, cer = cer_s(val_pairs)

            summary = tf.Summary(value=[tf.Summary.Value(tag="cer", simple_value=cer),
                                        tf.Summary.Value(tag="val_loss", simple_value=avg_loss)])
            summary_writer.add_summary(summary, epoch)

            print('Current error rate is : %.4f' % cer)
        print('Epoch %d] eval end' % epoch)

        #############################################################

    summary_writer.close()


if __name__ == '__main__':
    tf.app.run()
