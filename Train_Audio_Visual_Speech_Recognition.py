import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import tensorflow as tf
from models.audio_video_speech_recognition import Lipreading as Model
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

        print('[Epoch %d] train begin ' % epoch)
        train_total_loss = 0
        sess.run(train_init_op)
        i = 0
        while True:
            try:

                loss = model.train(sess)
                train_total_loss += loss
                print('\n   [%d ] Loss: %.4f' % (i, loss))
                if count % 100 == 0:
                    train_summary = model.merge(sess)
                    summary_writer.add_summary(train_summary, count)
                count += 1
                i += 1
            except:
                print('break')
                break
        train_loss = train_total_loss / max(i, 1)
        epoch_summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss", simple_value=train_loss)])
        summary_writer.add_summary(epoch_summary, epoch)
        saver.save(sess, os.path.join(model_dir, model_name + str(epoch)))
        print('[Epoch %d] train end ' % epoch)

        #############################################################

    summary_writer.close()


if __name__ == '__main__':
    tf.app.run()
