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

tf.flags.DEFINE_string('vocab_path', '/home/jack/model/original_english_word_count.txt', 'dictionary path')

def main(unused_argv):

    vocab = Vocabulary(FLAGS.vocab_path)

    model_path = '/data-private/nas/src/Multi_Model/AE_MSR2019:10:02:01:29:18/ckp30'
    model_name = 'ckp'
    model_config = ModelConfig()
    train_config = TrainingConfig()

    val_dataset = build_dataset(model_config.val_tfrecord_list, batch_size=model_config.batch_size, is_training=False)
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    val_init_op = iterator.make_initializer(val_dataset)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.intra_op_parallelism_threads = 24
    config.inter_op_parallelism_threads = 24

    model = Model(model_config=model_config, iterator=iterator, train_config=train_config)

    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    vL = tf.trainable_variables()
    r_saver = tf.train.Saver(var_list=vL, max_to_keep=FLAGS.NUM_EPOCH)
    r_saver.restore(sess, model_path)

    print('Model compiled')

    sess.run(val_init_op)
    val_pairs = []
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
        except:
            break
        counts, cer = cer_s(val_pairs)
        print('Current error rate is : %.4f' % cer) 
        
        #############################################################


if __name__ == '__main__':
    tf.app.run()
