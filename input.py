from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
def image_left_right_flip(image):
    return tf.image.flip_left_right(image)


def video_left_right_flip(video):
    return tf.map_fn(image_left_right_flip, video)

class Vocabulary(object):
    '''vocabulary wrapper'''

    def __init__(self, dictionary):
        self.id_to_word, self.word_to_id = self._extract_charater_vocab(dictionary)

    def _extract_charater_vocab(self, dictionary):
        '''get label_to_text'''
        words = []
        with open(dictionary, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                words.append(line.strip('\n').split(' ')[0])
        #words = sorted(words)
        # print(words)

        special_words = ['<PAD>', '<EOS>', '<BOS>']
        # special_words = []
        int_to_vocab = {idx: word for idx, word in enumerate(special_words + words)}
        vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
        # print(int_to_vocab)
        return int_to_vocab, vocab_to_int


def build_dataset(filenames, batch_size, buffer_size=40000, repeat=None, num_threads=8, shuffle=False, is_training=True):
    dataset = tf.data.TFRecordDataset(filenames)  # 建立用于读取tfrecord的datastet
    if is_training:
        dataset = dataset.map(_train_parse_function, num_parallel_calls=24)
    else:
        dataset = dataset.map(_val_parse_function, num_parallel_calls=24)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    # 固定长度
    dataset = dataset.padded_batch(batch_size, padded_shapes=([None, 112, 112, 1], [None, 80], [None, 80], []),
    # 变长
    # dataset = dataset.padded_batch(batch_size, padded_shapes=([None, 112, 112, 3], [None], [None], []),
                                   padding_values=(0.0, 0.0, 0.0, 0))
    # if repeat != None:
    # dataset = dataset.repeat()

    return dataset


def _train_parse_function(example_proto):
    context_features = {
        "label_length": tf.FixedLenFeature([], dtype=tf.int64),
        "label_lrw500": tf.FixedLenFeature([], dtype=tf.int64),
        "frames_length": tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        "mouth_frames": tf.FixedLenSequenceFeature([], dtype=tf.string),
        "mix_audio_frames": tf.FixedLenSequenceFeature([], dtype=tf.string),
        "audio_frames": tf.FixedLenSequenceFeature([], dtype=tf.string),
        "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=example_proto,
        context_features=context_features,
        sequence_features=sequence_features
    )

    #video
    frames = sequence_parsed["mouth_frames"]
    #labels = sequence_parsed["labels"]
    #labels = tf.cast(labels, dtype=tf.int32)
    frames = tf.decode_raw(frames, np.uint8)
    # print("frames",frames)

    # 定长
    # frames = tf.reshape(frames, (77, 112, 112, 3))
    # 变长
    frames = tf.reshape(frames, (-1, 112, 112, 1))
    frames = tf.image.convert_image_dtype(frames, dtype=tf.float32)

    sample = tf.random_uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)
    option = tf.less(sample, 0.5)
    frames = tf.cond(option,
                     lambda: tf.map_fn(image_left_right_flip, frames),
                     lambda: tf.map_fn(tf.identity, frames))

    mouth_frames = tf.subtract(frames, 0.5)
    mouth_frames = tf.multiply(mouth_frames, 2)

    #audio

    audio_frames = sequence_parsed["audio_frames"]
    #labels = sequence_parsed["labels"]

    audio_frames = tf.decode_raw(audio_frames, np.float32)

    # 定长
    audio_frames = tf.reshape(audio_frames, (-1, 80))
    # 不定长
    # frames = tf.reshape(frames, (label_length, 112, 112, 3))
    audio_frames = tf.image.convert_image_dtype(audio_frames, dtype=tf.float32)
    # frames = tf.cast(frames, dtype=tf.float32)

    audio_frames = tf.subtract(audio_frames, 0.5)
    audio_frames = tf.multiply(audio_frames, 2)

    #mix_audio
    mix_audio_frames = sequence_parsed["mix_audio_frames"]
    # labels = sequence_parsed["labels"]

    mix_audio_frames = tf.decode_raw(mix_audio_frames, np.float32)

    # 定长
    mix_audio_frames = tf.reshape(mix_audio_frames, (-1, 80))
    # 不定长
    # frames = tf.reshape(frames, (label_length, 112, 112, 3))
    mix_audio_frames = tf.image.convert_image_dtype(mix_audio_frames, dtype=tf.float32)
    # frames = tf.cast(frames, dtype=tf.float32)

    mix_audio_frames = tf.subtract(mix_audio_frames, 0.5)
    mix_audio_frames = tf.multiply(mix_audio_frames, 2)
    #labels = tf.cast(labels, dtype=tf.int32)

    # frame_list = tf.unstack(frames)
    # for i in range(len(frame_list)):
    #     frame_list[i] = tf.image.random_flip_left_right(frame_list[i])
    # norm_frames = tf.stack(frame_list)

    #tgt_in = tf.concat(([2], labels), 0)  # 加上sos
    #tgt_out = tf.concat((labels, [1]), 0)  # 加上eos

    frames_length = context_parsed["frames_length"]
    frames_length = tf.cast(frames_length, dtype=tf.int32)

    return mouth_frames, mix_audio_frames, audio_frames, frames_length


def _val_parse_function(example_proto):
    context_features = {
        "label_length": tf.FixedLenFeature([], dtype=tf.int64),
        "label_lrw500": tf.FixedLenFeature([], dtype=tf.int64),
        "frames_length": tf.FixedLenFeature([], dtype=tf.int64)
    }

    sequence_features = {
        "mouth_frames": tf.FixedLenSequenceFeature([], dtype=tf.string),
        "mix_audio_frames": tf.FixedLenSequenceFeature([], dtype=tf.string),
        "audio_frames": tf.FixedLenSequenceFeature([], dtype=tf.string),
        "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=example_proto,
        context_features=context_features,
        sequence_features=sequence_features
    )

    #labels
    #labels = sequence_parsed["labels"]
    #labels = tf.cast(labels, dtype=tf.int32)
    #tgt_in = tf.concat(([2], labels), 0)   # 加上sos
    #tgt_out = tf.concat((labels, [1]), 0)  # 加上eos

    #video

    mouth_frames = sequence_parsed["mouth_frames"]
    #labels = sequence_parsed["labels"]
    #labels = tf.cast(labels, dtype=tf.int32)
    mouth_frames = tf.decode_raw(mouth_frames, np.uint8)
    # print("frames",frames)

    # 定长
    # frames = tf.reshape(frames, (77, 112, 112, 3))
    # 变长
    mouth_frames = tf.reshape(mouth_frames, (-1, 112, 112, 1))
    mouth_frames = tf.image.convert_image_dtype(mouth_frames, dtype=tf.float32)

    mouth_frames = tf.subtract(mouth_frames, 0.5)
    mouth_frames = tf.multiply(mouth_frames, 2)

    #audio
    audio_frames = sequence_parsed["audio_frames"]
    #labels = sequence_parsed["labels"]

    audio_frames = tf.decode_raw(audio_frames, np.float32)

    # 定长
    audio_frames = tf.reshape(audio_frames, (-1, 80))
    # 不定长
    # frames = tf.reshape(frames, (label_length, 112, 112, 3))
    audio_frames = tf.image.convert_image_dtype(audio_frames, dtype=tf.float32)

    audio_frames = tf.subtract(audio_frames, 0.5)
    audio_frames = tf.multiply(audio_frames, 2)

    #mix_audio
    mix_audio_frames = sequence_parsed["mix_audio_frames"]
    # labels = sequence_parsed["labels"]

    mix_audio_frames = tf.decode_raw(mix_audio_frames, np.float32)

    # 定长
    mix_audio_frames = tf.reshape(mix_audio_frames, (-1, 80))
    # 不定长
    # frames = tf.reshape(frames, (label_length, 112, 112, 3))
    mix_audio_frames = tf.image.convert_image_dtype(mix_audio_frames, dtype=tf.float32)

    mix_audio_frames = tf.subtract(mix_audio_frames, 0.5)
    mix_audio_frames = tf.multiply(mix_audio_frames, 2)

    frames_length = context_parsed["frames_length"]
    frames_length = tf.cast(frames_length, dtype=tf.int32)
    return mouth_frames, mix_audio_frames, audio_frames, frames_length


if __name__ == '__main__':
    path = "/home/jack/dataset/w22lml/tfrecord/word_counts.txt"
    int_to_vocab, vocab_to_int = Vocabulary(path)._extract_charater_vocab(path)
    print([int_to_vocab[i] for i in [100, 712, 460, 367, 90]])
    print([vocab_to_int[char] for char in ["知", "足", "常", "乐"]])
