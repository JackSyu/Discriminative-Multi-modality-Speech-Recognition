# encoding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
from video_preprocess import Video
import os, fnmatch, sys, errno
from skimage import io
#import detect_face
import tensorflow as tf
import numpy as np
import glob
import librosa.display
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

dic_path = "/data-private/nas/src/mtcnn-tensorflow/original_english_word_count.txt"

def _extract_charater_vocab(dictionary):
    '''get label_to_text'''
    words = []
    with open(dictionary, 'r') as f:
        lines = f.readlines()
        #print "lines",lines
        for line in lines:
            words.append(line.strip('\n').split(' ')[0])
    #words = sorted(words)
    special_words = ['<PAD>', '<EOS>', '<BOS>', ' ']
        # special_words = []
    int_to_vocab = {idx: word for idx, word in enumerate(special_words + words)}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
    # print(int_to_vocab)
    return int_to_vocab, vocab_to_int

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature_list(values):
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature_list(values):
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])


def _to_sequence_example(sub_mouth_frames, audio_frames, sub_labels):
    '''Build a SequenceExample proto for an video-label pair.
    Args:
        video_path: Path of video data.
        label_path: Path of label data.
        vocab: A Vocabulary object.
    Returns:
        A SequenceExample proto.
    '''

    labels = sub_labels
    label_len = len(labels)
    mouth_frames =  sub_mouth_frames

    #print("mouth_frames.shape",mouth_frames )
    mouth_frames_byte = [mouth_frame.tobytes() for mouth_frame in mouth_frames]
    audio_frames_byte = [audio_frame.tobytes() for audio_frame in audio_frames]

    example = tf.train.SequenceExample(
        # context 用来放置非序列化的部分
        context=tf.train.Features(feature={
            "words_num": _int64_feature(label_len)

        }),
        # feature_lists用来放置变长序列
        feature_lists=tf.train.FeatureLists(feature_list={
            "mouth_frames": _bytes_feature_list(mouth_frames_byte),
            "audio_frames": _bytes_feature_list(audio_frames_byte),
            "labels": _int64_feature_list(labels)
        })
    )
    return example

def snr_factor(signal, noise, snr_db):
    s = signal
    n = noise

    if s.size != n.size:
        raise Exception('signal and noise must have the same length')

    eq = np.sqrt(np.var(s) / np.var(n))
    factor = eq * (10 ** (-snr_db / 20.0))

    return factor


SOURCE_PATH = '/data-private/nas/lrs3/trainval'
FACE_PREDICTOR_PATH = '/home/xub/src/audio_video_process_to_tfrecord/shape_predictor_68_face_landmarks.dat'

if __name__ == '__main__':
    #############################video process###################
    gg=1
    j = 0
    md = 0
    sess = tf.Session()

    noise_path = "/data-private/nas/lrw/noisy_audio"
    int_to_vocab, vocab_to_int = _extract_charater_vocab(dic_path)

    filename = ("/data-private/nas/lrs3/trainval/data2.tfrecords" )
    writer = tf.python_io.TFRecordWriter(filename)

    for filepath in find_files(SOURCE_PATH, '*.mp4'):

        try:
            filepath_wo_ext = os.path.splitext(filepath)[0].split('\\')[-1]

            # process video to frames
            print("process video to frames")
            video = Video(vtype='face', face_predictor_path=FACE_PREDICTOR_PATH).from_video(filepath)
            print("jjjj: {}".format(filepath_wo_ext))

            # process wav
            print("process wav")
            filepath_no_ext = os.path.splitext(filepath)[0]
            wav_path = filepath_no_ext + ".wav"

            y, sr = librosa.load(wav_path, sr=None)
            D = np.abs(librosa.core.stft(y, n_fft=640, hop_length=160, win_length=640, window='hann'))
            noise_txt_path = noise_path + "/{}".format("noise.txt")
            n_D = np.loadtxt(noise_txt_path)
            b = int((D.shape[1] - n_D.shape[1]) / 2)
            a = D.shape[1] - n_D.shape[1] - b
            n_D = np.pad(n_D, ((0, 0), (a, b)), 'constant')

            D_T = (librosa.amplitude_to_db(D, ref=np.max)).T

            minMax = MinMaxScaler()
            audio_feature_std = minMax.fit_transform(D_T)
            print("original audio_feature shape", audio_feature_std.shape)

            length = int((audio_feature_std.shape)[0])

            original_feature = audio_feature_std
            if length == 4 * int(length / 4):
                loop_num = int(length / 4)

            else:
                loop_num = 1 + int(length / 4)

            audio_feature = np.zeros((loop_num, 1284), dtype=np.float32)

            PAD = np.zeros((1, 321), dtype=np.float32)
            if length == 4 * int(length / 4):
                for i in range(loop_num):
                    audio_feature[i, :] = np.hstack((original_feature[4 * i], original_feature[4 * i + 1],
                                                     original_feature[4 * i + 2], original_feature[4 * i + 3]))
            else:
                for i in range(loop_num):
                    if i != loop_num - 1:
                        audio_feature[i, :] = np.hstack((original_feature[4 * i], original_feature[4 * i + 1],
                                                         original_feature[4 * i + 2], original_feature[4 * i + 3]))
                    else:
                        if length - 4 * int(length / 4) == 1:
                            audio_feature[i, :] = np.hstack((original_feature[4 * i], PAD[0], PAD[0], PAD[0]))

                        elif length - 4 * int(length / 4) == 2:
                            audio_feature[i, :] = np.hstack(
                                (original_feature[4 * i], original_feature[4 * i + 1], PAD[0], PAD[0]))

                        elif length - 4 * int(length / 4) == 3:
                            audio_feature[i, :] = np.hstack(
                                (original_feature[4 * i], original_feature[4 * i + 1], original_feature[4 * i + 2],
                                 PAD[0]))

            #print("original audio feature", audio_feature.shape)
            audio_feature_frames = np.zeros((video.mouth.shape[0], 4*321), dtype=np.float32)
            if audio_feature.shape[0] > video.mouth.shape[0]:
                for fff in range(video.mouth.shape[0]):
                    audio_feature_frames[fff] = audio_feature[fff]
            else:
                audio_feature_frames = audio_feature
            print("final audio_feature", audio_feature_frames.shape)
            # process labels
            print("process labels")

            txt_path = filepath_no_ext + ".txt"
            # print "txt path ",txt_path
            with open(txt_path, 'r') as f:
                txt = f.readlines()

                # whole words in one sample
                whole_info = txt[0]
                print("whole txt info", whole_info)

                # the length of whole words in one sample
                role = whole_info.split(":")
                line_spoke = role[1].strip()

                length = len(line_spoke)
                print('word', line_spoke)

                txt_labels = [vocab_to_int[token] for token in line_spoke]
                print('text label', txt_labels)
                sequence_example = _to_sequence_example(video.mouth, audio_feature_frames, txt_labels)

                if video.missing_Detect == True:
                    md += 1
                    print("the %dth missing detection number" % md)
                else:
                    writer.write(sequence_example.SerializeToString())
                    j += 1
                    print("the %dth sample number" % j)
        except:
            print("the %dth failed" % gg, filepath)
            gg += 1
            continue

        #############################txt label process##################
    writer.close()

