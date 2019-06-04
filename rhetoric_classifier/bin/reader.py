'''
   Created by zhiqliu on 2018/12/18.
'''


import os
import codecs
import collections
import numpy as np
import conf
from log import g_log_inst
import tensorflow as tf
from tensorflow.contrib.learn import preprocessing
import sys
from gensim.models import Word2Vec
np.set_printoptions(threshold=np.nan)

# global variable
TRAINSET_FILENAME = 'trainset_1225_cc.txt'
VALIDSET_FILENAME = 'validset_1225_cc.txt'
TESTSET_FILENAME = 'testset_1225_cc.txt'

class a(object):
    def __init__(self):
        return

def tokenizer_fn(iterator):
    return (x.split('|') for x in iterator)

def create_iter(data_path):
    with codecs.open(os.path.join(data_path, TRAINSET_FILENAME), 'r', 'utf-8') as rfd:
        for line in rfd:
            if (len(line.strip('\n').split('\t'))) != 3:
                continue
            yield line.strip('\n').split('\t')

def write_vocabulary(vocab_processor, outfile):
    vocab_size = len(vocab_processor.vocabulary_)
    vocab_list = [vocab_processor.vocabulary_._reverse_mapping[id_]
        for id_ in range(vocab_size)]
    with codecs.open(outfile, 'w') as wfd:
        wfd.write('\n'.join(vocab_list))
    print('Saved vocabulary to {}'.format(outfile))

def create_vocab(input_iter, min_freq):
    vp = preprocessing.VocabularyProcessor(
        max_document_length = 100,
        min_frequency = min_freq,
        tokenizer_fn = tokenizer_fn)
    vp.fit(input_iter)
    return vp

def save_vocab():
    #vocab_size = len(vocab_processor.vocabulary_)
    input_iter = create_iter('../data4')
    input_iter = (tokens for _, tokens, _ in input_iter)
    vocab = create_vocab(input_iter, min_freq = 0)
    vocab.save(os.path.join('../data4', 'vocab_processor.bin'))
    write_vocabulary(vocab, os.path.join('../data4', 'vocabulary.txt'))
    vocab_size = len(vocab.vocabulary_)
    word2id = dict()
    vocab_list = [vocab.vocabulary_._reverse_mapping[id_]
        for id_ in range(vocab_size)]
    for index, v in enumerate(vocab_list):
        word2id[v] = index
    dump_word_embeddings(word2id)
   
def _build_data_pairs(fname, with_label=False):
    data_pairs = []
    vp = preprocessing.VocabularyProcessor.restore('../data4/vocab_processor.bin')
    with codecs.open(fname, 'r', 'utf-8') as rfd:
        for line in rfd:
                if len(line.strip('\n').split('\t')) != 3:
                    continue
                data_id, tokens, domain = line.strip('\n').split('\t')
                token_list = tokens.split('|')
                data = [x.lower() for x in token_list if len(x.strip())]
                if len(data) == 0:
                    continue
                sequence = np.array(list(vp.transform([('|'.join(data))]))[0])
                if with_label:
                    label = int(str(domain))
                    print (label)
                    #label = dic[str(domain)]
                else:
                    label = 100
                pair = [data_id, label] + list(sequence)  # [id, label, sequence]
                data_pairs.append(pair)
    g_log_inst.get().info('_build_data_pairs() success, fname=%s, len(data_pairs)=%s, with_label=%s' % (
    fname, len(data_pairs), with_label))
    return data_pairs

def load_predict_data(data_path, fname, with_label=False):
    pred_data = _build_data_pairs(os.path.join(data_path, fname), with_label)
    return pred_data, {}

def load_train_data(data_path):
    trainset_fname = os.path.join(data_path, TRAINSET_FILENAME)
    validset_fname = os.path.join(data_path, VALIDSET_FILENAME)
    testset_fname = os.path.join(data_path, TESTSET_FILENAME)
    train_data = _build_data_pairs(trainset_fname, with_label=True)
    valid_data = _build_data_pairs(validset_fname, with_label=True)
    test_data = _build_data_pairs(testset_fname, with_label=True)
    return [train_data, valid_data, test_data], {}

def pairs_iterator(raw_datas, batch_size, num_steps):
    # truncate if raw input is too long and padding if too short
    # note here raw_datas has format [id, label, sequence(..)],
    # i.e. first two elements of elements of raw_datas are id and label
    raw_datas = list(map(lambda x: x[0:num_steps + 2], raw_datas))
    seq_lengths = list(map(lambda x: len(x) - 2, raw_datas))
    padded_datas = list(map(lambda x: x + [0] * (num_steps - (len(x) - 2)), raw_datas))

    epoch_size = len(padded_datas) // batch_size
    padded_datas = np.array(padded_datas[0: epoch_size * batch_size])
    padded_data_array = np.reshape(padded_datas, (epoch_size, batch_size, num_steps + 2))
    # keep the same length with padded_datas
    seq_lengths = np.array(seq_lengths[0: epoch_size * batch_size])
    seq_lengths_array = np.reshape(seq_lengths, (epoch_size, batch_size))

    for i in range(epoch_size):
        sequences = padded_data_array[i, :, 2:]
        ## a batch label array is constructed here
        labels = padded_data_array[i, :, 1]
        data_ids = padded_data_array[i, :, 0]
        labels = np.repeat(labels, repeats=1, axis=0)
        labels = np.reshape(labels, (-1, 1))
        seq_lens = seq_lengths_array[i, :]
        yield (data_ids, sequences, labels, seq_lens)

import codecs
def data_process(input, output):
    dic = {'No-rhetoric': 0, 'Metaphor': 1, 'Personification': 2}
    fw = codecs.open(output, 'w', 'utf-8')
    with codecs.open(input, 'r', 'utf-8') as fr:
        for line in fr:
            lines = line.strip().split('\t')
            lines[2] = str(dic[lines[2]])
            fw.write('\t'.join(lines) + '\n')
    fw.close()

   
def dump_word_embeddings(word2id):
        #import random as np
        emb_size = 300
        vocab_size = len(word2id)

        word2vec = Word2Vec.load('../data2/word2vec.model')
        embeddings = np.random.randn(vocab_size, emb_size)
        for word, idx in word2id.items():
            if word in word2vec:
                embeddings[idx, :] = word2vec[word]
            else:
                embeddings[idx, :] = np.random.randn(emb_size)
        print(embeddings.shape)
        np.save('../data2/word2vec_new.model', embeddings)

if __name__ == '__main__':
    g_log_inst.start('../log/reader.log', __name__, 'DEBUG')
    save_vocab()
    #data_process('../data/train.txt', '../data/trainset.txt')
    #data_process('../data/test.txt', '../data/testset.txt')
