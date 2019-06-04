import numpy as np
import tensorflow as tf
import sys
import time
import random
import pickle as pkl
import math
random.seed(time.time())

from model import Seq2SeqModel, _START_VOCAB

# import tokenizer
try:
    from wordseg_python import Global
except:
    Global = None

tf.app.flags.DEFINE_boolean("is_train", True, "Set to False to inference.")
tf.app.flags.DEFINE_integer("symbols", 40000, "vocabulary size.")
tf.app.flags.DEFINE_integer("topic_symbols", 1000, "topic vocabulary size.")
tf.app.flags.DEFINE_integer("full_kl_step", 8000, "Total steps to finish annealing")
tf.app.flags.DEFINE_integer("embed_units", 100, "Size of word embedding.")
tf.app.flags.DEFINE_integer("units", 256, "Size of hidden units.")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size to use during training.")
tf.app.flags.DEFINE_string("data_dir", "data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory.")
tf.app.flags.DEFINE_integer("per_checkpoint", 1000, "How many steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("inference_version", 0, "The version for inferencing.")
tf.app.flags.DEFINE_boolean("log_parameters", True, "Set to True to show the parameters")
tf.app.flags.DEFINE_string("inference_path", "", "Set filename of inference, default isscreen")
tf.app.flags.DEFINE_string("num_keywords", 4, "Number of keywords extracted from poems")

FLAGS = tf.app.flags.FLAGS

def load_data(path, fname):
    # data sample: (origin sentences, response sentence, keyword, rhetorical label)
    # origin sentences: tokenized sequence
    # response sentence: tokenized  sequence
    # keyword: keywords extracted from context.
    # rhetorical label: one-hot rhetorical label of sentences (annotated by rhetorical classifier)
    with open('%s/%s.origin' % (path, fname)) as f:
        ori_sent = [line.strip().split() for line in f.readlines()]
    with open('%s/%s.response' % (path, fname)) as f:
        rep_sent = [line.strip().split() for line in f.readlines()]
    with open('%s/%s.keyword' % (path, fname)) as f:
        keyword = [line.strip().split() for line in f.readlines()]
    with open('%s/%s.label' % (path, fname)) as f:
        label = [line.strip().split('\t') for line in f.readlines()]    
    data = []
    for o, r, k, l in zip(ori_sent, rep_sent, keyword, label):
        data.append({'ori_sent': o, 'rep_sent': r, 'keyword': k, 'label':l})
    return data

def build_vocab(path, data, stop_list, rhetoric_word_list):
    print("Creating vocabulary...")
    vocab = {}
    vocab_topic = {}
    for i, pair in enumerate(data):
        if i % 100000 == 0:
            print("    processing line %d" % i)
        for token in pair['ori_sent']+pair['rep_sent']:
            if token in vocab:
                vocab[token] += 1
            else:
                vocab[token] = 1
        for token in pair['keyword']:
            if token not in stop_list: # remove stopwords from vocab_topic
                if token in vocab_topic:
                    vocab_topic[token] += 1
                else:
                    vocab_topic[token] = 1
    vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    vocab_topic_list = sorted(vocab_topic, key = vocab_topic.get, reverse = True)

    if len(vocab_list) > FLAGS.symbols:
        vocab_list = vocab_list[:FLAGS.symbols] # remove words with low frequency from vocab_list
    vocab_topic_list_new = []
    for word in vocab_topic_list:
        if word in vocab_list:
            vocab_topic_list_new.append(word) # keep topic words in vocab_list
    if len(vocab_topic_list_new) > FLAGS.topic_symbols:
        vocab_topic_list_new = vocab_topic_list_new[:FLAGS.topic_symbols]

    content_pos_list = [] # record the position of content words
    content_cnt = 0
    for ele in vocab_list:
        if ele not in rhetoric_word_list:
            content_cnt += 1
            content_pos_list.append(vocab_list.index(ele))
    print ('content_cnt = ', content_cnt)

    rhetoric_pos_list = [] # record the position of rhetoric words
    for ele in rhetoric_word_list.items():
        if ele[0] in vocab_list:
            rhetoric_pos_list.append(vocab_list.index(ele[0]))

    # Load pre-trained word vectors from path/vector.txt
    print("Loading word vectors...")
    vectors = {}
    with open('%s/vector.txt' % path) as f:
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print("    processing line %d" % i)
            s = line.strip()
            word = s[:s.find(' ')]
            vector = s[s.find(' ')+1:]
            vectors[word] = vector

    embed = []
    for word in vocab_list:
        if word in vectors:
            vector = map(float, vectors[word].split())
        else:
            vector = np.zeros((FLAGS.embed_units), dtype=np.float32)
        embed.append(vector)
    embed = np.array(embed, dtype=np.float32)

    return vocab_list, embed, vocab_topic_list_new, content_pos_list, rhetoric_pos_list


def gen_batched_data(data):
    encoder_len = max([len(item['ori_sent']) for item in data])+1
    decoder_len = max([len(item['rep_sent']) for item in data])+1
    
    ori_sents, rep_sents, oris_length, reps_length, labels = [], [], [], [], []
    def padding(sent, l):
        return sent + ['_EOS'] + ['_PAD'] * (l-len(sent)-1)
        
    for item in data:
        ori_sents.append(padding(item['ori_sent'], encoder_len))
        rep_sents.append(padding(item['rep_sent'], decoder_len))
        oris_length.append(len(item['post'])+1)
        reps_length.append(len(item['response'])+1)
        labels.append(item['label'])

    batched_data = {'ori_sents': np.array(ori_sents),
            'rep_sents': np.array(rep_sents),
            'oris_length': oris_length,
            'reps_length': reps_length,
            'labels':np.array(labels)}
    return batched_data


def train(model, sess, data_train, global_t):
    batched_data = gen_batched_data(data_train)
    outputs = model.step_decoder(sess, batched_data, global_t = global_t)
    return outputs


def evaluate(model, sess, data_dev):
    # Evaluation on dev set
    loss = np.zeros((1, ))
    kl_loss, dec_loss, classifier_loss = np.zeros((1, )), np.zeros((1, )), np.zeros((1, ))
    st, ed, times = 0, FLAGS.batch_size, 0
    while st < len(data_dev):
        selected_data = data_dev[st:ed]
        batched_data = gen_batched_data(selected_data)
        outputs = model.step_decoder(sess, batched_data, forward_only=True, global_t = FLAGS.full_kl_step)
        kl_loss += outputs[1]
        dec_loss += outputs[2]
        classifier_loss += outputs[3]
        loss += outputs[0]
        st, ed = ed, ed+FLAGS.batch_size
        times += 1
    loss /= times
    kl_loss /= times
    dec_loss /= times
    classifier_loss /= times
    show = lambda a: '[%s]' % (' '.join(['%.2f' % x for x in a]))
    print('perplexity on dev set: %s  kl_loss: %s  dec_loss: %s  classifier_loss: %s ' % (show(np.exp(dec_loss)), show(kl_loss), show(dec_loss), show(classifier_loss)))


def inference(model, sess, ori_sents, label_no):
    length = [len(p)+1 for p in posts]
    def padding(sent, l):
        return sent + ['_EOS'] + ['_PAD'] * (l-len(sent)-1)

    batched_ori_sents = [padding(p, max(length)) for p in ori_sents]
    batched_data = {'ori_sents': np.array(batched_ori_sents),
            'reps_length': np.array(length, dtype=np.int32)}

    results_inf = model.inference(sess, batched_data, label_no)
    rep_sents = results_inf[0]
    results = []
    res_cnt = 0
    for response in rep_sents:
        result = []
        token_cnt = 0
        for token in response:
            if token != '_EOS':
                result.append(token)
                token_cnt += 1
            else:
                break
        res_cnt += 1
        results.append(result)
    return results


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # load dataset
    data_train = load_data(FLAGS.data_dir, 'train')
    data_dev = load_data(FLAGS.data_dir, 'dev')

    # load stopword list
    stop_list = {}
    stop_file = open('stopword_utf8.txt', 'r')
    line_stop = stop_file.readline()
    while line_stop:
        temp = line_stop.strip()
        if temp not in stop_list:
            stop_list[temp] = 1
        else:
            stop_list[temp] += 1
        line_stop = stop_file.readline()
    stop_file.close()
    print ('stop_list=', len(stop_list))

    # load function-related word list
    rhetoric_words_list = {}
    rhetoric_words_file = open('rhetoric_words.txt', 'r')
    line_rhetoric = rhetoric_words_file.readline()
    while line_rhetoric:
        temp = line_rhetoric.strip()
        if temp not in rhetoric_words_list:
            rhetoric_words_list[temp] = 1
        else:
            rhetoric_words_list[temp] += 1
        line_func = rhetoric_words_file.readline()
        rhetoric_words_file.close()
    print ('rhetoric_words_list=', len(rhetoric_words_list))

    # build vocabularies
    vocab, embed, vocab_topic, content_pos, rhetoric_words_pos = build_vocab(FLAGS.data_dir, data_train, stop_list, rhetoric_words_list)
    print ('num_topic_vocab=', len(vocab_topic))
    print ('num_rhetoric_vocab=', len(rhetoric_words_pos))

    # Training mode
    if FLAGS.is_train:
        model = Seq2SeqModel(
                FLAGS.symbols, 
                FLAGS.embed_units,
                FLAGS.units,
                is_train=True,
                vocab=vocab,
                content_pos=content_pos,
                rhetoric_pos = rhetoric_words_pos,
                embed=embed,
                full_kl_step=FLAGS.full_kl_step)

        if FLAGS.log_parameters:
            model.print_parameters()
        
        if tf.train.get_checkpoint_state(FLAGS.train_dir):
            print("Reading model parameters from %s" % FLAGS.train_dir)
            model.saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))
            model.symbol2index.init.run()
        else:
            print("Created model with fresh parameters.")
            tf.global_variables_initializer().run()
            model.symbol2index.init.run()
        
        temp_total_losses, total_loss_step, kl_loss_step, decoder_loss_step, classifier_loss_step, time_step = np.zeros((1, )), np.zeros((1, )), np.zeros((1, )), np.zeros((1, )), np.zeros((1, )), .0
        previous_losses = [1e18]*6

        num_batch = len(data_train) / FLAGS.batch_size
        random.shuffle(data_train)
        pre_train = [data_train[i:i+FLAGS.batch_size] for i in range(0, len(data_train), FLAGS.batch_size)]
        if len(data_train) % FLAGS.batch_size != 0:
            pre_train.pop() 
        random.shuffle(pre_train)
        ptr = 0
        global_t = 0
        while True:
            if model.global_step.eval() % FLAGS.per_checkpoint == 0:
                show = lambda a: '[%s]' % (' '.join(['%.2f' % x for x in a]))
                print("global step %d learning rate %.4f step-time %.2f perplexity %s kl_loss %s dec_loss %s dis_loss %s"
                        % (model.global_step.eval(), model.learning_rate.eval(), 
                            time_step, show(np.exp(decoder_loss_step)), show(kl_loss_step), show(decoder_loss_step), show(classifier_loss_step)))
                model.saver.save(sess, '%s/checkpoint' % FLAGS.train_dir, global_step=model.global_step)
                evaluate(model, sess, data_dev)
                if np.sum(temp_total_losses) > max(previous_losses):
                    sess.run(model.learning_rate_decay_op)
                previous_losses = previous_losses[1:]+[np.sum(temp_total_losses)]
                temp_total_losses, total_loss_step, kl_loss_step, decoder_loss_step, classifier_loss_step, time_step = np.zeros((1, )), np.zeros((1, )), np.zeros((1, )), np.zeros((1, )), np.zeros((1, )), .0

            global_t = model.global_step.eval()
            start_time = time.time()
            temp_loss = train(model, sess, pre_train[ptr], global_t)
            total_loss_step += temp_loss[0] / FLAGS.per_checkpoint
            kl_loss_step += temp_loss[1] / FLAGS.per_checkpoint
            decoder_loss_step += temp_loss[2] / FLAGS.per_checkpoint
            classifier_loss_step += temp_loss[3] / FLAGS.per_checkpoint
            if global_t>=1:
                temp_total_losses += decoder_loss_step + kl_loss_step*FLAGS.full_kl_step/global_t + classifier_loss_step
            time_step += (time.time() - start_time) / FLAGS.per_checkpoint
            ptr += 1
            if ptr == num_batch:
                random.shuffle(pre_train)
                ptr = 0
            
    else:
        model = Seq2SeqModel(
                FLAGS.symbols, 
                FLAGS.embed_units, 
                FLAGS.units,
                is_train=False,
                content_pos = content_pos,
                rhetoric_pos = rhetoric_words_pos,
                vocab=None)

        if FLAGS.inference_version == 0:
            model_path = tf.train.latest_checkpoint(FLAGS.train_dir)
        else:
            model_path = '%s/checkpoint-%08d' % (FLAGS.train_dir, FLAGS.inference_version)
        print('restore from %s' % model_path)
        model.saver.restore(sess, model_path)
        model.symbol2index.init.run()

        # tokenizer
        def split(sent):
            if Global == None:
                return sent.split()

            sent = sent.decode('utf-8', 'ignore').encode('gbk', 'ignore')
            tuples = [(word.decode("gbk").encode("utf-8"), pos) 
                    for word, pos in Global.GetTokenPos(sent)]
            return [each[0] for each in tuples]


        posts = []
        posts_ori = []
        with open(FLAGS.inference_path) as f:
            for line in f:
                sent = line.strip()
                posts_ori.append(sent)
                cur_post = split(sent)
                posts.append(cur_post)

        responses = [[], [], []]
        st, ed = 0, FLAGS.batch_size
        while st < len(posts):
            for i in range(3):
                temp = inference(model, sess, posts[st: ed], i)
                responses[i] += temp
            st, ed = ed, ed+FLAGS.batch_size

        with open(FLAGS.inference_path+'.out', 'w') as f:
            for i in range(len(posts)):
                for k in range(3):
                    f.writelines('%s\n' % (''.join(responses[k][i])))
                f.writelines('\n')
