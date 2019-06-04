import numpy as np
import tensorflow as tf
import my_attention_decoder_fn
import my_loss
import my_seq2seq

from tensorflow.python.ops.nn import dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import GRUCell, LSTMCell, MultiRNNCell
from tensorflow.contrib.lookup.lookup_ops import HashTable, KeyValueTensorInitializer
from tensorflow.contrib.layers.python.layers import layers
from output_projection import output_projection_layer
from tensorflow.python.ops import variable_scope

from utils import sample_gaussian
from utils import gaussian_kld

PAD_ID = 0
UNK_ID = 1
GO_ID = 2
EOS_ID = 3
_START_VOCAB = ['_PAD', '_UNK', '_GO', '_EOS']

class Seq2SeqModel(object):
    def __init__(self,
            num_symbols,
            num_embed_units,
            num_units,
            is_train,
            vocab=None,
            content_pos=None,
            rhetoric_pos = None,
            embed=None,
            learning_rate=0.1,
            learning_rate_decay_factor=0.9995,
            max_gradient_norm=5.0,
            max_length=30,
            latent_size=128,
            use_lstm=False,
            num_classes=3,
            full_kl_step=80000,
            mem_slot_num=4,
            mem_size=128):
        
        self.ori_sents = tf.placeholder(tf.string, shape=(None, None))
        self.ori_sents_length = tf.placeholder(tf.int32, shape=(None))
        self.rep_sents = tf.placeholder(tf.string, shape=(None, None))
        self.rep_sents_length = tf.placeholder(tf.int32, shape=(None))
        self.labels = tf.placeholder(tf.float32, shape=(None, num_classes))
        self.use_prior = tf.placeholder(tf.bool)
        self.global_t = tf.placeholder(tf.int32)
        self.content_mask = tf.reduce_sum(tf.one_hot(content_pos, num_symbols, 1.0, 0.0), axis = 0)
        self.rhetoric_mask = tf.reduce_sum(tf.one_hot(rhetoric_pos, num_symbols, 1.0, 0.0), axis = 0)

        topic_memory = tf.zeros(name="topic_memory", dtype=tf.float32,
                                  shape=[None, mem_slot_num, mem_size])

        w_topic_memory = tf.get_variable(name="w_topic_memory", dtype=tf.float32,
                                    initializer=tf.random_uniform([mem_size, mem_size], -0.1, 0.1))

        # build the vocab table (string to index)
        if is_train:
            self.symbols = tf.Variable(vocab, trainable=False, name="symbols")
        else:
            self.symbols = tf.Variable(np.array(['.']*num_symbols), name="symbols")
        self.symbol2index = HashTable(KeyValueTensorInitializer(self.symbols, 
            tf.Variable(np.array([i for i in range(num_symbols)], dtype=np.int32), False)), 
            default_value=UNK_ID, name="symbol2index")

        self.ori_sents_input = self.symbol2index.lookup(self.ori_sents)
        self.rep_sents_target = self.symbol2index.lookup(self.rep_sents)
        batch_size, decoder_len = tf.shape(self.rep_sents)[0], tf.shape(self.rep_sents)[1]
        self.rep_sents_input = tf.concat([tf.ones([batch_size, 1], dtype=tf.int32)*GO_ID,
            tf.split(self.rep_sents_target, [decoder_len-1, 1], 1)[0]], 1)
        self.decoder_mask = tf.reshape(tf.cumsum(tf.one_hot(self.rep_sents_length-1,
            decoder_len), reverse=True, axis=1), [-1, decoder_len])        
        
        # build the embedding table (index to vector)
        if embed is None:
            # initialize the embedding randomly
            self.embed = tf.get_variable('embed', [num_symbols, num_embed_units], tf.float32)
        else:
            # initialize the embedding by pre-trained word vectors
            self.embed = tf.get_variable('embed', dtype=tf.float32, initializer=embed)

        self.pattern_embed = tf.get_variable('pattern_embed', [num_classes, num_embed_units], tf.float32)
        
        self.encoder_input = tf.nn.embedding_lookup(self.embed, self.ori_sents_input)
        self.decoder_input = tf.nn.embedding_lookup(self.embed, self.rep_sents_input)

        if use_lstm:
            cell_fw = LSTMCell(num_units)
            cell_bw = LSTMCell(num_units)
            cell_dec = LSTMCell(2*num_units)
        else:
            cell_fw = GRUCell(num_units)
            cell_bw = GRUCell(num_units)
            cell_dec = GRUCell(2*num_units)

        # origin sentence encoder
        with variable_scope.variable_scope("encoder"):
            encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.encoder_input, 
                self.ori_sents_length, dtype=tf.float32)
            post_sum_state = tf.concat(encoder_state, 1)
            encoder_output = tf.concat(encoder_output, 2)

        # response sentence encoder
        with variable_scope.variable_scope("encoder", reuse = True):
            decoder_state, decoder_last_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.decoder_input, 
                self.rep_sents_length, dtype=tf.float32)
            response_sum_state = tf.concat(decoder_last_state, 1)

        # recognition network
        with variable_scope.variable_scope("recog_net"):
            recog_input = tf.concat([post_sum_state, response_sum_state], 1)
            recog_mulogvar = tf.contrib.layers.fully_connected(recog_input, latent_size * 2, activation_fn=None, scope="muvar")
            recog_mu, recog_logvar = tf.split(recog_mulogvar, 2, axis=1)

        # prior network
        with variable_scope.variable_scope("prior_net"):
            prior_fc1 = tf.contrib.layers.fully_connected(post_sum_state, latent_size * 2, activation_fn=tf.tanh, scope="fc1")
            prior_mulogvar = tf.contrib.layers.fully_connected(prior_fc1, latent_size * 2, activation_fn=None, scope="muvar")
            prior_mu, prior_logvar = tf.split(prior_mulogvar, 2, axis=1)

        latent_sample = tf.cond(self.use_prior,
                                lambda: sample_gaussian(prior_mu, prior_logvar),
                                lambda: sample_gaussian(recog_mu, recog_logvar))


        # classifier
        with variable_scope.variable_scope("classifier"):
            classifier_input = latent_sample
            pattern_fc1 = tf.contrib.layers.fully_connected(classifier_input, latent_size, activation_fn=tf.tanh, scope="pattern_fc1")
            self.pattern_logits = tf.contrib.layers.fully_connected(pattern_fc1, num_classes, activation_fn=None, scope="pattern_logits")

        self.label_embedding = tf.matmul(self.labels, self.pattern_embed)

        output_fn, my_sequence_loss = output_projection_layer(2*num_units, num_symbols, latent_size, num_embed_units, self.content_mask, self.rhetoric_mask)

        attention_keys, attention_values, attention_score_fn, attention_construct_fn = my_attention_decoder_fn.prepare_attention(encoder_output, 'luong', 2*num_units)

        with variable_scope.variable_scope("dec_start"):
            temp_start = tf.concat([post_sum_state, self.label_embedding, latent_sample], 1)
            dec_fc1 = tf.contrib.layers.fully_connected(temp_start, 2*num_units, activation_fn=tf.tanh, scope="dec_start_fc1")
            dec_fc2 = tf.contrib.layers.fully_connected(dec_fc1, 2*num_units, activation_fn=None, scope="dec_start_fc2")

        if is_train:
            # rnn decoder
            topic_memory = self.update_memory(topic_memory, encoder_output)
            extra_info = tf.concat([self.label_embedding, latent_sample, topic_memory], 1)

            decoder_fn_train = my_attention_decoder_fn.attention_decoder_fn_train(dec_fc2, 
                attention_keys, attention_values, attention_score_fn, attention_construct_fn, extra_info)
            self.decoder_output, _, _ = my_seq2seq.dynamic_rnn_decoder(cell_dec, decoder_fn_train, 
                self.decoder_input, self.rep_sents_length, scope = "decoder")

            # calculate the loss
            self.decoder_loss = my_loss.sequence_loss(logits = self.decoder_output, 
                targets = self.rep_sents_target, weights = self.decoder_mask,
                extra_information = latent_sample, label_embedding = self.label_embedding, softmax_loss_function = my_sequence_loss)
            temp_klloss = tf.reduce_mean(gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar))
            self.kl_weight = tf.minimum(tf.to_float(self.global_t)/full_kl_step, 1.0)
            self.klloss = self.kl_weight * temp_klloss
            temp_labels = tf.argmax(self.labels, 1)
            self.classifierloss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pattern_logits, labels=temp_labels))
            self.loss = self.decoder_loss + self.klloss + self.classifierloss  # need to anneal the kl_weight
            
            # building graph finished and get all parameters
            self.params = tf.trainable_variables()
        
            # initialize the training process
            self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
            self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
            self.global_step = tf.Variable(0, trainable=False)
            
            # calculate the gradient of parameters
            opt = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
            gradients = tf.gradients(self.loss, self.params)
            clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, 
                    max_gradient_norm)
            self.update = opt.apply_gradients(zip(clipped_gradients, self.params), 
                    global_step=self.global_step)

        else:
            # rnn decoder
            topic_memory = self.update_memory(topic_memory, encoder_output)
            extra_info = tf.concat([self.label_embedding, latent_sample, topic_memory], 1)
            decoder_fn_inference = my_attention_decoder_fn.attention_decoder_fn_inference(output_fn, 
                dec_fc2, attention_keys, attention_values, attention_score_fn, 
                attention_construct_fn, self.embed, GO_ID, EOS_ID, max_length, num_symbols, extra_info)
            self.decoder_distribution, _, _ = my_seq2seq.dynamic_rnn_decoder(cell_dec, decoder_fn_inference, scope="decoder")
            self.generation_index = tf.argmax(tf.split(self.decoder_distribution,
                [2, num_symbols-2], 2)[1], 2) + 2 # for removing UNK
            self.generation = tf.nn.embedding_lookup(self.symbols, self.generation_index)
            
            self.params = tf.trainable_variables()

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2, 
                max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def read_memory(self, memory, enc_state):
        mem_slot = self.mem_slot_num
        mem_size = self.mem_size
        with tf.variable_scope("read_memory"):
            w_d = tf.get_variable(name="w_d", initializer=tf.random_normal(
                [mem_size, mem_size], stddev=0.1))
            w_e = tf.get_variable(name="w_e", initializer=tf.random_normal(
                [mem_size, mem_size], stddev=0.1))
            v = tf.get_variable(name="v", initializer=tf.random_normal(
                [mem_size], stddev=0.1))

            # Compute attention.
            m_ = tf.matmul(tf.reshape(memory, [-1, mem_size]), w_d),
            m = tf.reshape(m_, [-1, mem_slot, mem_size])
            s_ = tf.matmul(enc_state, w_e)
            s = tf.expand_dims(s_, axis=1)
            a = tf.reduce_sum(tf.tanh(m + s) * v, axis=-1)

            # Compute scores.
            aligns = tf.nn.softmax(a)
            max_val = tf.reduce_max(aligns, axis=1)
            max_val = tf.expand_dims(max_val, axis=1)
            if self.mode == "train":
                scores = tf.tanh(1e3 * (aligns - max_val)) + 1.0
            elif self.mode == "predict":
                scores = tf.sign(aligns - max_val) + 1.0
            final_scores = tf.expand_dims(scores, axis=-1)

            # Update memory.
            tiled_state = tf.tile(s_, [1, mem_slot])
            final_state = tf.reshape(tiled_state, [-1, mem_slot, mem_size])
            memory_result = final_scores * final_state

        return memory_result

    def print_parameters(self):
        for item in self.params:
            print('%s: %s' % (item.name, item.get_shape()))
    
    def step_decoder(self, session, data, forward_only=False, global_t=None):
        input_feed = {self.ori_sents: data['ori_sents'],
                self.ori_sents_length: data['ori_sents_length'],
                self.rep_sents: data['rep_sents'],
                self.rep_sents_length: data['rep_sents_length'],
                self.labels: data['labels'],
                self.use_prior: False}
        if global_t is not None:
            input_feed[self.global_t] = global_t
        if forward_only:  #On the dev set
            output_feed = [self.loss, self.klloss, self.decoder_loss, self.classifierloss]
        else:
            output_feed = [self.loss, self.klloss, self.decoder_loss, self.classifierloss, self.gradient_norm, self.update]
        return session.run(output_feed, input_feed)
    
    def inference(self, session, data, label_no):
        if label_no == 0:
            temp_labels = np.tile(np.array([1, 0, 0]),(len(data['ori_sents']),1))
        else:
            if label_no == 1:
                temp_labels = np.tile(np.array([0, 1, 0]), (len(data['ori_sents']), 1))
            else:
                temp_labels = np.tile(np.array([0, 0, 1]), (len(data['ori_sents']), 1))
        input_feed = {self.ori_sents: data['ori_sents'], self.ori_sents_length: data['ori_sents_length'],
                      self.rep_sents: data['ori_sents'], self.rep_sents_length: data['rep_sents_length'],
                      self.labels: temp_labels, self.use_prior: True}
        output_feed = [self.generation]
        return session.run(output_feed, input_feed)
