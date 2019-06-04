'''
   Created by zhiqliu on 2018/12/18.
'''

print('Importing')
import numpy as np
import tensorflow as tf
import os
from datetime import datetime
import argparse
import tensorflow as tf
from classifier_train import BiLSTM_Attention_Model
import os
import conf

FLAGS = tf.flags.FLAGS
'''
tf.flags.DEFINE_integer('lstm_size', 256, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_integer('embedding_size', 256, 'size of embedding')
tf.flags.DEFINE_string('converter_path', 'model/model_0731/converter.pkl', 'model/name/converter.pkl')
tf.flags.DEFINE_string('checkpoint_path', 'model/model_0731', 'checkpoint path')
tf.flags.DEFINE_string('start_string', u'', 'use this string to start generating')
tf.flags.DEFINE_integer('max_length', 200, 'max length to generate')
'''

def flatten(x):
    result = []
    for el in x:
         if isinstance(el, tuple):
               result.extend(flatten(el))
         else:
               result.append(el)
    return result

def get_config():
    if FLAGS.model == 'bilstm_attention_mdl':
        return conf.DefaultConfig()
    else:
        raise ValueError('Invalid model: %s', FLAGS.model)


def save_models():
    config = get_config()
    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
        with tf.variable_scope(FLAGS.model, reuse=None, initializer=initializer):
            model = BiLSTM_Attention_Model(is_training=False, config=config)
        tf.initialize_all_variables().run()
        # add ops to save and restore all the variables.
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('../model/training-ckpt/bilstm_attention_mdl/')

        if ckpt and ckpt.model_checkpoint_path:
            print ('loading......')
            saver.restore(session, ckpt.model_checkpoint_path)
        builder = tf.saved_model.builder.SavedModelBuilder("model_pb")
        print (model.input_data,model.targets)
        inputs = {'input_x': tf.saved_model.utils.build_tensor_info(model.input_data),
                  'labels': tf.saved_model.utils.build_tensor_info(model.targets)}

        # y 为最终需要的输出结果tensor
        print (model.predicts)
        print (model.logits)
        outputs = {'predict': tf.saved_model.utils.build_tensor_info(model.predicts),
                   'score': tf.saved_model.utils.build_tensor_info(model.logits)}

        signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs, 'test_sig_name')

        builder.add_meta_graph_and_variables(session, ['test_saved_model'], {'test_signature': signature})
        builder.save()
        print('Frozen graph saved.')


def load_models(_x, _target):
    with tf.Session() as sess:
        signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        input_key = 'input_x'
        input_labels = 'labels'

        #proba_prediction_key = 'predict'
        score_key = 'score'
        meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], "model_pb")
        # 从meta_graph_def中取出SignatureDef对象
        signature = meta_graph_def.signature_def

        # 从signature中找出具体输入输出的tensor name
        x_tensor_name = signature[signature_key].inputs[input_key].name
        label_y = signature[signature_key].inputs[input_labels].name

        #proba_prediction_name = signature[signature_key].outputs[proba_prediction_key].name
        score_name = signature[signature_key].outputs[score_key].name
        #print (proba_prediction_name, score_name)
        # 获取tensor 并inference
        x = sess.graph.get_tensor_by_name(x_tensor_name)
        target = sess.graph.get_tensor_by_name(label_y)
        #preds = sess.graph.get_tensor_by_name(proba_prediction_name)
        scores = sess.graph.get_tensor_by_name(score_name)

        feed = {x: _x, target: _target}
        score = sess.run(scores,feed_dict=feed)
        print (score)


def maxminnorm(array):
    mincols=array.min(axis=0)
    t = []
    array = [i + abs(mincols)+0.1 for i in array]
    print (array)
    sums = sum(array)
    for i in range(len(array)):
        t.append(array[i]/sums)
    return t


def freeze_model(checkpoint, save_path):
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  #config.batch_size = 1
  #config = get_config()
  with tf.Session(config=config) as sess:
    print('Starting session')
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(checkpoint))
    print (ckpt)
    print('Restoring meta graph')
    saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta')
    print('Restoring model at {}'.format(ckpt.model_checkpoint_path))
    saver.restore(sess, ckpt.model_checkpoint_path)

    print('Loading ops')
    print(tf.get_collection('input_x'))
    input_data = tf.get_collection('input_x')[0]
    print(tf.get_collection('labels'))
    target = tf.get_collection('labels')[0]
    print (tf.get_collection('score'))
    pred_scores = tf.get_collection('score')[0]

    print('Make builder')
    builder = tf.saved_model.builder.SavedModelBuilder(save_path)
    info_in1 = tf.saved_model.utils.build_tensor_info(input_data)
    info_in2 = tf.saved_model.utils.build_tensor_info(target)


    info_pred_scores = tf.saved_model.utils.build_tensor_info(pred_scores)

    input_dict = {'input_x': info_in1,
                  'labels': info_in2}

    output_dict = {'score': info_pred_scores}

    prediction_signature = (tf.saved_model.signature_def_utils.build_signature_def(
        inputs=input_dict, outputs=output_dict,
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    builder.add_meta_graph_and_variables(
      sess, [tf.saved_model.tag_constants.SERVING],
      signature_def_map={
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature})

    builder.save()
    print('Frozen graph saved.')



if __name__ == "__main__":
    #save_models()
    #freeze_model('../model/training-ckpt/bilstm_attention_mdl/', 'model_pb')
    string = '212_86_1138_1_227_425_8139_988_30_150_22_0_372_95_14_244_22_1454_485_0_0_0_0_0_0_0_0_0_0_0'
    lists = [int(i) for i in string.split('_')]
    _x = [lists]
    _target = [[0]]
    load_models(_x, _target)
    #freeze_model('../model/training-ckpt/bilstm_attention_mdl/', 'freeze_model')



