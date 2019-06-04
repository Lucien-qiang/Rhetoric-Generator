'''
   Created by zhiqliu on 2018/12/18.
'''


class Converter(object):
    label2domain = {
        0:'No-rhetoric',
        1:'Metaphor',
        2:'Personification'

    }

class DefaultConfig(object):
    """New config."""
    init_scale = 0.1
    learning_rate = 0.1
    max_grad_norm = 10
    num_layers = 2
    num_steps = 30
    hidden_size = 256
    max_epoch = 100
    keep_prob = 0.8
    lr_decay = 0.5
    perplexity_thres = 0.001
    batch_size = 128
    num_labels = 3
    vocab_size = 3834
    num_top_labels = 3
    attention_size = 30
    converter = Converter()
