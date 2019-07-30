import tensorflow as tf
from models.tpa import SingleTPA
from models.tpa_single_lstm import TPASINGLELSTM
from models.tpa_bidirectional_lstm import TPABIDIRECTIONALLSTM


def config_setup():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    return config


def create_graph(para, mode, version):
    if mode == "train":
        para.mode = 'train'
    else:
        para.mode = 'eval'

    graph = tf.Graph()
    with graph.as_default():
        with tf.variable_scope('model'):
            model = None
            if version == "single_tpa":
                model = SingleTPA(para)
            elif version == "tpa_single_lstm":
                model = TPASINGLELSTM(para)
            elif version == "tpa_bidirectional_lstm":
                model = TPABIDIRECTIONALLSTM(para)
            else:
                raise ValueError('can not find model')
    return graph, model


def train_load_weight(model_dir, sess, model):
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt:
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())


def eval_load_weight(model_dir, sess, model):
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt:
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise FileNotFoundError('can not find model checkpoint')