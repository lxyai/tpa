import tensorflow as tf
import numpy as np
import os
from utils.lib import config_setup

def train(para, model, mts):
    with tf.Session(config=config_setup()) as sess:
        ckpt = tf.train.get_checkpoint_state(para.model_dir)
        if ckpt:
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        for i in range(para.num_epoch):
            print('Starting train....')
            epoch = mts.train_status['epoch']
            while epoch == mts.train_status['epoch']:
                x, y = mts.get_train_batch()
                loss, global_step, _ = sess.run([model.loss, model.global_step, model.update],
                                                feed_dict={model.x: x,
                                                           model.y: y,
                                                           model.model: 'train'})

                if global_step % 10 == 0:
                    print('global step:{} loss:{} cost time'.format(global_step, loss))
            [global_step] = sess.run([model.global_step])
            model.saver.save(sess, os.path.join(para.model_dir, 'model.ckpt'), global_step=global_step)

            all_outputs = []
            all_labels = []
            valid_loss = 0
            while not mts.validation_status['stop']:
                x, y = mts.get_validation_batch()
                loss, outputs, labels = sess.run([model.loss, model.all_rnn_outputs, model.labels],
                                                 feed_dict={model.x: x,
                                                            model.y: y,
                                                            model.mode: 'validation'})

                all_outputs.append(outputs)
                all_labels.append(labels)
                valid_loss += loss

            mts.valiation_initailizer()
            all_outputs = np.concatenate(all_outputs, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            valid_mape = np.mean(np.abs((all_outputs - all_labels) / all_labels))
            print('valid_mape:{} valid_loss:{}'.format(valid_mape, valid_loss))


