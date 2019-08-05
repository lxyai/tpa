import logging
import tensorflow as tf
from tensorflow.layers import dense
from measure_map_of_tpa.attention_wrapper import TemporalPatternAttentionCellWrapper


class PolyRNN:
    def __init__(self, para):
        self.para = para
        self.dtype = tf.float32
        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        self._build_graph()
        if self.para.mode == "train":
            self._build_optimizer()

        self.saver = tf.train.Saver(max_to_keep=3)

    def _build_graph(self):
        print("Building {} graph".format(self.para['mode']))

        # rnn_inputs: [batch_size, max_len, input_size]
        # rnn_inputs_len: [batch_size]
        # target_outputs: [batch_size, max_len, output_size]
        # self.rnn_inputs, self.rnn_inputs_len, self.target_outputs = self.data_generator.inputs(
        #     self.para.mode, self.para.batch_size)
        x_length = self.para.data_config['x']['range'][1] - self.para.data_config['x']['range'][0] + 1
        y_length = self.para.data_config['y']['range'][1] - self.para.data_config['y']['range'][0] + 1
        x_size = len(self.para.data_config['x']['key'])
        y_size = len(self.para.data_config['y']['key'])
        self.x = tf.placeholder(self.dtype, [None, x_length, x_size], name='x')
        self.y = tf.placeholder(self.dtype, [None, y_length, y_size])

        # rnn_inputs_embed: [batch_size, max_len, num_units]
        self.rnn_inputs_embed = tf.nn.relu(
            dense(self.x, self.para.num_units))

        # all_rnn_states: [batch_size, max_len, num_units]
        # final_rnn_states: [LSTMStateTuple], len = num_layers
        # LSTMStateTuple: (c: [batch_size, num_units],
        #                  h: [batch_size, num_units])
        # self.rnn_inputs_embed = tf.unstack(self.rnn_inputs_embed, axis=1)
        self.all_rnn_states, self.final_rnn_states = tf.nn.dynamic_rnn(
            cell=self._build_rnn_cell(),
            inputs=self.rnn_inputs_embed,
            dtype=self.dtype,
        )

        # final_rnn_states: [batch_size, num_units]
        self.final_rnn_states = tf.concat(
            [self.final_rnn_states[i][1] for i in range(self.para.num_layers)],
            1,
        )

        # all_rnn_outputs: [batch_size, output_size]
        self.all_rnn_outputs = dense(self.final_rnn_states,
                                     y_size)

        if self.para.highway > 0:
            reg_outputs = tf.transpose(
                self.x[:, -self.para.highway:, :], [0, 2, 1])
            reg_outputs = dense(reg_outputs, 1)
            self.all_rnn_outputs += tf.squeeze(reg_outputs)

        self.labels = tf.squeeze(self.y)

        if self.para.mode == "train" or self.para.mode == "validation":
            self.loss = self._compute_loss(
                outputs=self.all_rnn_outputs, labels=self.labels)

    def _build_optimizer(self):
        print("Building optimizer")

        trainable_variables = tf.trainable_variables()
        if self.para.decay > 0:
            lr = tf.train.exponential_decay(
                self.para.learning_rate,
                self.global_step,
                self.para.decay,
                0.995,
                staircase=True,
            )
        else:
            lr = self.para.learning_rate
        self.opt = tf.train.AdamOptimizer(lr)
        gradients = tf.gradients(self.loss, trainable_variables)
        clip_gradients, _ = tf.clip_by_global_norm(gradients,
                                                   self.para.max_gradient_norm)
        self.update = self.opt.apply_gradients(
            zip(clip_gradients, trainable_variables),
            global_step=self.global_step,
        )

    def _compute_loss(self, outputs, labels):
        """
        outputs: [batch_size, output_size]
        labels: [batch_size, output_size]
        """
        loss = tf.reduce_mean(
            tf.losses.absolute_difference(
                labels=labels, predictions=outputs))
        return loss

    def _build_single_cell(self):
        cell = tf.contrib.rnn.LSTMBlockCell(self.para.num_units)
        if self.para.mode == "train":
            cell = tf.contrib.rnn.DropoutWrapper(
                cell=cell,
                input_keep_prob=(1.0 - self.para.dropout),
                output_keep_prob=(1.0 - self.para.dropout),
                state_keep_prob=(1.0 - self.para.dropout),
            )
        cell = TemporalPatternAttentionCellWrapper(
            cell,
            self.para.attention_len,
        )
        return cell

    def _build_rnn_cell(self):
        return tf.contrib.rnn.MultiRNNCell(
            [self._build_single_cell() for _ in range(self.para.num_layers)])
