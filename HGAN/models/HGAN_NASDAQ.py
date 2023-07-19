import argparse
import copy
import os
import random
from time import time
from warnings import simplefilter

import numpy as np
import tensorflow as tf

from evaluator import evaluate
from load_data import load_EOD_data, load_relation_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
simplefilter(action='ignore', category=FutureWarning)

try:
    from tensorflow.python.ops.nn_ops import leaky_relu
except ImportError:
    from tensorflow.python.framework import ops
    from tensorflow.python.ops import math_ops


    def leaky_relu(features, alpha=0.2, name=None):
        with ops.name_scope(name, "LeakyRelu", [features, alpha]):
            features = ops.convert_to_tensor(features, name="features")
            alpha = ops.convert_to_tensor(alpha, name="alpha")
            return math_ops.maximum(alpha * features, features)

seed = 1234
np.random.seed(seed)
tf.set_random_seed(seed)


class ReRaLSTM:
    def __init__(self, data_path, market_name, tickers_fname, relation_name,
                 parameters, steps=1, epochs=50, batch_size=None, flat=False, gpu=False, in_pro=False):

        seed = 1234
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)

        self.data_path = data_path
        self.market_name = market_name
        self.tickers_fname = tickers_fname
        self.relation_name = relation_name
        # load data
        self.tickers = np.genfromtxt(os.path.join(data_path, '..', tickers_fname),
                                     dtype=str, delimiter='\t', skip_header=False)

        print('#tickers selected:', len(self.tickers))
        self.eod_data, self.mask_data, self.gt_data, self.price_data = \
            load_EOD_data(data_path, market_name, self.tickers, steps)

        # relation data
        rname_tail = {'sector_industry': '_industry_relation_2022_new.npy',
                      'wikidata': '_wiki_relation_2022_new.npy'}

        self.motif, self.rel_encoding, self.rel_mask = load_relation_data(
            os.path.join(self.data_path, '..', 'relation', self.relation_name,
                         self.market_name + rname_tail[self.relation_name]),
            os.path.join(self.data_path, '..', 'relation', 'sector_industry',
                         self.market_name + '_industry_relation_2022_new.npy'),
            os.path.join(self.data_path, '..', 'relation', self.relation_name,
                         self.market_name + '_wiki_relation_2022_motif.npy')
        )
        print('relation encoding shape:', self.rel_encoding.shape)
        print('relation mask shape:', self.rel_mask.shape)

        self.parameters = copy.copy(parameters)
        self.steps = steps
        self.epochs = epochs
        self.flat = flat
        self.inner_prod = in_pro
        if batch_size is None:
            self.batch_size = len(self.tickers)
        else:
            self.batch_size = batch_size

        self.trade_dates = self.mask_data.shape[1]
        self.time_start = 221
        self.valid_index = 978
        self.test_index = 1230
        self.month = 12

        self.fea_dim = 10
        self.gpu = gpu

    def get_batch(self, offset=None):
        if offset is None:
            offset = random.randrange(self.time_start, self.valid_index)
        seq_len = self.parameters['seq']
        mask_batch = self.mask_data[:, offset: offset + seq_len + self.steps]
        mask_batch = np.min(mask_batch, axis=1)
        return self.eod_data[:, offset:offset + seq_len, :], \
               np.expand_dims(mask_batch, axis=1), \
               np.expand_dims(
                   self.price_data[:, offset + seq_len - 1], axis=1
               ), \
               np.expand_dims(
                   self.gt_data[:, offset + seq_len + self.steps - 1], axis=1
               )

    def train(self):
        if self.gpu == True:
            device_name = '/gpu:0'
        else:
            device_name = '/cpu:0'
        print('device name:', device_name)
        with tf.device(device_name):
            tf.reset_default_graph()

            seed = 1234
            random.seed(seed)
            np.random.seed(seed)
            tf.set_random_seed(seed)

            ground_truth = tf.placeholder(tf.float32, [self.batch_size, 1])
            mask = tf.placeholder(tf.float32, [self.batch_size, 1])
            feature_eod = tf.placeholder(tf.float32,
                                         [self.batch_size, self.parameters['seq'], self.fea_dim])

            base_price = tf.placeholder(tf.float32, [self.batch_size, 1])
            all_one = tf.ones([self.batch_size, 1], dtype=tf.float32)

            relation = tf.constant(self.motif, dtype=tf.float32)
            rel_mask = tf.constant(self.rel_mask, dtype=tf.float32)
            motif = tf.layers.dense(relation, units=1,
                                    activation=leaky_relu)

            with tf.variable_scope('eod'):
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(
                    self.parameters['unit']
                )
                initial_state = lstm_cell.zero_state(self.batch_size,
                                                     dtype=tf.float32)
                outputs, final_state = tf.nn.dynamic_rnn(
                    lstm_cell, feature_eod, dtype=tf.float32,
                    initial_state=initial_state
                )
                feature_10 = outputs[:, -1, :]
            with tf.variable_scope('attention'):
                lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(
                    self.parameters['unit']
                )
                initial_state_1 = lstm_cell_1.zero_state(self.batch_size,
                                                         dtype=tf.float32)
                outputs_1, final_state_1 = tf.nn.dynamic_rnn(
                    lstm_cell_1, feature_eod[:, :, :5], dtype=tf.float32,
                    initial_state=initial_state_1
                )
                feature = outputs_1[:, -1, :]

            if self.inner_prod:
                inner_weight = tf.matmul(feature, feature, transpose_b=True)
                weight = tf.multiply(inner_weight, rel_weight[:, :, -1])
            else:
                weight = tf.zeros([self.batch_size, self.batch_size], dtype=tf.float32)
                head = self.parameters['head']
                unit = self.parameters['unit']

                for i in range(head):
                    feature_tmp = feature_10[:, i * (unit // head): (i + 1) * (unit // head)]

                    head_weight = tf.layers.dense(feature_tmp, units=1,
                                                  activation=leaky_relu)
                    tail_weight = tf.layers.dense(feature_tmp, units=1,
                                                  activation=leaky_relu)
                    weight += tf.add(
                        tf.add(
                            tf.matmul(head_weight, all_one, transpose_b=True),
                            tf.matmul(all_one, tail_weight, transpose_b=True)
                        ), motif[:, :, -1])

                weight /= head
            weight_masked = tf.nn.softmax(tf.add(rel_mask, weight), dim=0)
            outputs_proped = tf.matmul(weight_masked, feature)

            if self.flat:

                outputs_concated = tf.layers.dense(
                    tf.concat([feature_10, outputs_proped], axis=1),
                    units=self.parameters['unit'], activation=leaky_relu,
                    kernel_initializer=tf.glorot_uniform_initializer()
                )
            else:
                outputs_concated = tf.concat([feature, outputs_proped], axis=1)

            # FC Layer
            prediction = tf.layers.dense(
                outputs_concated, units=1, activation=leaky_relu, name='reg_fc',
                kernel_initializer=tf.glorot_uniform_initializer()
            )

            return_ratio = tf.div(tf.subtract(prediction, base_price), base_price)
            reg_loss = tf.losses.mean_squared_error(
                ground_truth, return_ratio, weights=mask
            )
            pre_pw_dif = tf.subtract(
                tf.matmul(return_ratio, all_one, transpose_b=True),
                tf.matmul(all_one, return_ratio, transpose_b=True)
            )
            gt_pw_dif = tf.subtract(
                tf.matmul(all_one, ground_truth, transpose_b=True),
                tf.matmul(ground_truth, all_one, transpose_b=True)
            )
            mask_pw = tf.matmul(mask, mask, transpose_b=True)
            rank_loss = tf.reduce_mean(
                tf.nn.relu(
                    tf.multiply(
                        tf.multiply(pre_pw_dif, gt_pw_dif),
                        mask_pw
                    )
                )
            )
            loss = reg_loss + tf.cast(self.parameters['alpha'], tf.float32) * \
                   rank_loss
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.parameters['lr']
            ).minimize(loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        best_valid_pred = np.zeros(
            [len(self.tickers), self.test_index - self.valid_index],
            dtype=float
        )
        best_valid_gt = np.zeros(
            [len(self.tickers), self.test_index - self.valid_index],
            dtype=float
        )
        best_valid_mask = np.zeros(
            [len(self.tickers), self.test_index - self.valid_index],
            dtype=float
        )
        best_test_pred = np.zeros(
            [len(self.tickers), self.trade_dates - self.parameters['seq'] -
             self.test_index - self.steps + 1], dtype=float
        )
        best_test_gt = np.zeros(
            [len(self.tickers), self.trade_dates - self.parameters['seq'] -
             self.test_index - self.steps + 1], dtype=float
        )
        best_test_mask = np.zeros(
            [len(self.tickers), self.trade_dates - self.parameters['seq'] -
             self.test_index - self.steps + 1], dtype=float
        )
        best_test_perf = {
            'mse': np.inf, 'mrrt': 0.0, 'btl': 0.0
        }
        best_valid_loss = np.inf

        batch_offsets = np.arange(start=self.time_start, stop=self.valid_index, dtype=int)

        should_stop = False
        for i in range(self.epochs):
            t1 = time()
            np.random.shuffle(batch_offsets)
            tra_loss = 0.0
            tra_reg_loss = 0.0
            tra_rank_loss = 0.0
            for j in range(self.time_start, self.valid_index - self.parameters['seq'] -
                                            self.steps + 1):
                eod_batch, mask_batch, price_batch, gt_batch = self.get_batch(
                    batch_offsets[j - self.time_start])
                feed_dict = {
                    feature_eod: eod_batch,
                    mask: mask_batch,
                    ground_truth: gt_batch,
                    base_price: price_batch
                }
                cur_loss, cur_reg_loss, cur_rank_loss, batch_out = \
                    sess.run((loss, reg_loss, rank_loss, optimizer),
                             feed_dict)
                tra_loss += cur_loss
                tra_reg_loss += cur_reg_loss
                tra_rank_loss += cur_rank_loss

            # test on validation set
            cur_valid_pred = np.zeros(
                [len(self.tickers), self.test_index - self.valid_index],
                dtype=float
            )
            cur_valid_gt = np.zeros(
                [len(self.tickers), self.test_index - self.valid_index],
                dtype=float
            )
            cur_valid_mask = np.zeros(
                [len(self.tickers), self.test_index - self.valid_index],
                dtype=float
            )
            val_loss = 0.0
            val_reg_loss = 0.0
            val_rank_loss = 0.0

            for cur_offset in range(
                    self.valid_index - self.parameters['seq'] - self.steps + 1,
                    self.test_index - self.parameters['seq'] - self.steps + 1
            ):
                eod_batch, mask_batch, price_batch, gt_batch = self.get_batch(
                    cur_offset)
                feed_dict = {
                    feature_eod: eod_batch,
                    mask: mask_batch,
                    ground_truth: gt_batch,
                    base_price: price_batch
                }
                cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = \
                    sess.run((loss, reg_loss, rank_loss,
                              return_ratio), feed_dict)
                val_loss += cur_loss
                val_reg_loss += cur_reg_loss
                val_rank_loss += cur_rank_loss
                cur_valid_pred[:, cur_offset - (self.valid_index -
                                                self.parameters['seq'] -
                                                self.steps + 1)] = \
                    copy.copy(cur_rr[:, 0])
                cur_valid_gt[:, cur_offset - (self.valid_index -
                                              self.parameters['seq'] -
                                              self.steps + 1)] = \
                    copy.copy(gt_batch[:, 0])
                cur_valid_mask[:, cur_offset - (self.valid_index -
                                                self.parameters['seq'] -
                                                self.steps + 1)] = \
                    copy.copy(mask_batch[:, 0])
            cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt,
                                      cur_valid_mask, month=self.month)

            # test on testing set
            cur_test_pred = np.zeros(
                [len(self.tickers), self.trade_dates - self.test_index],
                dtype=float
            )
            cur_test_gt = np.zeros(
                [len(self.tickers), self.trade_dates - self.test_index],
                dtype=float
            )
            cur_test_mask = np.zeros(
                [len(self.tickers), self.trade_dates - self.test_index],
                dtype=float
            )
            test_loss = 0.0
            test_reg_loss = 0.0
            test_rank_loss = 0.0

            for cur_offset in range(
                    self.test_index - self.parameters['seq'] - self.steps + 1,
                    self.trade_dates - self.parameters['seq'] - self.steps + 1
            ):
                eod_batch, mask_batch, price_batch, gt_batch = self.get_batch(
                    cur_offset)
                feed_dict = {
                    feature_eod: eod_batch,
                    mask: mask_batch,
                    ground_truth: gt_batch,
                    base_price: price_batch
                }
                cur_loss, cur_reg_loss, cur_rank_loss, cur_rr, weightmasked = \
                    sess.run((loss, reg_loss, rank_loss,
                              return_ratio, weight_masked), feed_dict)
                test_loss += cur_loss
                test_reg_loss += cur_reg_loss
                test_rank_loss += cur_rank_loss

                cur_test_pred[:, cur_offset - (self.test_index -
                                               self.parameters['seq'] -
                                               self.steps + 1)] = \
                    copy.copy(cur_rr[:, 0])
                cur_test_gt[:, cur_offset - (self.test_index -
                                             self.parameters['seq'] -
                                             self.steps + 1)] = \
                    copy.copy(gt_batch[:, 0])
                cur_test_mask[:, cur_offset - (self.test_index -
                                               self.parameters['seq'] -
                                               self.steps + 1)] = \
                    copy.copy(mask_batch[:, 0])
            cur_test_perf = evaluate(cur_test_pred, cur_test_gt, cur_test_mask, month=self.month)
            if val_loss / (self.test_index - self.valid_index) < \
                    best_valid_loss:
                best_valid_loss = val_loss / (self.test_index -
                                              self.valid_index)
                best_valid_perf = copy.copy(cur_valid_perf)
                best_valid_gt = copy.copy(cur_valid_gt)
                best_valid_pred = copy.copy(cur_valid_pred)
                best_valid_mask = copy.copy(cur_valid_mask)
                best_test_perf = copy.copy(cur_test_perf)
                best_test_gt = copy.copy(cur_test_gt)
                best_test_pred = copy.copy(cur_test_pred)
                best_test_mask = copy.copy(cur_test_mask)
                print('hBetter valid loss:', best_valid_loss)
                print('Better test performance:', best_test_perf)
            t4 = time()
            print('epoch:', i, ('time: %.4f ' % (t4 - t1)))

        print('\nBest Test performance:', best_test_perf)

        sess.close()
        tf.reset_default_graph()
        return best_valid_pred, best_valid_gt, best_valid_mask, \
               best_test_pred, best_test_gt, best_test_mask

    def update_model(self, parameters):
        for name, value in parameters.items():
            self.parameters[name] = value
        return True


if __name__ == '__main__':
    desc = 'train a relational rank lstm model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-p', help='path of EOD data',
                        default='../data/dim10')
    parser.add_argument('-m', help='market name', default='NASDAQ')
    parser.add_argument('-t', help='fname for selected tickers')
    parser.add_argument('-l', default=16,
                        help='length of historical sequence for feature')
    parser.add_argument('-u', default=64,
                        help='number of hidden units in lstm')
    parser.add_argument('-s', default=1,
                        help='steps to make prediction')
    parser.add_argument('-r', default=0.001,
                        help='learning rate')
    parser.add_argument('-a', default=10,
                        help='alpha, the weight of ranking loss')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='use gpu')
    parser.add_argument('-head', type=int,
                        default=2,
                        help='multi-head number')
    parser.add_argument('-rn', '--rel_name', type=str,
                        default='wikidata',
                        help='relation type: sector_industry or wikidata')
    parser.add_argument('-ip', '--inner_prod', type=int, default=0)
    args = parser.parse_args()

    if args.t is None:
        args.t = args.m + '_final_list_new.csv'
    args.gpu = (args.gpu == 1)

    args.inner_prod = (args.inner_prod == 1)

    parameters = {'seq': int(args.l), 'unit': int(args.u), 'lr': float(args.r),
                  'alpha': float(args.a), 'head': int(args.head)}
    print('arguments:', args)
    print('parameters:', parameters)
    RR_LSTM = ReRaLSTM(
        data_path=args.p,
        market_name=args.m,
        tickers_fname=args.t,
        relation_name=args.rel_name,
        parameters=parameters,
        steps=1, epochs=50, batch_size=None, gpu=args.gpu,
        in_pro=args.inner_prod,
    )
    pred_all = RR_LSTM.train()
