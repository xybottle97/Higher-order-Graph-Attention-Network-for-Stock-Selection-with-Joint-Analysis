
import os

import numpy as np
import tensorflow as tf
from motifcluster import motifadjacency
from tensorflow.python.ops.nn_ops import leaky_relu


def load_EOD_data(data_path, market_name, tickers, steps=1):
    eod_data = []
    masks = []
    ground_truth = []
    base_price = []
    for index, ticker in enumerate(tickers):
        single_EOD = np.genfromtxt(
            os.path.join(data_path, market_name + '_' + ticker + '_final.csv'),
            dtype=np.float32, delimiter=',', skip_header=False
        )
        if market_name == 'NASDAQ':
            #remove the last day since lots of missing data
            single_EOD = single_EOD[:-1, :]
        if index == 0:
            print('single EOD data shape:', single_EOD.shape)
            eod_data = np.zeros([len(tickers), single_EOD.shape[0],
                                 single_EOD.shape[1] - 1], dtype=np.float32)
            masks = np.ones([len(tickers), single_EOD.shape[0]],
                            dtype=np.float32)
            ground_truth = np.zeros([len(tickers), single_EOD.shape[0]],
                                    dtype=np.float32)
            base_price = np.zeros([len(tickers), single_EOD.shape[0]],
                                  dtype=np.float32)
        for row in range(single_EOD.shape[0]):
            if abs(single_EOD[row][5] + 1234) < 1e-8:
                masks[index][row] = 0.0
            elif row > steps - 1 and abs(single_EOD[row - steps][5] + 1234) \
                    > 1e-8:
                ground_truth[index][row] = \
                    (single_EOD[row][5] - single_EOD[row - steps][5]) / \
                    single_EOD[row - steps][5]
            for col in range(single_EOD.shape[1]):
                if abs(single_EOD[row][col] + 1234) < 1e-8:
                    single_EOD[row][col] = 1.1
        eod_data[index, :, :] = single_EOD[:, 1:]
        base_price[index, :] = single_EOD[:, 5]
    return eod_data, masks, ground_truth, base_price


def load_real_EOD_data(data_path, market_name, tickers, steps=1):
    eod_data = []
    masks = []
    ground_truth = []
    base_price = []
    for index, ticker in enumerate(tickers):
        single_EOD = np.genfromtxt(
            os.path.join(data_path, market_name + '_' + ticker + '_1_2022.csv'),
            dtype=np.float32, delimiter=',', skip_header=False
        )
        if index == 0:
            print('single EOD data shape:', single_EOD.shape)
            eod_data = np.zeros([len(tickers), single_EOD.shape[0],
                                 single_EOD.shape[1] - 1], dtype=np.float32)
            masks = np.ones([len(tickers), single_EOD.shape[0]],
                            dtype=np.float32)
            ground_truth = np.zeros([len(tickers), single_EOD.shape[0]],
                                    dtype=np.float32)
            base_price = np.zeros([len(tickers), single_EOD.shape[0]],
                                  dtype=np.float32)
        for row in range(single_EOD.shape[0]):
            if abs(single_EOD[row][-1] + 1234) < 1e-8:
                masks[index][row] = 0.0
            elif row > steps - 1 and abs(single_EOD[row - steps][-1] + 1234) \
                    > 1e-8:
                ground_truth[index][row] = \
                    (single_EOD[row][-1] - single_EOD[row - steps][-1]) / \
                    single_EOD[row - steps][-1]
            for col in range(single_EOD.shape[1]):
                if abs(single_EOD[row][col] + 1234) < 1e-8:
                    single_EOD[row][col] = 1.1
        eod_data[index, :, :] = single_EOD[:, 1:]
        base_price[index, :] = single_EOD[:, -1]
    return eod_data, masks, ground_truth, base_price

def load_graph_relation_data(relation_file, lap=False):
    relation_encoding = np.load(relation_file)
    print('relation encoding shape:', relation_encoding.shape)
    rel_shape = [relation_encoding.shape[0], relation_encoding.shape[1]]
    mask_flags = np.equal(np.zeros(rel_shape, dtype=int),
                          np.sum(relation_encoding, axis=2))
    ajacent = np.where(mask_flags, np.zeros(rel_shape, dtype=float),
                       np.ones(rel_shape, dtype=float))
    degree = np.sum(ajacent, axis=0)
    for i in range(len(degree)):
        degree[i] = 1.0 / degree[i]
    np.sqrt(degree, degree)
    deg_neg_half_power = np.diag(degree)
    if lap:
        return np.identity(ajacent.shape[0], dtype=float) - np.dot(
            np.dot(deg_neg_half_power, ajacent), deg_neg_half_power)
    else:
        return np.dot(np.dot(deg_neg_half_power, ajacent), deg_neg_half_power)

def load_relation_data(relation_file_wiki,relation_file_industry,motif_file):
    relation_encoding_wiki = np.load(relation_file_wiki)
    relation_encoding_industry = np.load(relation_file_industry)
    relation_encoding=np.concatenate((relation_encoding_wiki,relation_encoding_industry),axis=2)
    print('wiki relation encoding shape:', relation_encoding_wiki.shape)
    print('industry relation encoding shape:', relation_encoding_industry.shape)
    rel_shape = [relation_encoding_wiki.shape[0], relation_encoding_wiki.shape[1]]
    relation=tf.constant(relation_encoding_wiki, dtype=tf.float32)
    relation = tf.layers.dense(relation, units=1, activation= leaky_relu)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    rel = relation.eval()
    sess.close()
    mask_flags = np.equal(np.zeros(rel_shape, dtype=int),
                          rel[:, :, -1])
    mask = np.where(mask_flags, np.ones(rel_shape) * -1e9, np.zeros(rel_shape))

    motif=relation_encoding
    # -------------------------------------------------------
    for i in ['M3','M4','M8','M10','M13']:
        print(i)
        motif_Mi = motifadjacency.build_motif_adjacency_matrix(np.squeeze(rel), i, "struc", "mean")
        motif_Mi = motif_Mi.todense()
        motif_Mi = (motif_Mi-np.min(motif_Mi)) / (np.max(motif_Mi)-np.min(motif_Mi))
        motif_Mi=np.expand_dims(motif_Mi, axis=2)
        motif=np.concatenate([motif,motif_Mi],axis=2)

    print(motif.shape)




    return motif,relation_encoding,mask

def build_SFM_data(data_path, market_name, tickers):
    eod_data = []
    for index, ticker in enumerate(tickers):
        single_EOD = np.genfromtxt(
            os.path.join(data_path, market_name + '_' + ticker + '_1_2022.csv'),
            dtype=np.float32, delimiter=',', skip_header=False
        )
        if index == 0:
            print('single EOD data shape:', single_EOD.shape)
            eod_data = np.zeros([len(tickers), single_EOD.shape[0]],
                                dtype=np.float32)

        for row in range(single_EOD.shape[0]):
            if abs(single_EOD[row][-1] + 1234) < 1e-8:
                # handle missing data
                if row < 3:
                    # eod_data[index, row] = 0.0
                    for i in range(row + 1, single_EOD.shape[0]):
                        if abs(single_EOD[i][-1] + 1234) > 1e-8:
                            eod_data[index][row] = single_EOD[i][-1]
                            # print(index, row, i, eod_data[index][row])
                            break
                else:
                    eod_data[index][row] = np.sum(
                        eod_data[index, row - 3:row]) / 3
                    # print(index, row, eod_data[index][row])
            else:
                eod_data[index][row] = single_EOD[row][-1]
        # print('test point')
    #np.save(market_name + '_sfm_data', eod_data)
    #np.save(r'E:/桌面/RSSR/data' + market_name + '_wiki_relation.npy', eod_data)