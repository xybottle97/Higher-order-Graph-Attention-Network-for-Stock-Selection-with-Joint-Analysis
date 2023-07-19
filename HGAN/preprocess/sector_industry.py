import argparse
import copy
import json
import os

import numpy as np


class SectorPreprocessor:
    def __init__(self, data_path, market_name):
        self.data_path = data_path
        self.date_format = '%Y-%m-%d %H:%M:%S'
        self.market_name = market_name

    def generate_sector_relation(self, industry_ticker_file,
                                 selected_tickers_fname):
        selected_tickers = np.genfromtxt(
            os.path.join(self.data_path,  selected_tickers_fname),
            dtype=str, delimiter='\t', skip_header=False
        )
        print('#tickers selected:', len(selected_tickers))
        ticker_index = {}
        for index, ticker in enumerate(selected_tickers):
            ticker_index[ticker] = index
        with open(industry_ticker_file, 'r') as fin:
            industry_tickers = json.load(fin)
        print('#industries: ', len(industry_tickers))
        valid_industry_count = 0
        valid_industry_index = {}
        connection_count=0
        for industry in industry_tickers.keys():
            connection_count+=len(industry_tickers[industry])*(len(industry_tickers[industry])-1)
            if len(industry_tickers[industry]) > 1:
                valid_industry_index[industry] = valid_industry_count
                valid_industry_count += 1
        one_hot_industry_embedding = np.identity(valid_industry_count + 1,
                                                 dtype=int)
        ticker_relation_embedding = np.zeros(
            [len(selected_tickers), len(selected_tickers),
             valid_industry_count + 1], dtype=int)
        print(ticker_relation_embedding[0][0].shape)
        print('connections count:', connection_count, 'ratio:', connection_count / float(len(selected_tickers) * len(selected_tickers)))
        for industry in valid_industry_index.keys():
            cur_ind_tickers = industry_tickers[industry]
            if len(cur_ind_tickers) <= 1:
                print('shit industry:', industry)
                continue
            ind_ind = valid_industry_index[industry]
            for i in range(len(cur_ind_tickers)):
                if cur_ind_tickers[i] in ticker_index.keys():
                    left_tic_ind = ticker_index[cur_ind_tickers[i]]
                    ticker_relation_embedding[left_tic_ind][left_tic_ind] = \
                        copy.copy(one_hot_industry_embedding[ind_ind])
                    ticker_relation_embedding[left_tic_ind][left_tic_ind][-1] = 1
                for j in range(i + 1, len(cur_ind_tickers)):
                    if cur_ind_tickers[i] in ticker_index.keys() and cur_ind_tickers[j] in ticker_index.keys():
                        right_tic_ind = ticker_index[cur_ind_tickers[j]]
                        ticker_relation_embedding[left_tic_ind][right_tic_ind] = \
                            copy.copy(one_hot_industry_embedding[ind_ind])
                        ticker_relation_embedding[right_tic_ind][left_tic_ind] = \
                            copy.copy(one_hot_industry_embedding[ind_ind])

        # handle shit industry and n/a tickers
        for i in range(len(selected_tickers)):
            ticker_relation_embedding[i][i][-1] = 1
        print(ticker_relation_embedding.shape)


if __name__ == '__main__':
    desc = "pre-process sector data market by market, including listing all " \
           "trading days, all satisfied stocks (5 years & high price), " \
           "normalizing and compansating data"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-path', help='path of EOD data')
    parser.add_argument('-market', help='market name')
    args = parser.parse_args()

    if args.path is None:
        args.path = 'E:/桌面/RSSR/data'
    if args.market is None:
        args.market = 'NASDAQ'

    processor = SectorPreprocessor(args.path, args.market)

    processor.generate_sector_relation(
        (args.path+'/relation/sector_industry/'+
                     processor.market_name +'_industry_ticker_2022.json'),
        processor.market_name + '_final_list_new.csv'
    )