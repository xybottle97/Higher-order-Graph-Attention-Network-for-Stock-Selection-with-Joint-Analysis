import os
from datetime import datetime

import numpy as np
import pandas as pd


def time_interval(cur_time, start_time, end_time):
    compare_date_format = '%Y-%m-%d'
    my_date_format = '%d/%m/%Y %H:%M:%S'
    curtime = datetime.strptime(cur_time, my_date_format)
    start_time = datetime.strptime(start_time, compare_date_format)
    end_time = datetime.strptime(end_time, compare_date_format)

    if start_time <= curtime <= end_time:
        return True
    else:
        return False

market_name='NYSE'
date_format = '%d/%m/%Y %H:%M:%S'
trading_dates = np.genfromtxt('D:/桌面/RSSR/data/'+market_name+'_aver_line_dates_2022.csv',dtype=str, delimiter=',', skip_header=False)
print('#trading dates:', len(trading_dates))
end_date = datetime.strptime(trading_dates[-1], date_format)



# transform the trading dates into a dictionary with index
index_tra_dates = {}
tra_dates_index = {}
for index, date in enumerate(trading_dates):
        tra_dates_index[date] = index
        index_tra_dates[index] = date

pad_begin=30
features = np.ones([len(trading_dates)-pad_begin , 7],dtype=float) * -1234

tickers = np.genfromtxt('E:/桌面/RSSR/data/'+market_name+'_final_list.csv', dtype=str, delimiter='\t',skip_header=False)
error_dict = {}
final_list = []
for index, ticker in enumerate(tickers):
    try:
        pad_begin = 29
        features = np.ones([len(trading_dates) - pad_begin, 6], dtype=float) * 0.5
        single_financial = np.genfromtxt(
            os.path.join('E:/桌面/RSSR/data/financial_statements', market_name + '_' + ticker +
                         '_financial_statement.csv'), dtype=str, delimiter=',',
            skip_header=True)
        file_name = os.path.join('E:/桌面/finance_gogogo_NYSE/', market_name + '_' + ticker + '_financial_res.csv')

        if single_financial.ndim == 1 or len(single_financial) < 32:
            continue
        final_list.append(ticker)
        for i in range(3,8):
            _range = max(single_financial[:, i].astype(float)) - min(single_financial[:, i].astype(float))
            single_financial[:, i] = np.round((single_financial[:, i].astype(float) - min(single_financial[:, i].astype(float)))/ _range, 12)
            #mu = np.mean(single_financial[:, i].astype(float), axis=0)
            #sigma = np.std(single_financial[:, i].astype(float), axis=0)
            #single_financial[:, i] = np.round((single_financial[:, i].astype(float) - mu) / sigma, 4)


        for row in range(pad_begin, len(trading_dates)):
            features[row-pad_begin][0] = row-pad_begin
            date = trading_dates[row-pad_begin]
            for financial_row in single_financial:
                if time_interval(date, financial_row[0], financial_row[1]):
                    features[row-pad_begin][1:6] = financial_row[3:8]
                    break


        file_name = os.path.join('E:/桌面/financial/', market_name + '_' + ticker + '_financial_res.csv')

        np.savetxt(file_name, features, delimiter=',', fmt='%.6f')

    except Exception as error:
        print('now, the series is %s, ticket is %s' % (index, ticker))
        print('error:', error)
        error_dict[ticker] = error





final_list = pd.DataFrame(data=final_list)

print(error_dict)


