import numpy as np
import os
import os

import numpy as np

market_name='NYSE'
tickers = np.genfromtxt('E:/桌面/NYSE_final_new.csv', dtype=str, delimiter='\t', skip_header=False)
error_dict = {}
for index, ticker in enumerate(tickers):
    try:
        financial_res = np.genfromtxt(
            os.path.join('E:/桌面/financial/', market_name + '_' + ticker +
                         '_financial_res.csv'), dtype=str, delimiter=',',
            skip_header=False)


        EDO_tickets = np.genfromtxt(
            os.path.join('E:/桌面/RSSR/data/2022-07-05', market_name + '_' + ticker +
                         '_1_2022.csv'), dtype=str, delimiter=',',
            skip_header=False)
        final_tickets_information = np.concatenate((EDO_tickets, financial_res[:,1:]), axis=1).astype(float)


        file_name = os.path.join('E:/桌面/RSSR/data/final_NYSE/', market_name + '_' + ticker + '_final.csv')

        np.savetxt(file_name, final_tickets_information, delimiter=',', fmt='%.6f')
    except Exception as error:
        print('ticket %s has error: %s' % (ticker, error))
        error_dict[ticker] = error
print(error_dict)





