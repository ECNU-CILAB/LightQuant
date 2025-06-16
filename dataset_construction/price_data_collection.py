import pandas as pd
import baostock as bs
from datetime import datetime, timedelta


class A_Stocks_DataCollection:
    def __init__(self, start_date, end_date, current_date, stocks_num=10000):
        self.start_date = start_date
        self.end_date = end_date
        self.current_date = current_date
        self.stocks_num = stocks_num


    def get_sz50_stocks(self):

        all_stocks = self.get_stocks_code_name()
        sz50_stocks = pd.read_csv('../llm_factor/CSMD50.csv', encoding="utf-8")
        result = pd.merge(all_stocks, sz50_stocks[['code']], on='code', how='inner')
        return result

    def get_data(self):
        result = self.get_sz50_stocks()
        result = self.get_stock_code_industry(result)
        self.get_history_k_data(self.start_date, self.end_date, result)

    # def get_data(self):
    #     result = self.get_stocks_code_name()
    #     result = self.get_stock_code_industry(result)
    #     self.get_history_k_data(self.start_date, self.end_date, result)

    def get_new_data(self):
        result = self.get_stocks_code_name()
        result = self.get_stock_code_industry(result)
        self.get_history_k_data(self.current_date, self.current_date, result)

    def get_stocks_code_name(self):

        lg = bs.login()

        print('login respond error_code:' + lg.error_code)
        print('login respond  error_msg:' + lg.error_msg)


        rs = bs.query_all_stock(day=self.previous_date())
        print('query_all_stock respond error_code:' + rs.error_code)
        print('query_all_stock respond  error_msg:' + rs.error_msg)


        data_list = []
        while (rs.error_code == '0') and rs.next():

            data_list.append(rs.get_row_data())
        result = pd.DataFrame(data_list, columns=rs.fields)
        print(result)

        return result

    def get_stock_code_industry(self, result):

        lg = bs.login()

        print('login respond error_code:' + lg.error_code)
        print('login respond  error_msg:' + lg.error_msg)


        rs = bs.query_stock_industry()
        print('query_stock_industry error_code:' + rs.error_code)
        print('query_stock_industry respond  error_msg:' + rs.error_msg)


        industry_list = []
        while (rs.error_code == '0') & rs.next():

            industry_list.append(rs.get_row_data())
        industry_result = pd.DataFrame(industry_list, columns=rs.fields)


        result = pd.merge(result, industry_result[['code', 'industry', 'industryClassification']], on='code', how='left')


        bs.logout()

        return result

    def get_history_k_data(self, start_date, end_date, result):

        lg = bs.login()

        print('login respond error_code:' + lg.error_code)
        print('login respond  error_msg:' + lg.error_msg)


        for code in result["code"]:
            rs = bs.query_history_k_data_plus(code=code,
                                              fields="date,code,open,high,low,close,preclose,volume,"
                                                     "amount,adjustflag,turn,tradestatus,pctChg,peTTM,"
                                                     "pbMRQ,psTTM,pcfNcfTTM,isST",
                                              start_date=start_date,
                                              end_date=end_date,
                                              frequency="d",
                                              adjustflag="3")  # frequency="d"取日k线，adjustflag="3"默认不复权
            k_data_list = []
            while (rs.error_code == '0') and rs.next():

                k_data_list.append(rs.get_row_data())
            k_result = pd.DataFrame(k_data_list, columns=rs.fields)
            k_result = pd.merge(k_result, result[['code', 'code_name','industry', 'industryClassification']], on='code',
                              how='left')
            if start_date==end_date:
                k_result.to_csv(f"../data/A_Stocks_Data/{k_result['code'][0]}.csv",encoding="utf-8-sig", index=False, mode="a",header=0)
            else:
                k_result.to_csv(f"../data/A_Stocks_Data/{k_result['code'][0]}.csv",encoding="utf-8-sig", index=False)

            

        print('query_history_k_data_plus respond error_code:' + rs.error_code)
        print('query_history_k_data_plus respond  error_msg:' + rs.error_msg)

        bs.logout()

    def previous_date(self):
        date_format = '%Y-%m-%d'
        date_obj = datetime.strptime(self.current_date, date_format)
        previous_date = date_obj - timedelta(days=1)
        previous_date_str = previous_date.strftime(date_format)
        return previous_date_str


if __name__ == "__main__":

    start_date = "2019-01-01"
    end_date = "2024-12-18"
    current_date = "2024-12-19"


    data_collector = A_Stocks_DataCollection(start_date, end_date, current_date)


    data_collector.get_data()


    # data_collector.get_new_data()

