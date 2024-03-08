import numpy as np
import pandas as pd
import ta
import pickle
import hashlib
from tqdm import tqdm
import concurrent.futures
import threading
from FIBONACHI import FibonacciRetracement
# from sklearn.preprocessing import MinMaxScaler
class PREPARE_DATA :
    def calculate_price_change_features(self, data):
        data['Price_Change'] = data['close'].diff()
        data['Price_Change_Percent'] = data['Price_Change'] / data['close'].shift(1) * 100
        data['Price_Rise'] = data['Price_Change'].apply(lambda x: 1 if x > 0 else 0)
        data['Price_Fall'] = data['Price_Change'].apply(lambda x: 1 if x < 0 else 0)
    
        return data

    def calculate_percentage_change(self, data):
        data['Percentage_Change'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1) * 100
        return data
    
    def calculate_volume_features(self, data):
        data['Volume_Rate_of_Change'] = (data['volume'] - data['volume'].shift(1)) / data['volume'].shift(1) * 100
        data['Average_Volume'] = data['volume'].rolling(window=20).mean()
        data['Volume_to_Average_Volume_Ratio'] = data['volume'] / data['Average_Volume']
        data['Total_Volume'] = data['volume'].rolling(window=20).sum()
        data['Previous_Volume'] = data['volume'].shift(1)
        return data


    def calculate_volume_features(self, data):
        # محاسبه فیچرهای مرتبط با حجم معاملات
        data['Volume_Rate_of_Change'] = (data['volume'] - data['volume'].shift(1)) / data['volume'].shift(1) * 100
        data['Average_Volume'] = data['volume'].rolling(window=20).mean()
        data['Volume_to_Average_Volume_Ratio'] = data['volume'] / data['Average_Volume']
        data['Total_Volume'] = data['volume'].rolling(window=20).sum()
        data['Previous_Volume'] = data['volume'].shift(1)

        return data

    def calculate_atr(self, data, window=14):
        tr = pd.DataFrame()
        tr['high-low'] = data['high'] - data['low']
        tr['high-close_prev'] = abs(data['high'] - data['close'].shift(1))
        tr['low-close_prev'] = abs(data['low'] - data['close'].shift(1))
        tr['true_range'] = tr[['high-low', 'high-close_prev', 'low-close_prev']].max(axis=1)
        atr = tr['true_range'].rolling(window=window).mean()
        return atr
    
    def moving_average(self, data, window_size):
        moving_averages = data.rolling(window=window_size).mean()
        moving_averages[:window_size-1] = 0
        return moving_averages

    def detect_candle_8(self, row):
        open_val = row['open']
        high_val = row['high']
        low_val = row['low']
        close_val = row['close']

        if open_val == close_val:  # شمع معمولی
            return 0
        elif open_val > close_val:  # شمع نزولی
            if high_val == open_val and low_val == close_val:  # شمع نزولی پایینی (bearish engulfing)
                return 1
            elif high_val == open_val:  # شمع نزولی با بالای سایه بالا
                return 2
            elif low_val == close_val:  # شمع نزولی با پایینی سایه پایین
                return 3
            else:
                return 4
        else:  # شمع صعودی
            if high_val == close_val and low_val == open_val:  # شمع صعودی پایینی (bullish engulfing)
                return 5
            elif high_val == close_val:  # شمع صعودی با بالای سایه بالا
                return 6
            elif low_val == open_val:  # شمع صعودی با پایینی سایه پایین
                return 7
            else:
                return 8


    def detect_candle_20(self, row):
        open_val = row['open']
        high_val = row['high']
        low_val = row['low']
        close_val = row['close']

        if open_val == close_val:  # شمع معمولی
            return 0
        elif open_val > close_val:  # شمع نزولی
            if high_val == open_val and low_val == close_val:  # شمع نزولی پایینی (bearish engulfing)
                return 1
            elif high_val == open_val:  # شمع نزولی با بالای سایه بالا
                return 2
            elif low_val == close_val:  # شمع نزولی با پایینی سایه پایین
                return 3
            else:
                return 4
        else:  # شمع صعودی
            if high_val == close_val and low_val == open_val:  # شمع صعودی پایینی (bullish engulfing)
                return 5
            elif high_val == close_val:  # شمع صعودی با بالای سایه بالا
                return 6
            elif low_val == open_val:  # شمع صعودی با پایینی سایه پایین
                return 7
            else:
                return 8

    def ready(self, data, Forbidden_list):

        data = self.calculate_percentage_change(data)
        data = self.calculate_volume_features(data)
        data = self.calculate_price_change_features(data)
        data = self.calculate_volume_features(data)
        data['ATR'] = self.calculate_atr(data, 10)
        data['ATR1'] = self.calculate_atr(data, 14)
        data['ATR2'] = self.calculate_atr(data, 21)
        data['ATR3'] = self.calculate_atr(data, 50)

        fibonacci_retracement = FibonacciRetracement(pd.DataFrame(data['close']))
        fibonacci_retracement.calculate_fibonacci_retracement()
        data['fibo1'] = fibonacci_retracement.df['fibonacci_1']
        data['fibo2'] = fibonacci_retracement.df['fibonacci_2']
        data['fibo3'] = fibonacci_retracement.df['fibonacci_3']
        data['fibo4'] = fibonacci_retracement.df['fibonacci_4']
        data['fibo5'] = fibonacci_retracement.df['fibonacci_5']
        data['fibo6'] = fibonacci_retracement.df['fibonacci_6']

        data['candle_types_8'] = data.apply(self.detect_candle_8, axis=1)
        data['candle_types_20'] = data.apply(self.detect_candle_20, axis=1)

        data['AveragePrice'] = (data['open'] + data['high'] + data['low'] + data['close']) / 4

        data['PriceRangeRatio'] = (data['high'] - data['low']) / data['close']

        data['RSI'] = ta.momentum.rsi(data['close'], fillna=True)

        data['RSI-volume'] = ta.momentum.rsi(data['volume'], fillna=True)

        data['Percentage_Change'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1) * 100

        data['ROC'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1) * 100

        data['ReturnDifference'] = data['close'].diff()

        data['TradingVolumeRatio'] = data['volume'] / data['volume'].rolling(window=10).mean()

        data['PriceToMovingAverageRatio'] = data['close'] / data['close'].rolling(window=20).mean()

        data['candle_height'] = data['high'] - data['low']
        data['candle_body'] = data['close'] - data['open']
        data['upper_shadow'] = data['high'] - data[['open', 'close']].max(axis=1)
        data['lower_shadow'] = data[['open', 'close']].min(axis=1) - data['low']

        for i in range (3, 150):
            data[f'ma{i}'] = self.moving_average(data['close'], i)

        for i in range (3, 150):
            data[f'ma-volume{i}'] = self.moving_average(data['volume'], i)


        data = data[151:]
        Forbidden_list = Forbidden_list[151:]

        # print("start to calculate indicators")
        indicators = ta.add_all_ta_features(data.copy(), "open", "high", "low", "close", "volume", fillna=True)
        indicators.drop(columns=["open", "high", "low", "close", "volume", "RSI", "AveragePrice", "Hour"], inplace=True)
        data_with_indicators = pd.concat([data, indicators], axis=1)
        # print("calculate indicators finished")

        data = data_with_indicators



        target = data['close']
        target = ((target - target.shift(-1) )> 0 ).map({True: 1, False: 0}).fillna(0)
        Hour = data["Hour"]
        data = data - data.shift(+1)
        
        data = data[1:]
        Forbidden_list = Forbidden_list[1:]
        Hour = Hour[1:]
        target = target[1:]

        # data = data[:-1]
        # Forbidden_list = Forbidden_list[:-1]
        # Hour = Hour[:-1]
        # target = target[:-1]

        data["Hour"] = Hour
        data.reset_index(inplace=True,drop=True)
        target.reset_index(inplace=True, drop=True)
        Forbidden_list.reset_index(inplace = True, drop =True )
        return data, target, Forbidden_list





