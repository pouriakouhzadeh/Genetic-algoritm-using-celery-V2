import pandas as pd
import numpy as np


class CLEAN_DATA :
        def remove_outliers_iqr(self, data):
                # محاسبه Q1 و Q3
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)

            # محاسبه IQR
            IQR = Q3 - Q1

            # تعیین حد پایین و حد بالای IQR
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # حذف داده‌های پرت
            cleaned_data = data[(data >= lower_bound) & (data <= upper_bound)]
            return cleaned_data
        
        def clear(self, data):
                # data.drop(columns = 'time', inplace = True)
                data.dropna(inplace=True)
                # data['close'] = self.remove_outliers_iqr(data['close'])
                # data['open'] = self.remove_outliers_iqr(data['open'])
                # data['high'] = self.remove_outliers_iqr(data['high'])
                # data['low'] = self.remove_outliers_iqr(data['low'])
                # data['volume'] = self.remove_outliers_iqr(data['volume'])
                # data.dropna(inplace=True)
                return data