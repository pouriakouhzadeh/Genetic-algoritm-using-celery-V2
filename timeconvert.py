import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import datetime

class TimeConvert:
    
    def __init__(self, num_workers=None):
        self.num_workers = num_workers if num_workers is not None else 8  # تعداد ورکرها را تعیین کنید یا به صورت پیش‌فرض 4 قرار دهید
    
    def _convert_time(self, row):
        temp = pd.to_datetime(row["time"])
        return temp.hour
    
    def exec(self, data):
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # اجرای پارالل _convert_time برای هر ردیف از داده‌ها
            hours = list(executor.map(self._convert_time, [row for _, row in data.iterrows()]))
        
        # اضافه کردن ساعت به عنوان یک ستون جدید به داده‌ها
        data["Hour"] = hours
        data.drop(columns='time', inplace=True)
        
        return data
