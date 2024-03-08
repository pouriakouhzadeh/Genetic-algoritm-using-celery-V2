import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class PageCreatorParallel:
    def _create_page(self, args):
        """یک تکه از دیتاست را بر اساس شاخص‌های داده‌شده ایجاد می‌کند."""
        dataset, start_idx, time_step = args
        return dataset[(start_idx - time_step + 1) : (start_idx + 1)].reshape(-1)

    def create_dataset(self, dataset, target, time_step, max_workers=8):
        """دیتاست و هدف را با استفاده از اجرای پارالل ایجاد می‌کند."""
        time_step = int(time_step)
        
        # اطمینان از اینکه dataset یک numpy array است
        if not isinstance(dataset, np.ndarray):
            dataset = np.array(dataset)
        
        # ایجاد ارگومان‌ها برای هر صفحه
        args = [(dataset, i, time_step) for i in range(time_step, len(dataset))]
        
        data = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # اجرای پارالل با پیشرفت نمایش داده شده توسط tqdm
            for page in tqdm(executor.map(self._create_page, args), total=len(args)):
                data.append(page)
                
        data = np.array(data)
        target = target[time_step:]
        return data, target
