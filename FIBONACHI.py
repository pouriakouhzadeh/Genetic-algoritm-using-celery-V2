import pandas as pd

class FibonacciRetracement:
    def __init__(self, df):
        self.df = df

    def fibonacci_retracement_values(self, close_data, levels):
        max_val = max(close_data)
        min_val = min(close_data)
        diff = max_val - min_val

        retracement_values = []
        for level in levels:
            retracement_val = max_val - (level * diff)
            retracement_values.append(retracement_val)

        return retracement_values

    def calculate_fibonacci_retracement(self):
        levels = [0, 0.236, 0.382, 0.5, 0.618, 1]
        for index, row in self.df.iterrows():
            close_data = [row['close']]
            fibonacci_values = self.fibonacci_retracement_values(close_data, levels)
            for i, val in enumerate(fibonacci_values):
                self.df.at[index, f'fibonacci_{i+1}'] = val


