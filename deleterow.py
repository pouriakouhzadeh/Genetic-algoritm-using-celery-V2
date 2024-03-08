import pandas as pd
from tqdm import tqdm

class DeleteRow:
    
    def exec(self, data, target, primit_hours):
        data = pd.DataFrame(data)
        target = pd.DataFrame(target)
        data.reset_index(inplace = True, drop = True)
        target.reset_index(inplace = True, drop = True)
        data['primit_hours'] = primit_hours
        target['primit_hours'] = primit_hours
        data.dropna(inplace=True)
        target.dropna(inplace=True)
        data.drop(columns = 'primit_hours', inplace = True)
        target.drop(columns = 'primit_hours', inplace = True)
        data.reset_index(inplace= True, drop = True)
        target.reset_index(inplace= True, drop = True)

        return data, target