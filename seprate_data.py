import pandas as pd

class SEPRATE_DATA:
    def start(self, data, target, labels, n):
        data = pd.DataFrame(data)
        target = pd.DataFrame(target)
        data['target'] = target
        result_data = []
        result_target = []

        for i in range(n):
            variable_name = f"data_{i}"
            globals()[variable_name] = data[labels == i]
            
            variable_name_target = f"target_{i}"
            globals()[variable_name_target] = globals()[variable_name]['target']

            globals()[variable_name].drop(columns='target', inplace=True)
            variable_name_target = pd.DataFrame(globals()[variable_name_target])

            globals()[variable_name].reset_index(inplace=True, drop=True)
            variable_name_target.reset_index(inplace=True, drop=True)

            result_data.append(globals()[variable_name])
            result_target.append(variable_name_target)

        return result_data, result_target

# Example usage:
# seprate_data_instance = SEPRATE_DATA()
# result_data, result_target = seprate_data_instance.start(data, target, labels, n)
