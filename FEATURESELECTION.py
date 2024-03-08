from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import numpy as np
import warnings
import pandas as pd

# Silent alerts --------------------------------------------
warnings.filterwarnings("ignore")

# Making objects -------------------------------------------
class FeatureSelection:

    def select(self, data, target, n, **rf_params):
        n = int(n)
        data = pd.DataFrame(data)  # این خط باید قبل از استفاده از `SelectFromModel` قرار گیرد.
        model = RandomForestClassifier(**rf_params, n_jobs=-1)
        sfm = SelectFromModel(estimator=model, threshold=-np.inf, max_features=n)
        sfm.fit(data, target)
        X_selected = sfm.transform(data)
        
        # Getting the names of the selected features
        feature_names = np.array(data.columns)
        selected_features = feature_names[sfm.get_support()]

        return pd.DataFrame(X_selected, columns=selected_features), selected_features

# Example usage
# Assuming `data` is a DataFrame and `target` is the target column
# fs = FeatureSelection()
# X_selected, selected_features = fs.select(data, target, n=5, n_estimators=100, max_depth=5)
# print("Selected features:", selected_features)
