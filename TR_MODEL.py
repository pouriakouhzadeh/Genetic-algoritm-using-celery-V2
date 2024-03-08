from timeconvert import TimeConvert
from selecttimetodelete import SelectTimeToDeleteOptimized
from preparing_data import PREPARE_DATA
from PAGECREATOR import PageCreatorParallel
from deleterow import DeleteRow
from FEATURESELECTION import FeatureSelection
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

def normalize_data(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled


def ACC_BY_THRESHHOLD(self, y_test, predictions_proba, TH):
    predictions_proba = pd.DataFrame(predictions_proba)
    predictions_proba.reset_index(inplace = True ,drop =True)
    y_test.reset_index(inplace = True ,drop =True)
    TH = TH / 100
    try :
        wins = 0
        loses = 0
        for i in range(len(y_test)) :
            if predictions_proba[1][i] > TH :
                if y_test['close'][i] == 1 :
                    wins = wins + 1
                else :
                    loses = loses + 1    
            if predictions_proba[0][i] > TH :
                if y_test['close'][i] == 0 :
                    wins = wins + 1
                else :
                    loses = loses + 1       
        # logging.info(f"Thereshhold wins = {wins}, Thereshhold loses = {loses}")
        return ( (wins * 100) / (wins + loses) , wins, loses)  
    except :
        return 0, 0, 0


class TrainModels:
    def Train(self, data, depth, page, feature, QTY, iter, Thereshhold, primit_hours=[]):
        print(f"depth:{depth}, page:{page}, features:{feature}, QTY:{QTY}, iter:{iter}, Thereshhold:{Thereshhold}, primit_hours:{primit_hours}")
        data = data[-QTY:]
        data = TimeConvert().exec(data)        
        data.reset_index(inplace=True, drop=True)
        primit_hours = SelectTimeToDeleteOptimized().exec(data, primit_hours)
        data, target, primit_hours = PREPARE_DATA().ready(data, primit_hours)
        data, target = PageCreatorParallel().create_dataset(data, target, page)
        primit_hours = primit_hours[page:]
        data, target = DeleteRow().exec(data, target, primit_hours)
        fs = FeatureSelection()
        selected_data, selected_features = fs.select(data, target, feature)
        data = selected_data.copy()
        data = normalize_data(data)
        X_train, X_temp, y_train, y_temp = train_test_split(data, target, test_size=0.2, random_state=1234)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1234)

        # print(f"primit_hours  = {primit_hours}")
        model = CatBoostClassifier(
            iterations=iter, # شروع با تعداد زیادی تکرار
            learning_rate=0.01,
            depth=depth,
            loss_function='Logloss',
            eval_metric='AUC', # معیار ارزیابی بسته به مسئله شما
            use_best_model=True, # استفاده از بهترین مدل بر روی داده‌های اعتبارسنجی
            verbose=50,
            task_type='CPU' # یا 'GPU' اگر سخت‌افزار مربوطه موجود است
        )
        train_pool = Pool(X_train, y_train)
        validation_pool = Pool(X_val, y_val)
        print("Start training model")
        model.fit(
            train_pool,
            eval_set=validation_pool,
            early_stopping_rounds=50 # توقف زودهنگام اگر بعد از 50 تکرار بهبودی در AUC اعتبارسنجی دیده نشود
        )
        print("End of training model")
        predictions_proba = model.predict_proba(X_test)
        ACC , wins , loses = ACC_BY_THRESHHOLD(self, y_test, predictions_proba, Thereshhold)
        print(f"ACC:{ACC}, wins:{wins}, loses:{loses}")
        return ACC , wins , loses

