from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from ACC_CALC import Acc_Calculator
import logging
import warnings
# Silent alerts --------------------------------------------
warnings.filterwarnings("ignore")

class Decision_Class :
    def TrainModels (self, X_train, y_train, X_test, y_test, Stage, i) :
        logging.basicConfig(filename="Best_models_ACC.log", level=logging.INFO)
        model_1 = GradientBoostingClassifier(learning_rate=0.05, n_estimators=500, max_depth=30, random_state=1234)
        model_2 = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=1234, n_jobs=-1)
        model_3 = CatBoostClassifier(iterations=5000, depth=16, learning_rate=0.01, loss_function='Logloss', verbose=True)
        model_4 = VotingClassifier(estimators=[
                ('lr', LogisticRegression(n_jobs=-1, random_state=1234)),
                ('rf', RandomForestClassifier(n_jobs=-1, random_state=1234)),
                ('gb', GradientBoostingClassifier(random_state=1234))
                ], voting='hard')
        model_5 = XGBClassifier(learning_rate=0.01, n_estimators=500, max_depth=30, random_state=1234, n_jobs=-1)
        model_6 = AdaBoostClassifier(n_estimators=5000, learning_rate=0.01, algorithm='SAMME.R', random_state=1234)
        model_7 = ExtraTreesClassifier(n_estimators=100, criterion='gini', random_state=1234, n_jobs=-1)
        model_8 = LGBMClassifier()
        model_9 = SVC(kernel='poly', degree=3, C=1, gamma='scale', probability=True, random_state=1234)
        model_10 = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=50, random_state=1234)

        ACC_CALC = Acc_Calculator()
        logging.info("*****************************************************************************")
        logging.info(f"Stage = {Stage}, Data number is : {i}")
        model_1.fit(X_train, y_train)
        predictions = model_1.predict(X_test)
        predictions_proba = model_1.predict_proba(X_test)
        A1 = ACC_CALC.ACC_BY_SKYLEARN(y_test, predictions)
        A2 = ACC_CALC.ACC_BY_THRESHHOLD(y_test, predictions_proba, 0.65)
        A3 = ACC_CALC.ACC_BY_THRESHHOLD_AUTO(y_test, predictions_proba, 20)
        print(f"model_1 : ACC_BY_SKYLEARN = {A1}, ACC_BY_THRESHHOLD 0.65 = {A2}, ACC_BY_THRESHHOLD_AUTO = {A3}")
        logging.info(f"model_1 : ACC_BY_SKYLEARN = {A1}, ACC_BY_THRESHHOLD 0.65 = {A2}, ACC_BY_THRESHHOLD_AUTO = {A3}")

        model_2.fit(X_train, y_train)
        predictions = model_2.predict(X_test)
        predictions_proba = model_2.predict_proba(X_test)
        A1 = ACC_CALC.ACC_BY_SKYLEARN(y_test, predictions)
        A2 = ACC_CALC.ACC_BY_THRESHHOLD(y_test, predictions_proba, 0.65)
        A3 = ACC_CALC.ACC_BY_THRESHHOLD_AUTO(y_test, predictions_proba, 20)
        print(f"model_2 : ACC_BY_SKYLEARN = {A1}, ACC_BY_THRESHHOLD 0.65 = {A2}, ACC_BY_THRESHHOLD_AUTO = {A3}")
        logging.info(f"model_2 : ACC_BY_SKYLEARN = {A1}, ACC_BY_THRESHHOLD 0.65 = {A2}, ACC_BY_THRESHHOLD_AUTO = {A3}")

        model_3.fit(X_train, y_train)
        predictions = model_3.predict(X_test)
        predictions_proba = model_3.predict_proba(X_test)
        A1 = ACC_CALC.ACC_BY_SKYLEARN(y_test, predictions)
        A2 = ACC_CALC.ACC_BY_THRESHHOLD(y_test, predictions_proba, 0.65)
        A3 = ACC_CALC.ACC_BY_THRESHHOLD_AUTO(y_test, predictions_proba, 20)
        print(f"model_3 : ACC_BY_SKYLEARN = {A1}, ACC_BY_THRESHHOLD 0.65 = {A2}, ACC_BY_THRESHHOLD_AUTO = {A3}")
        logging.info(f"model_3 : ACC_BY_SKYLEARN = {A1}, ACC_BY_THRESHHOLD 0.65 = {A2}, ACC_BY_THRESHHOLD_AUTO = {A3}")        

        model_4.fit(X_train, y_train)
        predictions = model_4.predict(X_test)
        predictions_proba = model_4.predict_proba(X_test)
        A1 = ACC_CALC.ACC_BY_SKYLEARN(y_test, predictions)
        A2 = ACC_CALC.ACC_BY_THRESHHOLD(y_test, predictions_proba, 0.65)
        A3 = ACC_CALC.ACC_BY_THRESHHOLD_AUTO(y_test, predictions_proba, 20)
        print(f"model_4 : ACC_BY_SKYLEARN = {A1}, ACC_BY_THRESHHOLD 0.65 = {A2}, ACC_BY_THRESHHOLD_AUTO = {A3}")
        logging.info(f"model_4 : ACC_BY_SKYLEARN = {A1}, ACC_BY_THRESHHOLD 0.65 = {A2}, ACC_BY_THRESHHOLD_AUTO = {A3}")                

        model_5.fit(X_train, y_train)
        predictions = model_5.predict(X_test)
        predictions_proba = model_5.predict_proba(X_test)
        A1 = ACC_CALC.ACC_BY_SKYLEARN(y_test, predictions)
        A2 = ACC_CALC.ACC_BY_THRESHHOLD(y_test, predictions_proba, 0.65)
        A3 = ACC_CALC.ACC_BY_THRESHHOLD_AUTO(y_test, predictions_proba, 20)
        print(f"model_5 : ACC_BY_SKYLEARN = {A1}, ACC_BY_THRESHHOLD 0.65 = {A2}, ACC_BY_THRESHHOLD_AUTO = {A3}")
        logging.info(f"model_5 : ACC_BY_SKYLEARN = {A1}, ACC_BY_THRESHHOLD 0.65 = {A2}, ACC_BY_THRESHHOLD_AUTO = {A3}")                        

        model_6.fit(X_train, y_train)
        predictions = model_6.predict(X_test)
        predictions_proba = model_6.predict_proba(X_test)
        A1 = ACC_CALC.ACC_BY_SKYLEARN(y_test, predictions)
        A2 = ACC_CALC.ACC_BY_THRESHHOLD(y_test, predictions_proba, 0.65)
        A3 = ACC_CALC.ACC_BY_THRESHHOLD_AUTO(y_test, predictions_proba, 20)
        print(f"model_6 : ACC_BY_SKYLEARN = {A1}, ACC_BY_THRESHHOLD 0.65 = {A2}, ACC_BY_THRESHHOLD_AUTO = {A3}")
        logging.info(f"model_6 : ACC_BY_SKYLEARN = {A1}, ACC_BY_THRESHHOLD 0.65 = {A2}, ACC_BY_THRESHHOLD_AUTO = {A3}")                        

        model_7.fit(X_train, y_train)
        predictions = model_7.predict(X_test)
        predictions_proba = model_7.predict_proba(X_test)
        A1 = ACC_CALC.ACC_BY_SKYLEARN(y_test, predictions)
        A2 = ACC_CALC.ACC_BY_THRESHHOLD(y_test, predictions_proba, 0.65)
        A3 = ACC_CALC.ACC_BY_THRESHHOLD_AUTO(y_test, predictions_proba, 20)
        print(f"model_7 : ACC_BY_SKYLEARN = {A1}, ACC_BY_THRESHHOLD 0.65 = {A2}, ACC_BY_THRESHHOLD_AUTO = {A3}")
        logging.info(f"model_7 : ACC_BY_SKYLEARN = {A1}, ACC_BY_THRESHHOLD 0.65 = {A2}, ACC_BY_THRESHHOLD_AUTO = {A3}")                                

        model_8.fit(X_train, y_train)
        predictions = model_8.predict(X_test)
        predictions_proba = model_8.predict_proba(X_test)
        A1 = ACC_CALC.ACC_BY_SKYLEARN(y_test, predictions)
        A2 = ACC_CALC.ACC_BY_THRESHHOLD(y_test, predictions_proba, 0.65)
        A3 = ACC_CALC.ACC_BY_THRESHHOLD_AUTO(y_test, predictions_proba, 20)
        print(f"model_8 : ACC_BY_SKYLEARN = {A1}, ACC_BY_THRESHHOLD 0.65 = {A2}, ACC_BY_THRESHHOLD_AUTO = {A3}")
        logging.info(f"model_8 : ACC_BY_SKYLEARN = {A1}, ACC_BY_THRESHHOLD 0.65 = {A2}, ACC_BY_THRESHHOLD_AUTO = {A3}")                                        

        model_9.fit(X_train, y_train)
        predictions = model_9.predict(X_test)
        predictions_proba = model_9.predict_proba(X_test)
        A1 = ACC_CALC.ACC_BY_SKYLEARN(y_test, predictions)
        A2 = ACC_CALC.ACC_BY_THRESHHOLD(y_test, predictions_proba, 0.65)
        A3 = ACC_CALC.ACC_BY_THRESHHOLD_AUTO(y_test, predictions_proba, 20)
        print(f"model_9 : ACC_BY_SKYLEARN = {A1}, ACC_BY_THRESHHOLD 0.65 = {A2}, ACC_BY_THRESHHOLD_AUTO = {A3}")
        logging.info(f"model_9 : ACC_BY_SKYLEARN = {A1}, ACC_BY_THRESHHOLD 0.65 = {A2}, ACC_BY_THRESHHOLD_AUTO = {A3}")                                                


        model_10.fit(X_train, y_train)
        predictions = model_10.predict(X_test)
        predictions_proba = model_10.predict_proba(X_test)
        A1 = ACC_CALC.ACC_BY_SKYLEARN(y_test, predictions)
        A2 = ACC_CALC.ACC_BY_THRESHHOLD(y_test, predictions_proba, 0.65)
        A3 = ACC_CALC.ACC_BY_THRESHHOLD_AUTO(y_test, predictions_proba, 20)
        print(f"model_10 : ACC_BY_SKYLEARN = {A1}, ACC_BY_THRESHHOLD 0.65 = {A2}, ACC_BY_THRESHHOLD_AUTO = {A3}")
        logging.info(f"model_10 : ACC_BY_SKYLEARN = {A1}, ACC_BY_THRESHHOLD 0.65 = {A2}, ACC_BY_THRESHHOLD_AUTO = {A3}")                                                
