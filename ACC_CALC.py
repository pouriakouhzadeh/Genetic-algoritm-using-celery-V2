from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import logging

class Acc_Calculator:
    logging.basicConfig(filename="Best_models_ACC.log", level=logging.INFO)
    def ACC_BY_SKYLEARN(self, y_test, predictions):
        predictions = pd.DataFrame(predictions)
        predictions.reset_index(inplace = True ,drop =True)
        y_test.reset_index(inplace = True ,drop =True)
        try :
            return accuracy_score(y_test, predictions)
        except :
            return 0 
        
    def ACC_BY_THRESHHOLD(self, y_test, predictions_proba, TH):
        predictions_proba = pd.DataFrame(predictions_proba)
        predictions_proba.reset_index(inplace = True ,drop =True)
        y_test.reset_index(inplace = True ,drop =True)
        try :
            wins = 0
            loses = 0
            for i in range(len(y_test)) :
                if predictions_proba[1][i] > TH :
                    if y_test.loc[i] == 1 :
                        wins = wins + 1
                    else :
                        loses = loses + 1    
                if predictions_proba[0][i] > TH :
                    if y_test.loc[i] == 0 :
                        wins = wins + 1
                    else :
                        loses = loses + 1       
            logging.info(f"Thereshhold wins = {wins}, Thereshhold loses = {loses}")
            return ( (wins * 100) / (wins + loses) , wins, loses)  
        except :
            return 0, 0, 0

    
    def ACC_BY_THRESHHOLD_AUTO(self, y_test, predictions_proba, Percent) :
        predictions_proba = pd.DataFrame(predictions_proba)
        predictions_proba.reset_index(inplace = True ,drop =True)
        y_test.reset_index(inplace = True ,drop =True)     
        try :
            max_acc = 0
            for TH in np.arange(0.5, 0.9, 0.01):
                wins = 0
                loses = 0
                for i in range(len(y_test)) :
                    if predictions_proba[1][i] > TH :
                        if y_test.loc[i] == 1 :
                            wins = wins + 1
                        else :
                            loses = loses + 1    
                    if predictions_proba[0][i] > TH :
                        if y_test.loc[i] == 0 :
                            wins = wins + 1
                        else :
                            loses = loses + 1      

                if ( ( (wins * 100) / (wins + loses) ) > max_acc )  and   ( (wins + loses) > ( (len(y_test) * Percent) / 100 ) ):
                    max_acc = (wins * 100) / (wins + loses) 

            logging.info(f"Auto wins = {wins}, Auto loses = {loses}")
            return max_acc        
        
        except :
            return 0