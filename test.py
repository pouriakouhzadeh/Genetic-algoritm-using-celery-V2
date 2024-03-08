from ACC_CALC import Acc_Calculator
import pandas as pd
import numpy as np
import random

predictions_1_proba = [random.random() for _ in range(1000)]
y_test1 = [random.randint(0, 1) for _ in range(1000)]

predictions_1_proba = pd.DataFrame(predictions_1_proba)
y_test1 = pd.DataFrame(y_test1)

predictions_1_proba[1] = predictions_1_proba[0].shift(+1) 

predictions_1_proba = predictions_1_proba[1:]
y_test1 = y_test1[1:]

ACC_CALC = Acc_Calculator()

A1 = ACC_CALC.ACC_BY_SKYLEARN(y_test1, y_test1)
A2 = ACC_CALC.ACC_BY_THRESHHOLD(y_test1, predictions_1_proba, 0.6)
A3 = ACC_CALC.ACC_BY_THRESHHOLD_AUTO(y_test1, predictions_1_proba, 20)

print(A1, A2, A3)