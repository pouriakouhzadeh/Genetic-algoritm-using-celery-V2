from timeconvert import TimeConvert
from selecttimetodelete import SelectTimeToDelete
from preparing_data import PREPARE_DATA
from PAGECREATOR import PageCreator
from deleterow import DeleteRow
from FEATURESELECTION import FeatureSelection
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
import pickle

def normalize_data(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled

class DesisionClass:
    def Desision(self, data, model, depth = 2 ,page = 2 ,feature = 30 ,QTY = 1000 , Thereshhold=65):
        data = TimeConvert().exec(data)
        data = data[-QTY:]
        data.reset_index(inplace = True, drop = True)
        Forbidden_list = SelectTimeToDelete().exec(data, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,0])
        data, target, Forbidden_list = PREPARE_DATA().ready(data, Forbidden_list)
        data, target = PageCreator().create_dataset(data, target, page)
        Forbidden_list = Forbidden_list[page:]
        data, target = DeleteRow().exec(data, target,Forbidden_list)
        data = FeatureSelection().select(data, target, feature).copy()
        data = normalize_data(data)
        position = "NAN"
        Answer = model.predict_proba(data)
        Answer = Answer[-1:]

        if Answer[0, 0] > (Thereshhold/100) :
            position = "SELL"
        if Answer[0, 1] > (Thereshhold/100) :
            position = "BUY"
            
        return position, Answer
