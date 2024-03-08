from celery_app import app
from TR_MODEL import TrainModels
import pandas as pd
import json


@app.task()
def train_model_task(currency_data, depth, page, feature, QTY, iter, Thereshhold, primit_hours):
    currency_data_df = pd.read_json(currency_data, orient='split')

    # ساخت DataFrame از دیکشنری یا لیست دیکشنری‌ها
    currency_data_df = pd.DataFrame(currency_data_df)


    print("New job recived start to proccessing ...............................................................")
    acc, wins, loses = TrainModels().Train(currency_data_df, depth, page, feature, QTY, iter, Thereshhold, primit_hours)
    return acc, wins, loses
