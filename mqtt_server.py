import paho.mqtt.client as mqtt
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
import pickle
import config
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv(r"C:\Users\prash\Desktop\data\CROP RECOMMENDATION PROJECT\csv files/Crop_recommendation.csv")


X = df.drop(['label'],axis=1)  # Independent Variable
Y = df['label']  # Dependent Variable / Target variable

le = LabelEncoder()
Y = le.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20,random_state=0)

XGB = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.5, gamma=0.3, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.25, max_delta_step=0, max_depth=4,
              min_child_weight=7, missing=1, monotone_constraints='()',
              n_estimators=100, n_jobs=8, num_parallel_tree=1,
              objective='multi:softprob', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=None, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)

XGB.fit(X_train, Y_train)

prediction = XGB.predict(X_test)

while True:
    import config
    def on_message(client, userdata, message):
        #import config
        print("Received Message: ",str(message.payload.decode("utf-8")))
        config.mes = str(message.payload.decode("utf-8"))
        config.mes = config.mes.split()


    mqttBroker = "test.mosquitto.org" #"mqtt.eclipseprojects.io" #"broker.hivemq.com"
    client = mqtt.Client("Server")
    client.connect(mqttBroker)


    client.loop_start()
    client.subscribe("aks/mm")
    client.on_message = on_message
    time.sleep(20)
    client.loop_stop()


    #print(config.mes)
    lst = []
    for i in config.mes:
        para = float(i)
        lst.append(para)
    lst = [lst]
    data = np.array(lst)
    prediction = XGB.predict(data)
    pred = str(le.inverse_transform(prediction))

    client.publish("me/nn",pred)

time.sleep(60)