import socket
import threading

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv(r"C:\Users\prash\Desktop\data\CROP RECOMMENDATION PROJECT\csv files/Crop_recommendation.csv")

#print(df.head())

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


data = np.array([[32,66,17,34.94,65.26,7.16,70.14]])
prediction = XGB.predict(data)
#print(list(le.inverse_transform(prediction)))

data = np.array([[90,42,43,20.87,82.00,6.50,202.93]])
#data = sc.transform(data)
prediction = XGB.predict(data)
pred = str(le.inverse_transform(prediction))


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print("Socket Connected")
#host ="192.168.43.132"
port = 9999
s.bind((socket.gethostname(), port))

s.listen(3)
print("Waiting for connections")

def work(c, addr):
    print("Connection has been established |" + " IP " + addr[0] + " | Port " + str(addr[1]))

    connected = True
    while connected:
        data = c.recv(2024).decode('utf-8')
        if data == 'quit':
            connected = False
        #print(data)
        if len(data)>0:
            data = data.split()
            lst = []
            for i in data:
                para = float(i)
                lst.append(para)
            lst = [lst]
            data = np.array(lst)
            # data = sc.transform(data)
            prediction = XGB.predict(data)
            pred = str(le.inverse_transform(prediction))
            c.send(bytes(pred, 'utf-8'))
        else:
            break

    c.close()

while True:
    c, addr = s.accept()
    thread = threading.Thread(target= work, args=(c, addr))
    thread.daemon = True
    thread.start()
    print()
    print(f"Active Connections {threading.active_count() - 1}")

