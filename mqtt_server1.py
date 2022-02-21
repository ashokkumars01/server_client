import paho.mqtt.client
import paho.mqtt.client as mqtt
import time
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import config
import warnings
warnings.filterwarnings("ignore")

model = pickle.load(open(r"C:\Users\prash\Desktop\data\CROP RECOMMENDATION PROJECT\pickle file\XGBoost.pkl", 'rb'))
le = LabelEncoder()



def on_message(client, userdata, message):
    #global mes
    print("Received Message: ",str(message.payload.decode("utf-8")))
    config.mes = str(message.payload.decode("utf-8"))
    config.mes = config.mes.split()

mqttBroker = "test.mosquitto.org" #"mqtt.eclipseprojects.io"
client = mqtt.Client("Server")
client.connect(mqttBroker)


client.loop_start()
client.subscribe("aks")
#client.publish("Message","Kumar")
client.on_message = on_message
time.sleep(30)
client.loop_stop()

print(config.mes)
#mes = mes.split()
lst = []
for i in config.mes:
    para = float(i)
    lst.append(para)
lst = [lst]
data = np.array(lst)
prediction = model.predict(data)
pred = str(le.inverse_transform(prediction))

client.publish("me",pred)

time.sleep(60)