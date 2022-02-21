import paho.mqtt.client as mqtt
from random import randrange, uniform
import time


mqttBroker = "test.mosquitto.org" #mqtt.eclipseprojects.io" #"broker.hivemq.com"
client = mqtt.Client("client")
client.connect(mqttBroker)



name = input("Enter the Parameters: ")
client.publish("aks/mm",name)
print("Parameters are sent")
#time.sleep(30)

def on_message(client, userdata, message):
    print("Received Message: ",str(message.payload.decode("utf-8")))

client.loop_start()
client.subscribe("me/nn")
client.on_message = on_message
time.sleep(20)
client.loop_stop()