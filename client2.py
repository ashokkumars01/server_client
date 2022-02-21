import socket

c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = "192.168.43.132"
port = 9999
c.connect((host, port))

while True:
    param = input("Enter parameters: ")
    if param == 'quit':
        break
    else:
        c.send(bytes(param,'utf-8'))

        print(c.recv(2024).decode('utf-8'))

c.close()

# 32 66 17 34.94 65.26 7.16 70.14
# 90 42 43 20.87 82.00 6.50 202.93