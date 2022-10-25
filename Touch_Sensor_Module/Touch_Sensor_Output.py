# Importing Libraries
import serial
import time
arduino = serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout=.1)

while True:
    ser_bytes = arduino.readline().decode("utf-8")
    
    print(ser_bytes)

