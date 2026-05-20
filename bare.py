import serial
import struct
import time

# ser = serial.Serial('/dev/ttyAMA0', 9600, timeout=1)

port = '/dev/ttyAMA0' 
baud = 115200
val = 0

try:
    ser = serial.Serial(port, baud, timeout=1)
    print(f"Opened {port} at {baud}")
except Exception as e:
    print(f"Error opening port: {e}")
    exit()

def send_float(value):
    # 'S' header + float
    packet = struct.pack('<cf', b'S', value)
    ser.write(packet)
    # Explicitly flush to ensure it leaves the Pi buffer immediately
    ser.flush() 

try:
    
    while True:
        val = val+0.01
        send_float(val)
        print(f"Sent: {val}")
        time.sleep(0.15) 
except KeyboardInterrupt:
    ser.close()