import serial
import time

port = '/dev/serial0'
baud = 115200
try:
    ser = serial.Serial(port, baud, timeout=1)
    print(f"Opened {port} at {baud} baud.")
    
    test_msg = b"CRITICAL_TEST"
    ser.write(test_msg)
    time.sleep(0.1)
    
    response = ser.read(ser.in_waiting)
    
    if response == test_msg:
        print("SUCCESS: Passed")
    else:
        print(f"FAILED: Received {response}")
    ser.close()
except Exception as e:
    print(f"Error: {e}")