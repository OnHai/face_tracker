import time
import sys
import os
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crtp.serialdriver import SerialDriver

URI = 'serial://ttyAMA0'
PAR = 'uhel_group.custom_val'
# CHANGE IN LIBRARY:
# /home/admin/face_tracker/venv/lib/python3.11/site-packages/cflib/crtp/serialdriver.py
# self.cpx = CPX(UARTTransport(device, 9600))
def console_callback(text: str):
    print(text, end='')

def run_counter(scf):
    cf = scf.cf
    counter = 0
    try:
        while True:
            cf.param.set_value(PAR, counter)
            print(f"Set {PAR}: {counter}")
            time.sleep(0.1)
            print(cf.param.get_value(PAR))
            counter += 1
            time.sleep(4.9)
    except KeyboardInterrupt:
        print("\nFlushing TX buffer ...")
        time.sleep(1.0)
        os._exit(0)

if __name__ == '__main__':
    SerialDriver.BAUD_RATE = 9600
    cflib.crtp.init_drivers(enable_serial_driver=True)
    
    cf = Crazyflie(rw_cache='./cache')
    cf.console.receivedChar.add_callback(console_callback)
    
    scf = SyncCrazyflie(URI, cf=cf)
    scf.open_link()
    print('[host] Connected. Starting counter loop.')
    run_counter(scf)