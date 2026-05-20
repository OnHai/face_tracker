import time
import os
import serial
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crtp.serialdriver import SerialDriver

URI = 'serial://ttyAMA0'
PAR = 'uhel_group.custom_val'
PORT = f"/dev/{URI.split('://')[1]}"

def resync_uart_state():
    print(f"Resyncing UART on {PORT}...")
    try:
        with serial.Serial(PORT, 9600, timeout=0.5) as s:
            s.reset_input_buffer()
            s.reset_output_buffer()
            
            s.write(b'\xFF' * 100)
            s.flush()
            time.sleep(0.2)
            
            s.reset_input_buffer()
    except Exception as e:
        print(f"UART resync failed: {e}")

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
        print("\nFlushing TX buffer and forcing hardware exit...")
        time.sleep(1.0)
        os._exit(0)

if __name__ == '__main__':
    resync_uart_state()
    
    SerialDriver.BAUD_RATE = 9600
    cflib.crtp.init_drivers(enable_serial_driver=True)
    
    cf = Crazyflie(rw_cache='./cache')
    cf.console.receivedChar.add_callback(console_callback)
    
    scf = SyncCrazyflie(URI, cf=cf)
    scf.open_link()
    print('[host] Connected. Starting counter loop.')
    run_counter(scf)