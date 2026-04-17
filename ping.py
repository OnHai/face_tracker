#!/usr/bin/env python3

import logging
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger

# Enable debug output
logging.basicConfig(level=logging.DEBUG)

URI = 'serial://ttyAMA0'

def console_callback(text: str):
    print(text, end='')
    
if __name__ == '__main__':
    print('[1] Initializing drivers...')
    cflib.crtp.init_drivers(enable_serial_driver=True)
    print('[2] Drivers initialized')
    
    cf = Crazyflie(rw_cache='./cache')
    cf.console.receivedChar.add_callback(console_callback)
    print('[3] Crazyflie object created')
    
    lg_stab = LogConfig(name='Stabilizer', period_in_ms=10)
    lg_stab.add_variable('stabilizer.roll', 'float')
    lg_stab.add_variable('stabilizer.pitch', 'float')
    lg_stab.add_variable('stabilizer.yaw', 'float')
    print('[4] Log config created')

    print('[5] Attempting SyncCrazyflie connection...')
    with SyncCrazyflie(URI) as scf:
        print('[6] Connected, use ctrl-c to quit.')
        print('[7] Starting SyncLogger...')
        with SyncLogger(scf, lg_stab) as logger:
            print('[8] Logger started')
            endTime = time.time() + 10
            for log_entry in logger:
                timestamp = log_entry[0]
                data = log_entry[1]
                logconf_name = log_entry[2]

                print('[%d][%s]: %s' % (timestamp, logconf_name, data))

                if time.time() > endTime:
                    break
    print('[9] Done')
