import time
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.crtp.serialdriver import SerialDriver

URI = 'serial://ttyAMA0'

def console_callback(text: str):
    print(text, end='')

if __name__ == '__main__':
    SerialDriver.BAUD_RATE = 9600
    cflib.crtp.init_drivers(enable_serial_driver=True)
    
    cf = Crazyflie(rw_cache='./cache')
    cf.console.receivedChar.add_callback(console_callback)
    
    lg_stab = LogConfig(name='Stabilizer', period_in_ms=100)
    lg_stab.add_variable('stabilizer.roll', 'float')
    lg_stab.add_variable('stabilizer.pitch', 'float')
    lg_stab.add_variable('stabilizer.yaw', 'float')

    with SyncCrazyflie(URI, cf=cf) as scf:
        print('[host] Connected, use ctrl-c to quit.')
        with SyncLogger(scf, lg_stab) as logger:
            endTime = time.time() + 10
            for log_entry in logger:
                timestamp = log_entry[0]
                data = log_entry[1]
                logconf_name = log_entry[2]

                print('[%d][%s]: %s' % (timestamp, logconf_name, data))

                if time.time() > endTime:
                    break