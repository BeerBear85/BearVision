# pylint: disable=E0401
import logging

if __name__ == "__main__":
    write_mode = 'w'
else:
    write_mode = 'a'

logging.basicConfig(filename='debug.log',
                    level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)-8s:%(name)s:%(message)s',
                    filemode=write_mode)

# Create a console handler and set level to info
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logging.getLogger('').addHandler(console_handler) # Add the console handler to the root logger

if __name__ == "__main__":
    import sys
    import os

    modules_abs_path = os.path.abspath("code/modules")
    sys.path.append(modules_abs_path)
    from ble_beacon_handler import BleBeaconHandler
    logger = logging.getLogger(__name__)

    logger.info("Starting BLE beacon handler test")
    my_ble_beacon_handler = BleBeaconHandler()
    my_ble_beacon_handler.start_scan()

    logger.info("BLE beacon handler test done")
