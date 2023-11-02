import bleak
import asyncio
import logging

class BleBeaconHandler:
    def __init__(self):
        return
    
    def start_scan(self):
        
        asyncio.run(self.discover_ble_devices()) #returns a list of bleak.backends.device.BLEDevice


    async def discover_ble_devices(self):
        scanner = bleak.BleakScanner(detection_callback=self.discovery_callback)
        #scanner.register_detection_callback(self.discovery_callback)
        await scanner.start()
        await asyncio.sleep(20.0)  # scan for X seconds
        await scanner.stop()
        #devices = await scanner.discover(timeout=2.5)
        #return devices
    
    def discovery_callback(self, device, advertisement_data):
        if device.name == 'KBPro_keys':
            #print(f'Device: {device.name}, RSSI: {advertisement_data.rssi}')
            print(f'Service data: {advertisement_data.service_data}')


    def decode_acc_measurment(self, service_data)
        return [0, 0 , 0] #x, y, z