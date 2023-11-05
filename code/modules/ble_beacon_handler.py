import bleak
import asyncio
import ctypes
import logging

logger = logging.getLogger(__name__)

KSENSOR_TYPE = 0x21
MIN_SENSOR_ADV_LEN = 3
SENSOR_MASK_VOLTAGE = 0x1
SENSOR_MASK_ACC_AIX = 0x8


class AccSensorValue:
    def __init__(self):
        self.x = None
        self.y = None
        self.z = None
        self.norm = None
        self.diff_from_1_g = None
    def get_value_string(self):
        return f'x: {self.x}, y: {self.y}, z: {self.z}, diff_from_1_g: {self.diff_from_1_g}'

class BleBeaconHandler:
    def __init__(self):
        self.advertisement_queue = asyncio.Queue()
        return
    
    def start_scan(self, timeout=0.0):
        #asyncio.run(self.look_for_advertisements(timeout))
        asyncio.run(self.start_scan_async(timeout))
        return
    
    async def start_scan_async(self, timeout=0.0):
        await asyncio.gather(self.process_advertisements(), self.look_for_advertisements(timeout))
        return
    
    async def process_advertisements(self):
        while True:
            advertisement = await self.advertisement_queue.get()
            acc = advertisement['acc_sensor']
            print(acc.get_value_string())
            self.advertisement_queue.task_done()
        return

    async def look_for_advertisements(self, timeout = 0.0):
        scanner = bleak.BleakScanner(detection_callback=self.discovery_callback)
        await scanner.start()
        if timeout == 0.0:
            future = asyncio.Future() # scan indefinitely
            await future
        else:
            await asyncio.sleep(timeout)  # scan for X seconds
        await scanner.stop()
        return
    
    async def discovery_callback(self, device, advertisement_data):
        if (device.name is not None) and ('KBPro' in device.name):
            logger.debug('Device: %s, RSSI: %s' % (device.name, advertisement_data.rssi))
            #print(f'Service data: {hex(advertisement_data.service_data)}')
            beaconData = advertisement_data.service_data

            battery_level= None
            acc_sensor = None
            for key in beaconData.keys():
                data = beaconData[key]
                type_indicator = data[0]
                #type_indicator_hex = hex(type_indicator)
                if type_indicator == KSENSOR_TYPE and len(data) >= MIN_SENSOR_ADV_LEN:
                    battery_level, acc_sensor = self.decode_sensor_data(data)
            if acc_sensor is not None:
                # make dict with relevant data
                data_dict = {}
                data_dict['address'] = device.address
                data_dict['name'] = device.name
                data_dict['rssi'] = advertisement_data.rssi
                data_dict['tx_power'] = advertisement_data.tx_power
                data_dict['batteryLevel'] = battery_level
                data_dict['acc_sensor'] = acc_sensor
                logger.info('Adding meas to que: acc_sensor: %s' % (acc_sensor.get_value_string()))
                await self.advertisement_queue.put(data_dict)
        return




    def decode_sensor_data(self, data):
        # see page 26 in the KBPro user manual
        nSrvIndex = 1 #skip frame type

        #First two bytes are the sensor mask
        sensor_mask = data[nSrvIndex:nSrvIndex+2]
        sensor_mask = int.from_bytes(sensor_mask, byteorder='big')
        #print(f'Sensor mask: {bin(sensor_mask)}')
        
        #for i in range(0, len(data)):
        #    print(f'data [{i}]: {hex(data[i])} {bin(data[i])}')

        batteryLevel = None
        accSensor = None

        if (sensor_mask & SENSOR_MASK_VOLTAGE) > 0:
            nSrvIndex = 3
            nBatteryLvs = data[nSrvIndex:nSrvIndex+2]
            nBatteryLvs = int.from_bytes(nBatteryLvs, byteorder='big')
            batteryLevel = nBatteryLvs

        if (sensor_mask & SENSOR_MASK_ACC_AIX) > 0:
            nSrvIndex = 7
            accSensor = AccSensorValue()
            bytes_x = data[nSrvIndex:nSrvIndex+2]
            accSensor.x = int.from_bytes(bytes_x, byteorder='big')
            accSensor.x = ctypes.c_int16(accSensor.x).value
            accSensor.x = accSensor.x/1000.0
            # Extract the next two bytes and convert them to an integer
            bytes_y = data[nSrvIndex+2:nSrvIndex+4]
            accSensor.y = int.from_bytes(bytes_y, byteorder='big')
            accSensor.y = ctypes.c_int16(accSensor.y).value
            accSensor.y = accSensor.y/1000.0
            # Extract the next two bytes and convert them to an integer
            bytes_z = data[nSrvIndex+4:nSrvIndex+6]
            accSensor.z = int.from_bytes(bytes_z, byteorder='big')
            accSensor.z = ctypes.c_int16(accSensor.z).value
            accSensor.z = accSensor.z/1000.0
            #calculate the norm
            accSensor.norm = (accSensor.x**2 + accSensor.y**2 + accSensor.z**2)**0.5
            accSensor.diff_from_1_g = abs(accSensor.norm - 1.0)
            #print(f'x: {accSensor.x}, y: {accSensor.y}, z: {accSensor.z}, norm: {accSensor.norm}, diff_from_1_g: {accSensor.diff_from_1_g}')

        return batteryLevel, accSensor