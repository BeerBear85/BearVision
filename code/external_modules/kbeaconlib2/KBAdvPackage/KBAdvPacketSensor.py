# FILEPATH: /E:/BearVision/repo/code/external_modules/kbeaconlib2/KBAdvPackage/KBAdvPacketSensor.py
from .KBAdvPacketBase import KBAdvPacketBase
from .KBUtility import KBUtility
from .KBAccSensorValue import KBAccSensorValue

class KBAdvPacketSensor(KBAdvPacketBase):
    SENSOR_MASK_VOLTAGE = 0x1
    SENSOR_MASK_TEMP = 0x2
    SENSOR_MASK_HUME = 0x4
    SENSOR_MASK_ACC_AIX = 0x8
    SENSOR_MASK_CUTOFF = 0x10
    SENSOR_MASK_PIR = 0x20
    SENSOR_MASK_LUX = 0x40
    SENSOR_MASK_VOC = 0x80
    SENSOR_MASK_CO2 = 0x200
    SENSOR_MASK_RECORD_NUM = 0x400

    def __init__(self):
        self.accSensor = None
        self.watchCutoff = None
        self.pirIndication = None
        self.temperature = None
        self.humidity = None
        self.batteryLevel = None
        self.luxValue = None
        self.vocElapseSec = None
        self.voc = None
        self.nox = None
        self.co2ElapseSec = None
        self.co2 = None
        self.newTHRecordNum = None

    def getAdvType(self):
        return KBAdvType.Sensor

    def getAccSensor(self):
        return self.accSensor

    def getWatchCutoff(self):
        return self.watchCutoff

    def getTemperature(self):
        return self.temperature

    def getHumidity(self):
        return self.humidity

    def getVersion(self):
        return 0

    def getBatteryLevel(self):
        return self.batteryLevel

    def getPirIndication(self):
        return self.pirIndication

    def getLuxValue(self):
        return self.luxValue

    def getVoc(self):
        return self.voc

    def getNox(self):
        return self.nox

    def getCo2(self):
        return self.co2

    def getNewTHRecordNum(self):
        return self.newTHRecordNum

    def getCo2ElapseSec(self):
        return self.co2ElapseSec

    def getVocElapseSec(self):
        return self.vocElapseSec

    def parseAdvPacket(self, beaconData):
        super().parseAdvPacket(beaconData)

        nSrvIndex = 1 #skip adv type

        sensorMaskHigh = ((beaconData[nSrvIndex] & 0xFF) << 8)
        nSensorMask = sensorMaskHigh + (beaconData[nSrvIndex+1] & 0xFF)
        nSrvIndex += 2

        if (nSensorMask & KBAdvPacketSensor.SENSOR_MASK_VOLTAGE) > 0:
            if nSrvIndex > (len(beaconData) - 2):
                return False
            nBatteryLvs = (beaconData[nSrvIndex] & 0xFF)
            nBatteryLvs = (nBatteryLvs << 8)
            nBatteryLvs += (beaconData[nSrvIndex+1] & 0xFF)
            self.batteryLevel = nBatteryLvs
            nSrvIndex += 2
        else:
            self.batteryLevel = None

        if (nSensorMask & KBAdvPacketSensor.SENSOR_MASK_TEMP) > 0:
            if nSrvIndex > (len(beaconData) - 2):
                return False
            tempHigh = beaconData[nSrvIndex]
            tempLow = beaconData[nSrvIndex+1]
            self.temperature = KBUtility.signedBytes2Float(tempHigh, tempLow)
            nSrvIndex += 2
        else:
            self.temperature = None

        if (nSensorMask & KBAdvPacketSensor.SENSOR_MASK_HUME) > 0:
            if nSrvIndex > (len(beaconData) - 2):
                return False
            humHigh = beaconData[nSrvIndex]
            humLow = beaconData[nSrvIndex+1]
            self.humidity = KBUtility.signedBytes2Float(humHigh, humLow)
            nSrvIndex += 2
        else:
            self.humidity = None

        if (nSensorMask & KBAdvPacketSensor.SENSOR_MASK_ACC_AIX) > 0:
            if nSrvIndex > (len(beaconData) - 6):
                return False
            self.accSensor = KBAccSensorValue()
            nAccValue = (beaconData[nSrvIndex] & 0xFF) << 8
            nAccValue += beaconData[nSrvIndex+1] & 0xFF
            self.accSensor.xAis = nAccValue

            nAccValue = (beaconData[nSrvIndex+2] & 0xFF) << 8
            nAccValue += beaconData[nSrvIndex+3] & 0xFF
            self.accSensor.yAis = nAccValue

            nAccValue = (beaconData[nSrvIndex+4] & 0xFF) << 8
            nAccValue += beaconData[nSrvIndex+5] & 0xFF
            self.accSensor.zAis = nAccValue
            nSrvIndex += 6
        else:
            self.accSensor = None

        if (nSensorMask & KBAdvPacketSensor.SENSOR_MASK_CUTOFF) > 0:
            if nSrvIndex > (len(beaconData) - 1):
                return False
            self.watchCutoff = beaconData[nSrvIndex]
            nSrvIndex += 1
        else:
            self.watchCutoff = None

        if (nSensorMask & KBAdvPacketSensor.SENSOR_MASK_PIR) > 0:
            if nSrvIndex > (len(beaconData) - 1):
                return False
            self.pirIndication = beaconData[nSrvIndex]
            nSrvIndex += 1
        else:
            self.pirIndication = None

        if (nSensorMask & KBAdvPacketSensor.SENSOR_MASK_LUX) > 0:
            if nSrvIndex > (len(beaconData) - 2):
                return False
            self.luxValue = (beaconData[nSrvIndex] & 0xFF) << 8
            self.luxValue += beaconData[nSrvIndex+1] & 0xFF
            nSrvIndex += 2
        else:
            self.luxValue = None

        if (nSensorMask & KBAdvPacketSensor.SENSOR_MASK_VOC) > 0:
            if nSrvIndex < (len(beaconData) - 5):
                return False
            self.vocElapseSec = beaconData[nSrvIndex] & 0xFF * 10
            self.voc = (beaconData[nSrvIndex+1] & 0xFF) << 8
            self.voc += beaconData[nSrvIndex+2] & 0xFF
            self.nox = (beaconData[nSrvIndex+3] & 0xFF) << 8
            self.nox += beaconData[nSrvIndex+4] & 0xFF
            nSrvIndex += 5
        else:
            self.voc = None
            self.nox = None

        if (nSensorMask & KBAdvPacketSensor.SENSOR_MASK_CO2) > 0:
            if nSrvIndex < (len(beaconData) - 3):
                return False
            self.co2ElapseSec = beaconData[nSrvIndex] & 0xFF * 10
            self.co2 = (beaconData[nSrvIndex+1] & 0xFF) << 8
            self.co2 += beaconData[nSrvIndex+2] & 0xFF
            nSrvIndex += 3
        else:
            self.co2 = None

        if (nSensorMask & KBAdvPacketSensor.SENSOR_MASK_RECORD_NUM) > 0:
            if nSrvIndex < (len(beaconData) - 3):
                return False
            if (beaconData[nSrvIndex] & 0x1) > 0:
                self.newTHRecordNum = (beaconData[nSrvIndex+1] & 0xFF) << 8
                self.newTHRecordNum += beaconData[nSrvIndex+2] & 0xFF
            nSrvIndex += 3
        else:
            self.newTHRecordNum = None

        return True

    @staticmethod
    def shortToInteger(s):
        if s < 0:
            return 65535+1+s
        else:
            return int(s)
