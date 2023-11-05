# FILEPATH: /E:/BearVision/repo/code/external_modules/kbeaconlib2/KBAdvPackage/KBAdvPacketHandler.java
import logging
from typing import Dict, List, Union
from android.bluetooth.le import ScanRecord
from com.kkmcn.kbeaconlib2.KBUtility import APPLE_MANUFACTURE_ID, KKM_MANUFACTURE_ID, PARCE_UUID_EDDYSTONE, PARCE_UUID_EXT_DATA
from com.kkmcn.kbeaconlib2.KBAdvPackage.KBAdvPacketBase import KBAdvPacketBase
from com.kkmcn.kbeaconlib2.KBAdvPackage.KBAdvPacketEddyTLM import KBAdvPacketEddyTLM
from com.kkmcn.kbeaconlib2.KBAdvPackage.KBAdvPacketEddyUID import KBAdvPacketEddyUID
from com.kkmcn.kbeaconlib2.KBAdvPackage.KBAdvPacketEddyURL import KBAdvPacketEddyURL
from com.kkmcn.kbeaconlib2.KBAdvPackage.KBAdvPacketIBeacon import KBAdvPacketIBeacon
from com.kkmcn.kbeaconlib2.KBAdvPackage.KBAdvPacketSensor import KBAdvPacketSensor
from com.kkmcn.kbeaconlib2.KBAdvPackage.KBAdvPacketSystem import KBAdvPacketSystem
from com.kkmcn.kbeaconlib2.KBAdvType import AdvNull, EddyTLM, EddyUID, EddyURL, IBeacon, Sensor, System

MIN_EDDY_URL_ADV_LEN = 3
MIN_EDDY_UID_ADV_LEN = 18
MIN_EDDY_TLM_ADV_LEN = 14
MIN_IBEACON_ADV_LEN = 0x17
MIN_SYSTEM_ADV_LEN = 11
MIN_SENSOR_ADV_LEN = 3

LOG_TAG = "KBAdvPacketHandler"

class KBAdvPacketHandler:
    def __init__(self):
        self.batteryPercent = None
        self.filterAdvType = 0
        self.mAdvPackets = {}

    def advPackets(self) -> List[KBAdvPacketBase]:
        return list(self.mAdvPackets.values())

    def setAdvTypeFilter(self, filterAdvType: int):
        self.filterAdvType = filterAdvType

    def getBatteryPercent(self) -> Union[int, None]:
        return self.batteryPercent

    def getAdvPacket(self, nAdvType: int) -> Union[KBAdvPacketBase, None]:
        return self.mAdvPackets.get(str(nAdvType))

    def removeAdvPacket(self):
        self.mAdvPackets.clear()

    def parseAdvPacket(self, record: ScanRecord, rssi: int, name: str) -> bool:
        nAdvType = AdvNull
        beaconData = None
        bParseDataRslt = False

        if record.getManufacturerSpecificData() is not None:
            beaconData = record.getManufacturerSpecificData(APPLE_MANUFACTURE_ID)
            if beaconData is not None:
                if len(beaconData) == MIN_IBEACON_ADV_LEN and beaconData[0] == 0x2 and beaconData[1] == 0x15:
                    nAdvType = IBeacon
            else:
                beaconData = record.getManufacturerSpecificData(KKM_MANUFACTURE_ID)
                if beaconData is not None:
                    if beaconData[0] == 0x21 and len(beaconData) >= MIN_SENSOR_ADV_LEN:
                        nAdvType = Sensor
                    elif beaconData[0] == 0x22 and len(beaconData) >= MIN_SYSTEM_ADV_LEN:
                        nAdvType = System

        if record.getServiceData() is not None:
            eddyData = record.getServiceData(PARCE_UUID_EDDYSTONE)
            if eddyData is not None:
                beaconData = eddyData
                if eddyData[0] == 0x10 and len(eddyData) >= MIN_EDDY_URL_ADV_LEN:
                    nAdvType = EddyURL
                elif eddyData[0] == 0x0 and len(eddyData) >= MIN_EDDY_UID_ADV_LEN:
                    nAdvType = EddyUID
                elif eddyData[0] == 0x20 and len(eddyData) >= MIN_EDDY_TLM_ADV_LEN:
                    nAdvType = EddyTLM
                elif eddyData[0] == 0x21 and len(eddyData) >= MIN_SENSOR_ADV_LEN:
                    nAdvType = Sensor
                elif eddyData[0] == 0x22 and len(eddyData) >= MIN_SYSTEM_ADV_LEN:
                    nAdvType = System
                else:
                    nAdvType = AdvNull

        if (self.filterAdvType & nAdvType) == 0:
            return False

        byExtenData = record.getServiceData(PARCE_UUID_EXT_DATA)
        if byExtenData is not None and len(byExtenData) > 2:
            self.batteryPercent = int(byExtenData[0] & 0xFF)
            if self.batteryPercent > 100:
                self.batteryPercent = 100

        if nAdvType != AdvNull:
            strAdvTypeKey = str(nAdvType)
            advPacket = self.mAdvPackets.get(strAdvTypeKey)
            bNewObj = False
            if advPacket is None:
                classNewObj = kbAdvPacketTypeObjects.get(strAdvTypeKey)
                try:
                    if classNewObj is not None:
                        advPacket = classNewObj.newInstance()
                except Exception as excpt:
                    logging.exception(excpt)
                    logging.error("create adv packet class failed")
                    return False
                bNewObj = True

            if advPacket is not None and advPacket.parseAdvPacket(beaconData):
                advPacket.updateBasicInfo(rssi)
                if bNewObj:
                    self.mAdvPackets[strAdvTypeKey] = advPacket
                bParseDataRslt = True

        return bParseDataRslt
