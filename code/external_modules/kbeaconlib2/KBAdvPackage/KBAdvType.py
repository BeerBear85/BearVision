# FILEPATH: /E:/BearVision/repo/code/external_modules/kbeaconlib2/KBAdvPackage/KBAdvType.java
class KBAdvType:
    AdvNull = 0x0
    Sensor = 0x01
    EddyUID = 0x2
    EddyTLM = 0x3
    EddyURL = 0x4
    IBeacon = 0x5
    System = 0x6
    KBAdvTypeMAXValue = 0x6

    SensorString = "KSensor"
    EddyUIDString = "UID"
    EddyTLMString = "TLM"
    EddyURLString = "URL"
    IBeaconString = "iBeacon"
    SystemString = "System"
    InvalidString = "Disabled"

    @staticmethod
    def getAdvTypeString(nAdvType):
        strAdv = ""
        if nAdvType == KBAdvType.AdvNull:
            strAdv = KBAdvType.InvalidString
        elif nAdvType == KBAdvType.Sensor:
            strAdv = KBAdvType.SensorString
        elif nAdvType == KBAdvType.EddyUID:
            strAdv = KBAdvType.EddyUIDString
        elif nAdvType == KBAdvType.EddyTLM:
            strAdv = KBAdvType.EddyTLMString
        elif nAdvType == KBAdvType.EddyURL:
            strAdv = KBAdvType.EddyURLString
        elif nAdvType == KBAdvType.IBeacon:
            strAdv = KBAdvType.IBeaconString
        elif nAdvType == KBAdvType.System:
            strAdv = KBAdvType.SystemString
        else:
            strAdv = "Unknown"
        return strAdv
