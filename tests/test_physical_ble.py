"""
Tests for real physical BLE tag communication.

This test only runs when explicitly triggered with: pytest -k physical_ble
It requires a physical KBPro BLE tag to be available and powered on.
"""

import asyncio
import sys
from pathlib import Path

import pytest

# Add module path
MODULE_DIR = Path(__file__).resolve().parents[1] / "code" / "modules"
sys.path.append(str(MODULE_DIR))

from ble_beacon_handler import BleBeaconHandler


class PhysicalBleTest:
    """Helper class to handle physical BLE tag testing."""
    
    def __init__(self):
        self.received_data = None
        self.data_received_event = asyncio.Event()
        self.handler = BleBeaconHandler()
        
    async def collect_ble_data(self, timeout=10.0):
        """Collect data from physical BLE tag with timeout."""
        # Override the advertisement queue processing to capture data
        original_process = self.handler.process_advertisements
        
        async def capture_data():
            """Capture the first advertisement data."""
            advertisement = await self.handler.advertisement_queue.get()
            self.received_data = advertisement
            self.data_received_event.set()
            print(f"\n--- BLE Tag Data Received ---")
            print(f"Tag ID (Address): {advertisement.get('address', 'N/A')}")
            print(f"Tag Name: {advertisement.get('name', 'N/A')}")
            print(f"RSSI: {advertisement.get('rssi', 'N/A')} dBm")
            if advertisement.get('distance') is not None:
                print(f"Distance: {advertisement.get('distance', 'N/A'):.2f} meters")
            
            acc = advertisement.get('acc_sensor')
            if acc:
                print(f"Accelerometer - X: {acc.x:.3f}g, Y: {acc.y:.3f}g, Z: {acc.z:.3f}g")
                print(f"Acceleration Norm: {acc.norm:.3f}g")
                print(f"Movement Detected: {acc.is_moving}")
            
            battery = advertisement.get('batteryLevel')
            if battery is not None:
                print(f"Battery Level: {battery}")
            print("--- End BLE Data ---\n")
            
        # Start both scanning and data capture
        scan_task = asyncio.create_task(self.handler.look_for_advertisements(timeout))
        capture_task = asyncio.create_task(capture_data())
        
        # Wait for either data to be received or timeout
        done, pending = await asyncio.wait(
            [scan_task, capture_task, asyncio.create_task(self.data_received_event.wait())],
            return_when=asyncio.FIRST_COMPLETED,
            timeout=timeout
        )
        
        # Cancel any remaining tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
                
        return self.received_data is not None


@pytest.mark.physical_ble
def test_physical_ble_tag_data():
    """
    Test that validates communication with a real BLE tag.
    
    Only runs when triggered with: pytest -k physical_ble
    
    Validates that:
    - Tag ID is received
    - RSSI (signal strength) is received 
    - Accelerometer data is received with expected format (X, Y, Z)
    
    The test fails if no data is retrieved.
    """
    
    print("\n=== Physical BLE Tag Test ===")
    print("Looking for KBPro BLE tags...")
    print("Make sure a physical BLE tag is powered on and nearby.")
    
    ble_test = PhysicalBleTest()
    
    # Try to collect BLE data with 10 second timeout
    success = asyncio.run(ble_test.collect_ble_data(timeout=10.0))
    
    # Test must fail if no data is received
    assert success, "No BLE tag data was received. Ensure a physical KBPro tag is powered on and nearby."
    
    # Validate required data is present
    data = ble_test.received_data
    assert data is not None, "No advertisement data captured"
    
    # Validate Tag ID (address)
    tag_id = data.get('address')
    assert tag_id is not None and tag_id != '', f"Tag ID missing or empty: {tag_id}"
    
    # Validate RSSI  
    rssi = data.get('rssi')
    assert rssi is not None, f"RSSI missing: {rssi}"
    assert isinstance(rssi, (int, float)), f"RSSI not numeric: {type(rssi)}"
    
    # Validate accelerometer data
    acc_sensor = data.get('acc_sensor')
    assert acc_sensor is not None, "Accelerometer data missing"
    
    # Validate accelerometer has X, Y, Z values
    assert hasattr(acc_sensor, 'x') and acc_sensor.x is not None, f"Accelerometer X missing: {getattr(acc_sensor, 'x', None)}"
    assert hasattr(acc_sensor, 'y') and acc_sensor.y is not None, f"Accelerometer Y missing: {getattr(acc_sensor, 'y', None)}"
    assert hasattr(acc_sensor, 'z') and acc_sensor.z is not None, f"Accelerometer Z missing: {getattr(acc_sensor, 'z', None)}"
    
    # Validate accelerometer values are numeric
    assert isinstance(acc_sensor.x, (int, float)), f"Accelerometer X not numeric: {type(acc_sensor.x)}"
    assert isinstance(acc_sensor.y, (int, float)), f"Accelerometer Y not numeric: {type(acc_sensor.y)}"
    assert isinstance(acc_sensor.z, (int, float)), f"Accelerometer Z not numeric: {type(acc_sensor.z)}"
    
    print("✓ All validations passed - Physical BLE tag communication successful!")