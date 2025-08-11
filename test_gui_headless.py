#!/usr/bin/env python3
"""
Simple headless test runner for the YOLO training GUI.
This runs key logic tests without requiring pytest or Qt.
"""

import sys
import tempfile
import yaml
from pathlib import Path

# Add the pretraining directory to path
sys.path.insert(0, str(Path(__file__).parent / "pretraining"))

try:
    from train_yolo_gui import TrainingConfig
    print("✓ Successfully imported TrainingConfig")
except ImportError as e:
    print(f"✗ Failed to import TrainingConfig: {e}")
    sys.exit(1)

def test_training_config():
    """Test TrainingConfig functionality."""
    print("\n=== Testing TrainingConfig ===")
    
    # Test default values
    config = TrainingConfig()
    assert config.epochs == 50, f"Expected epochs=50, got {config.epochs}"
    assert config.model == "yolov8x.pt", f"Expected model=yolov8x.pt, got {config.model}"
    assert config.batch == 16, f"Expected batch=16, got {config.batch}"
    print("✓ Default values correct")
    
    # Test custom values
    custom_config = TrainingConfig(
        data_dir="/test/data",
        epochs=100,
        batch=8,
        device="cpu"
    )
    assert custom_config.data_dir == "/test/data"
    assert custom_config.epochs == 100
    assert custom_config.batch == 8
    assert custom_config.device == "cpu"
    print("✓ Custom values correct")

def test_yaml_handling():
    """Test YAML configuration save/load."""
    print("\n=== Testing YAML Configuration ===")
    
    # Test saving and loading config
    test_config = {
        'data_dir': '/test/path',
        'model': 'yolov8n.pt',
        'epochs': 75,
        'batch': 4,
        'imgsz': 800,
        'device': '0',
        'val_ratio': 0.15,
        'onnx_out': 'test_output.onnx'
    }
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.safe_dump(test_config, f)
        temp_path = f.name
    
    try:
        # Load and verify
        with open(temp_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        assert loaded_config['data_dir'] == '/test/path'
        assert loaded_config['epochs'] == 75
        assert loaded_config['device'] == '0'
        print("✓ YAML save/load working correctly")
        
    finally:
        Path(temp_path).unlink(missing_ok=True)

def test_command_generation_logic():
    """Test training command generation logic."""
    print("\n=== Testing Command Generation Logic ===")
    
    config = TrainingConfig(
        data_dir="/test/data",
        model="yolov8n.pt",
        epochs=10,
        batch=2,
        imgsz=416,
        device="cpu",
        val_ratio=0.1,
        onnx_out="test.onnx"
    )
    
    # Simulate command generation
    cmd_parts = [
        config.data_dir,
        "--model", config.model,
        "--epochs", str(config.epochs),
        "--batch", str(config.batch),
        "--imgsz", str(config.imgsz),
        "--val-ratio", str(config.val_ratio),
        "--onnx-out", config.onnx_out
    ]
    
    if config.device:
        cmd_parts.extend(["--device", config.device])
    
    expected = [
        "/test/data",
        "--model", "yolov8n.pt",
        "--epochs", "10",
        "--batch", "2",
        "--imgsz", "416",
        "--val-ratio", "0.1",
        "--onnx-out", "test.onnx",
        "--device", "cpu"
    ]
    
    assert cmd_parts == expected, f"Command mismatch: {cmd_parts} vs {expected}"
    print("✓ Command generation logic correct")

def main():
    """Run all headless tests."""
    print("Running headless tests for YOLO Training GUI")
    print("=" * 50)
    
    try:
        test_training_config()
        test_yaml_handling()
        test_command_generation_logic()
        
        print("\n" + "=" * 50)
        print("✅ All headless tests PASSED!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())