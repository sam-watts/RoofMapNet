"""
Quick test to verify the training module is set up correctly.
"""

import sys
import torch


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import pytorch_lightning as pl
        print(f"✓ PyTorch Lightning {pl.__version__} imported successfully")
    except ImportError:
        print("✗ PyTorch Lightning not found. Install with: pip install pytorch-lightning")
        return False
    
    try:
        from roofmapnet.train import (
            RoofMapNetLightningModule,
            RoofMapNetDataModule,
            train,
            preprocess_roof_lines,
        )
        print("✓ Training module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import training module: {e}")
        return False
    
    try:
        from roofmapnet.models.roofmapnet import RoofMapNet
        print("✓ RoofMapNet model imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import model: {e}")
        return False
    
    try:
        from roofmapnet.datasets import WireframeDataset, collate
        print("✓ Dataset modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import datasets: {e}")
        return False
    
    return True


def test_model_initialization():
    """Test that the Lightning module can be initialized."""
    print("\nTesting model initialization...")
    
    try:
        from roofmapnet.train import RoofMapNetLightningModule
        
        model = RoofMapNetLightningModule(
            depth=4,
            head=[[2], [1], [2]],
            num_stacks=2,
            num_blocks=1,
            num_classes=5,
            lr=0.001,
        )
        print(f"✓ Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize model: {e}")
        return False


def test_forward_pass():
    """Test a forward pass through the model."""
    print("\nTesting forward pass...")
    
    try:
        from roofmapnet.train import RoofMapNetLightningModule
        import torch
        
        model = RoofMapNetLightningModule(
            depth=4,
            head=[[2], [1], [2]],
            num_stacks=2,
            num_blocks=1,
            num_classes=5,
        )
        model.eval()
        
        # Create dummy batch
        batch_size = 2
        images = torch.randn(batch_size, 3, 512, 512)
        
        # Create dummy targets
        targets = {
            'jmap': torch.zeros(batch_size, 1, 128, 128),
            'joff': torch.zeros(batch_size, 1, 2, 128, 128),
            'lmap': torch.zeros(batch_size, 128, 128),
        }
        
        # Create dummy meta
        meta_list = []
        for _ in range(batch_size):
            meta_list.append({
                'junc': torch.zeros(10, 2),
                'jtyp': torch.zeros(10, dtype=torch.uint8),
                'Lpos': torch.zeros(11, 11, dtype=torch.uint8),
                'Lneg': torch.zeros(11, 11, dtype=torch.uint8),
                'lpre': torch.zeros(340, 2, 2),
                'lpre_label': torch.zeros(340),
                'lpre_feat': torch.zeros(340, 8),
            })
        
        batch = (images, meta_list, targets)
        
        with torch.no_grad():
            output = model(batch)
        
        print(f"✓ Forward pass successful")
        print(f"  Output keys: {list(output.keys())}")
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_preprocessing():
    """Test data preprocessing function."""
    print("\nTesting data preprocessing...")
    
    try:
        from roofmapnet.train import preprocess_roof_lines
        import tempfile
        import os
        
        # Create dummy line data
        lines = [
            [(10, 20), (30, 40)],
            [(50, 60), (70, 80)],
            [(90, 100), (110, 120)],
        ]
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            preprocess_roof_lines(
                lines=lines,
                output_filename=tmp_path,
                image_size=512,
                heatmap_size=128,
            )
            
            # Verify output file was created and contains expected keys
            import numpy as np
            data = np.load(tmp_path)
            expected_keys = {'jmap', 'joff', 'lmap', 'junc', 'Lpos', 'Lneg', 'lpos', 'lneg'}
            actual_keys = set(data.keys())
            
            if expected_keys == actual_keys:
                print(f"✓ Preprocessing successful, generated all expected arrays")
                return True
            else:
                missing = expected_keys - actual_keys
                extra = actual_keys - expected_keys
                print(f"✗ Preprocessing output mismatch")
                if missing:
                    print(f"  Missing keys: {missing}")
                if extra:
                    print(f"  Extra keys: {extra}")
                return False
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    except Exception as e:
        print(f"✗ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("RoofMapNet Training Module Test Suite")
    print("=" * 60)
    
    results = []
    
    results.append(("Import Test", test_imports()))
    
    if results[-1][1]:  # Only continue if imports work
        results.append(("Model Initialization", test_model_initialization()))
        results.append(("Forward Pass", test_forward_pass()))
        results.append(("Data Preprocessing", test_preprocessing()))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result[1] for result in results)
    
    print("=" * 60)
    if all_passed:
        print("All tests passed! The training module is ready to use.")
        print("\nNext steps:")
        print("1. Prepare your data using preprocess_rid2.py")
        print("2. Update config/config.yaml with your settings")
        print("3. Run training with: python -m roofmapnet.train.main")
    else:
        print("Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
