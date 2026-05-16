#!/usr/bin/env python3
"""
YOLOv5 Object Detection Training on Custom Data
Based on: https://colab.research.google.com/github/roboflow/notebooks/blob/main/notebooks/train-yolov5-object-detection-on-custom-data.ipynb

This script trains a YOLOv5 model on custom object detection data from Roboflow.
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from datetime import datetime

def install_dependencies():
    """Install required packages"""
    print("Installing dependencies...")
    os.system("pip install -q roboflow")
    os.system("pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    os.system("pip install -q yolov5")
    print("✓ Dependencies installed")

def check_gpu():
    """Check GPU availability"""
    if torch.cuda.is_available():
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        return True
    else:
        print("⚠ GPU not available. Training will use CPU (slower)")
        return False

def download_dataset(api_key, workspace, project, version, format_type="yolov5"):
    """Download dataset from Roboflow"""
    print(f"\nDownloading dataset from Roboflow...")
    print(f"  Workspace: {workspace}")
    print(f"  Project: {project}")
    print(f"  Version: {version}")
    
    try:
        from roboflow import Roboflow
        
        rf = Roboflow(api_key=api_key)
        project_obj = rf.workspace(workspace).project(project)
        dataset = project_obj.version(version).download(format_type)
        
        print(f"✓ Dataset downloaded to: {dataset.location}")
        return dataset
    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        sys.exit(1)

def train_model(dataset_path, model_name="yolov5s", epochs=100, batch_size=16, 
                img_size=640, device=0, patience=20, save_period=10):
    """Train YOLOv5 model"""
    print(f"\nTraining YOLOv5 Model...")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {dataset_path}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Image Size: {img_size}")
    print(f"  Device: {'GPU' if device != 'cpu' else 'CPU'}")
    
    try:
        import yolov5
        
        # Load model
        print("\n  Loading model...")
        model = yolov5.load(f'{model_name}.pt')
        
        # Train
        print("  Starting training...")
        results = model.train(
            data=os.path.join(dataset_path, 'data.yaml'),
            epochs=epochs,
            imgsz=img_size,
            device=device,
            batch=batch_size,
            patience=patience,
            save_period=save_period,
            augment=True,
            cache='ram',
            workers=4,
            project='runs/detect',
            name=f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            exist_ok=False,
            pretrained=True,
            optimizer='SGD',
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            box=7.5,
            cls=0.5,
            cls_pw=1.0,
            obj=1.0,
            obj_pw=1.0,
            iou=0.7,
            label_smoothing=0.0,
            fliplr=0.5,
            flipud=0.0,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0,
            paste_in=0.0,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            multi_scale=False,
            save_txt=True,
            save_conf=False,
            save_json=False,
            verbose=True,
            half=False,
            dnn=False,
            rect=False,
            resume=False
        )
        
        print(f"✓ Training complete!")
        print(f"  Results saved to: {results.save_dir}")
        return results, model
        
    except Exception as e:
        print(f"✗ Error during training: {e}")
        sys.exit(1)

def validate_model(model, dataset_path):
    """Validate trained model"""
    print(f"\nValidating Model...")
    
    try:
        results = model.val(
            data=os.path.join(dataset_path, 'data.yaml'),
            imgsz=640,
            batch=16,
            conf=0.001,
            iou=0.6,
            device=0,
            half=False,
            save_json=True,
            project='runs/detect',
            name='val_results'
        )
        
        print(f"✓ Validation complete!")
        print(f"  mAP@0.5: {results.results_dict.get('metrics/mAP50', 'N/A')}")
        print(f"  mAP@0.5:0.95: {results.results_dict.get('metrics/mAP50-95', 'N/A')}")
        return results
        
    except Exception as e:
        print(f"✗ Error during validation: {e}")
        return None

def test_inference(model, test_image_path, conf_threshold=0.5):
    """Test inference on a sample image"""
    print(f"\nTesting Inference on: {test_image_path}")
    
    if not os.path.exists(test_image_path):
        print(f"⚠ Test image not found: {test_image_path}")
        return None
    
    try:
        results = model.predict(
            source=test_image_path,
            conf=conf_threshold,
            iou=0.45,
            imgsz=640,
            device=0,
            augment=False,
            visualize=False,
            line_width=3,
            half=False,
            dnn=False
        )
        
        print(f"✓ Inference complete!")
        print(f"  Detections: {len(results)}")
        
        # Show detections
        for i, result in enumerate(results):
            print(f"\n  Detection {i+1}:")
            print(f"    Classes detected: {result.names}")
            print(f"    Confidence: {result.conf}")
        
        return results
        
    except Exception as e:
        print(f"✗ Error during inference: {e}")
        return None

def export_model(model, export_format='torchscript'):
    """Export model to different formats"""
    print(f"\nExporting Model to {export_format}...")
    
    try:
        export_formats = ['torchscript', 'onnx', 'coreml', 'pb', 'tflite']
        
        if export_format not in export_formats:
            print(f"✗ Unsupported format. Choose from: {export_formats}")
            return None
        
        if export_format == 'torchscript':
            model.export(format='torchscript')
        elif export_format == 'onnx':
            model.export(format='onnx')
        else:
            print(f"⚠ Export to {export_format} requires additional setup")
        
        print(f"✓ Model exported successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error during export: {e}")
        return False

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Train YOLOv5 on custom data')
    
    # Roboflow arguments
    parser.add_argument('--api-key', type=str, default='qb5zteXrTNTRqVwRO5N0',
                        help='Roboflow API key')
    parser.add_argument('--workspace', type=str, default='shelf-product-detection',
                        help='Roboflow workspace name')
    parser.add_argument('--project', type=str, default='shelf-product-detection-5panm',
                        help='Roboflow project name')
    parser.add_argument('--version', type=int, default=4,
                        help='Dataset version')
    
    # Training arguments
    parser.add_argument('--model', type=str, default='yolov5s',
                        choices=['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'],
                        help='Model size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Image size')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device ID (use -1 for CPU)')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    
    # Action arguments
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip dataset download')
    parser.add_argument('--validate', action='store_true',
                        help='Validate model after training')
    parser.add_argument('--test-image', type=str,
                        help='Path to test image for inference')
    parser.add_argument('--export', type=str,
                        choices=['torchscript', 'onnx', 'coreml', 'pb', 'tflite'],
                        help='Export model format')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("YOLOv5 Custom Object Detection Training")
    print("=" * 60)
    
    # Install dependencies
    install_dependencies()
    
    # Check GPU
    device = args.device if args.device >= 0 else 'cpu'
    check_gpu()
    
    # Download dataset
    if not args.skip_download:
        dataset = download_dataset(
            args.api_key,
            args.workspace,
            args.project,
            args.version
        )
        dataset_path = dataset.location
    else:
        print("\nSkipping dataset download")
        dataset_path = './data'
    
    # Train model
    results, model = train_model(
        dataset_path,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        device=device,
        patience=args.patience
    )
    
    # Validate model
    if args.validate:
        validate_model(model, dataset_path)
    
    # Test inference
    if args.test_image:
        test_inference(model, args.test_image)
    
    # Export model
    if args.export:
        export_model(model, args.export)
    
    print("\n" + "=" * 60)
    print("Training pipeline complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()
