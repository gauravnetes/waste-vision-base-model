from ultralytics import YOLO
import os
import torch

def main():
    print("=== Waste Detection Training - 6K+ Dataset, 24 Classes ===")
    
    # Check GPU memory and optimize accordingly
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available GPU memory: {gpu_memory:.1f} GB")
        
        # Optimize settings based on available memory
        if gpu_memory >= 12:
            batch_size = 6
            img_size = 832
            workers = 8
        elif gpu_memory >= 8:
            batch_size = 4
            img_size = 832
            workers = 6
        else:
            batch_size = 4
            img_size = 640
            workers = 4
            print("⚠️  Limited GPU memory detected. Using smaller batch size.")
            
        print(f"Training config: batch={batch_size}, image_size={img_size}, workers={workers}")
    else:
        print("❌ No GPU detected! Training will be very slow.")
        return

    # Load YOLOv8 Large model - optimal for 6K+ dataset
    model_path = 'runs/waste_detection_6k_24classes4/weights/last.pt'
    model = YOLO(model_path)
    print("✅ Loaded YOLOv8-Large model")

    # Define paths
    data_yaml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset', 'data.yaml'))
    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'runs'))

    print(f"📊 Dataset path: {data_yaml_path}")
    print(f"💾 Results will be saved to: {project_path}")

    # Optimized training configuration for 24-class, 6K+ dataset
    print("\n🚀 Starting training...")
    results = model.train(
        # Core settings
        data=data_yaml_path,
        epochs=50,
        imgsz=img_size,
        batch=batch_size,
        project=project_path,
        name='waste_detection_6k_24classes',
        resume=True,
        
        # Optimizer settings (optimized for multi-class detection)
        optimizer='AdamW',
        lr0=0.001,           # Good starting point for 24 classes
        lrf=0.01,            # Final learning rate factor
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,     # Warmup for stable training
        
        # Loss weights (optimized for precision - important for 24 classes)
        box=7.5,             # Higher weight for better localization
        cls=1.0,             # Increased for 24-class classification
        dfl=1.5,             # Distribution focal loss
        
        # Data augmentation (balanced for multi-class)
        hsv_h=0.015,         # Hue augmentation
        hsv_s=0.7,           # Saturation
        hsv_v=0.4,           # Value/brightness
        degrees=15.0,        # Rotation (moderate for waste items)
        translate=0.1,       # Translation
        scale=0.9,           # Scale variation (important for waste)
        shear=2.0,           # Shear transformation
        perspective=0.0001,   # Perspective transformation
        flipud=0.0,          # No vertical flip for waste
        fliplr=0.5,          # Horizontal flip
        
        # Advanced augmentation
        mosaic=1.0,          # Mosaic augmentation (great for multi-class)
        mixup=0.15,          # Mixup (helps with class confusion)
        copy_paste=0.3,      # Copy-paste augmentation
        
        # Training optimization
        patience=20,         # Early stopping patience
        save_period=20,      # Save checkpoint frequency
        workers=workers,     # Data loading workers
        seed=42,             # Reproducible results
        
        # Performance optimizations
        amp=True,            # Automatic Mixed Precision
        fraction=1.0,        # Use full dataset
        profile=False,       # Disable profiling for speed
        
        # Validation settings
        val=True,
        plots=True,
        save_json=True,      # Save detailed metrics
        
        # Advanced settings
        cos_lr=True,         # Cosine learning rate scheduler
        close_mosaic=15,     # Disable mosaic in final epochs for precision
        auto_augment='randaugment',  # Additional augmentation
        
        # Multi-class specific
        single_cls=False,    # Multi-class detection
        overlap_mask=True,   # Handle overlapping objects
        mask_ratio=4,        # Mask downsampling ratio
    )
    
    print("\n✅ Training completed!")
    print(f"📈 Best results saved in: {project_path}/waste_detection_6k_24classes")
    print(f"🏆 Best model: {project_path}/waste_detection_6k_24classes/weights/best.pt")
    
    # Print final metrics if available
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print("\n📊 Final Metrics:")
        print(f"   mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
        print(f"   mAP50: {metrics.get('metrics/mAP50(B)', 'N/A')}")

if __name__ == '__main__':
    main()