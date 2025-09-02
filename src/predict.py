from ultralytics import YOLO
import os
import argparse
import glob
from pathlib import Path

def predict_with_model(model_path, image_source, conf_threshold=0.5, save_results=True, use_tta=False):
    """
    Run prediction with the trained model
    
    Args:
        model_path: Path to the trained .pt model
        image_source: Path to image/folder/video for prediction
        conf_threshold: Confidence threshold (0-1)
        save_results: Whether to save prediction results
        use_tta: Use Test Time Augmentation for better accuracy
    """
    
    print(f"🔍 Loading model: {model_path}")
    
    # Verify model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None
        
    # Load model
    try:
        model = YOLO(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Verify source exists
    if not os.path.exists(image_source):
        print(f"❌ Source not found: {image_source}")
        return None
    
    # Count images if it's a directory
    if os.path.isdir(image_source):
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        image_count = sum(len(glob.glob(os.path.join(image_source, ext))) for ext in image_extensions)
        print(f"📁 Found {image_count} images in directory")
    
    print(f"⚙️  Configuration:")
    print(f"   Confidence threshold: {conf_threshold}")
    print(f"   Test Time Augmentation: {use_tta}")
    print(f"   Save results: {save_results}")
    
    # Run prediction with optimized settings
    print("\n🚀 Running prediction...")
    
    try:
        results = model.predict(
            source=image_source,
            
            # Prediction settings
            conf=conf_threshold,      # Confidence threshold
            iou=0.45,                 # IoU threshold for NMS
            agnostic_nms=False,       # Class-agnostic NMS
            max_det=300,              # Maximum detections per image
            
            # Augmentation
            augment=use_tta,          # Test Time Augmentation
            
            # Output settings
            save=save_results,        # Save images with predictions
            save_txt=save_results,    # Save labels in YOLO format
            save_conf=save_results,   # Save confidence scores
            save_crop=save_results,   # Save cropped detections
            
            # Visualization
            show_labels=True,         # Show class labels
            show_conf=True,           # Show confidence scores
            line_width=2,             # Bounding box line width
            
            # Performance
            half=True,                # Use half precision for speed
            device=None,              # Auto-select device
            
            # Advanced settings
            retina_masks=True,        # High resolution masks if available
            embed=None,               # Return feature vectors
        )
        
        print("✅ Prediction completed successfully!")
        
        # Print summary statistics
        if results:
            total_detections = sum(len(r.boxes) if r.boxes is not None else 0 for r in results)
            print(f"📊 Summary:")
            print(f"   Total images processed: {len(results)}")
            print(f"   Total detections: {total_detections}")
            print(f"   Average detections per image: {total_detections/len(results):.1f}")
            
            # Show class distribution
            if total_detections > 0:
                class_counts = {}
                for r in results:
                    if r.boxes is not None:
                        for cls_id in r.boxes.cls:
                            cls_name = model.names[int(cls_id)]
                            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                
                print("\n📈 Class Distribution:")
                for cls_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"   {cls_name}: {count}")
        
        if save_results:
            print(f"💾 Results saved in: runs/detect/predict*/")
            
        return results
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Enhanced YOLOv8 Waste Detection Prediction")
    
    parser.add_argument('--model', required=True, 
                       help='Path to the trained .pt model file')
    parser.add_argument('--source', required=True, 
                       help='Path to image, folder, or video for prediction')
    parser.add_argument('--conf', type=float, default=0.5, 
                       help='Confidence threshold (0-1, default: 0.5)')
    parser.add_argument('--no-save', action='store_true', 
                       help='Do not save prediction results')
    parser.add_argument('--tta', action='store_true', 
                       help='Use Test Time Augmentation for better accuracy')
    parser.add_argument('--benchmark', action='store_true', 
                       help='Run with different confidence thresholds for comparison')
    
    args = parser.parse_args()
    
    if args.benchmark:
        print("🏃‍♂️ Running benchmark with multiple confidence thresholds...")
        conf_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        for conf in conf_thresholds:
            print(f"\n--- Testing with confidence threshold: {conf} ---")
            predict_with_model(
                model_path=args.model,
                image_source=args.source,
                conf_threshold=conf,
                save_results=False,  # Don't save during benchmark
                use_tta=args.tta
            )
    else:
        # Single prediction run
        predict_with_model(
            model_path=args.model,
            image_source=args.source,
            conf_threshold=args.conf,
            save_results=not args.no_save,
            use_tta=args.tta
        )

if __name__ == '__main__':
    main()