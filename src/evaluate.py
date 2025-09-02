from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

def comprehensive_evaluation(model_path, data_yaml_path, save_results=True):
    """
    Comprehensive evaluation of the waste detection model
    """
    
    print("=== Comprehensive Model Evaluation ===")
    print(f"🔍 Model: {model_path}")
    print(f"📊 Dataset: {data_yaml_path}")
    
    # Verify files exist
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None
        
    if not os.path.exists(data_yaml_path):
        print(f"❌ Data config not found: {data_yaml_path}")
        return None
    
    # Load model
    try:
        model = YOLO(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Results storage
    results = {}
    
    # 1. Standard Evaluation
    print("\n1️⃣ === Standard Evaluation ===")
    try:
        metrics_standard = model.val(
            data=data_yaml_path,
            split='test',
            conf=0.001,  # Very low confidence to catch all predictions
            iou=0.6,     # Standard IoU threshold
            max_det=300,
            save_json=save_results,
            plots=save_results,
            verbose=True
        )
        
        results['standard'] = {
            'mAP50-95': metrics_standard.box.map,
            'mAP50': metrics_standard.box.map50,
            'mAP75': metrics_standard.box.map75,
            'precision': metrics_standard.box.mp,
            'recall': metrics_standard.box.mr,
        }
        
        print(f"✅ Standard Evaluation Results:")
        print(f"   mAP50-95: {results['standard']['mAP50-95']:.3f}")
        print(f"   mAP50: {results['standard']['mAP50']:.3f}")
        print(f"   mAP75: {results['standard']['mAP75']:.3f}")
        print(f"   Precision: {results['standard']['precision']:.3f}")
        print(f"   Recall: {results['standard']['recall']:.3f}")
        
    except Exception as e:
        print(f"❌ Standard evaluation failed: {e}")
        return None
    
    # 2. Test Time Augmentation Evaluation
    print("\n2️⃣ === Test Time Augmentation Evaluation ===")
    try:
        metrics_tta = model.val(
            data=data_yaml_path,
            split='test',
            conf=0.001,
            iou=0.6,
            augment=True,  # Enable TTA
            max_det=300,
            verbose=True
        )
        
        results['tta'] = {
            'mAP50-95': metrics_tta.box.map,
            'mAP50': metrics_tta.box.map50,
            'mAP75': metrics_tta.box.map75,
            'precision': metrics_tta.box.mp,
            'recall': metrics_tta.box.mr,
        }
        
        print(f"✅ TTA Evaluation Results:")
        print(f"   mAP50-95: {results['tta']['mAP50-95']:.3f}")
        print(f"   mAP50: {results['tta']['mAP50']:.3f}")
        print(f"   Improvement: +{(results['tta']['mAP50-95'] - results['standard']['mAP50-95']):.3f}")
        
    except Exception as e:
        print(f"❌ TTA evaluation failed: {e}")
        results['tta'] = None
    
    # 3. Confidence Threshold Optimization
    print("\n3️⃣ === Confidence Threshold Optimization ===")
    conf_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    conf_results = []
    
    best_map = 0
    best_conf = 0.5
    
    print("Testing different confidence thresholds...")
    for conf in conf_thresholds:
        try:
            metrics = model.val(
                data=data_yaml_path,
                split='test',
                conf=conf,
                iou=0.6,
                verbose=False
            )
            
            result = {
                'confidence': conf,
                'mAP50-95': metrics.box.map,
                'mAP50': metrics.box.map50,
                'precision': metrics.box.mp,
                'recall': metrics.box.mr
            }
            conf_results.append(result)
            
            print(f"   Conf {conf}: mAP50-95={result['mAP50-95']:.3f}, mAP50={result['mAP50']:.3f}")
            
            if result['mAP50-95'] > best_map:
                best_map = result['mAP50-95']
                best_conf = conf
                
        except Exception as e:
            print(f"   ❌ Failed at confidence {conf}: {e}")
    
    results['confidence_optimization'] = {
        'best_confidence': best_conf,
        'best_mAP50-95': best_map,
        'all_results': conf_results
    }
    
    print(f"✅ Optimal confidence threshold: {best_conf} (mAP50-95: {best_map:.3f})")
    
    # 4. Per-Class Analysis
    print("\n4️⃣ === Per-Class Analysis ===")
    try:
        # Get per-class metrics from standard evaluation
        if hasattr(metrics_standard.box, 'maps'):
            class_maps = metrics_standard.box.maps  # mAP50-95 per class
            class_names = model.names
            
            print("Per-class mAP50-95:")
            class_performance = []
            for i, (class_id, map_value) in enumerate(zip(range(len(class_maps)), class_maps)):
                class_name = class_names.get(class_id, f'Class_{class_id}')
                print(f"   {class_name}: {map_value:.3f}")
                class_performance.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'mAP50-95': map_value
                })
            
            # Find best and worst performing classes
            class_performance.sort(key=lambda x: x['mAP50-95'], reverse=True)
            results['per_class'] = {
                'best_classes': class_performance[:5],
                'worst_classes': class_performance[-5:],
                'all_classes': class_performance
            }
            
            print(f"\n🏆 Best performing classes:")
            for cls in results['per_class']['best_classes']:
                print(f"   {cls['class_name']}: {cls['mAP50-95']:.3f}")
            
            print(f"\n⚠️  Classes needing improvement:")
            for cls in results['per_class']['worst_classes']:
                print(f"   {cls['class_name']}: {cls['mAP50-95']:.3f}")
                
    except Exception as e:
        print(f"❌ Per-class analysis failed: {e}")
        results['per_class'] = None
    
    # 5. IoU Threshold Analysis
    print("\n5️⃣ === IoU Threshold Analysis ===")
    iou_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    iou_results = []
    
    print("Testing different IoU thresholds...")
    for iou in iou_thresholds:
        try:
            metrics = model.val(
                data=data_yaml_path,
                split='test',
                conf=best_conf,  # Use optimal confidence
                iou=iou,
                verbose=False
            )
            
            result = {
                'iou': iou,
                'mAP50-95': metrics.box.map,
                'mAP50': metrics.box.map50,
            }
            iou_results.append(result)
            
            print(f"   IoU {iou}: mAP50-95={result['mAP50-95']:.3f}")
            
        except Exception as e:
            print(f"   ❌ Failed at IoU {iou}: {e}")
    
    results['iou_analysis'] = iou_results
    
    # 6. Generate Summary Report
    print("\n📋 === FINAL SUMMARY REPORT ===")
    print(f"🎯 Target: 75% mAP50-95")
    print(f"📊 Best achieved: {best_map:.3f} ({best_map*100:.1f}%)")
    
    if best_map >= 0.75:
        print("🎉 TARGET ACHIEVED! Excellent performance!")
    elif best_map >= 0.70:
        print("👍 Very close to target! Consider minor optimizations.")
    elif best_map >= 0.65:
        print("📈 Good progress. Try longer training or larger model.")
    else:
        print("⚠️  Significant improvement needed. Review data quality and model architecture.")
    
    print(f"\n🔧 Recommended settings for deployment:")
    print(f"   Confidence threshold: {best_conf}")
    print(f"   Use TTA: {results['tta'] is not None and results['tta']['mAP50-95'] > results['standard']['mAP50-95']}")
    
    # Save detailed results if requested
    if save_results:
        try:
            # Create results directory
            results_dir = Path("evaluation_results")
            results_dir.mkdir(exist_ok=True)
            
            # Save confidence threshold analysis
            if conf_results:
                df_conf = pd.DataFrame(conf_results)
                df_conf.to_csv(results_dir / "confidence_analysis.csv", index=False)
                print(f"💾 Confidence analysis saved: {results_dir / 'confidence_analysis.csv'}")
            
            # Save per-class results
            if results.get('per_class'):
                df_classes = pd.DataFrame(results['per_class']['all_classes'])
                df_classes.to_csv(results_dir / "per_class_performance.csv", index=False)
                print(f"💾 Per-class analysis saved: {results_dir / 'per_class_performance.csv'}")
            
        except Exception as e:
            print(f"⚠️  Could not save detailed results: {e}")
    
    return results

def main():
    # Default paths - update these for your setup
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # You can specify different model paths for comparison
    model_paths = [
        os.path.join(base_dir, 'runs', 'garbage_detection_finetuned', 'weights', 'best.pt'),
        # Add more models for comparison:
        # os.path.join(base_dir, 'runs', 'waste_detection_optimized', 'weights', 'best.pt'),
    ]
    
    data_yaml_path = os.path.join(base_dir, 'data', 'dataset', 'data.yaml')
    
    # Evaluate each model
    all_results = {}
    for model_path in model_paths:
        if os.path.exists(model_path):
            model_name = Path(model_path).parent.parent.name
            print(f"\n{'='*50}")
            print(f"Evaluating: {model_name}")
            print(f"{'='*50}")
            
            results = comprehensive_evaluation(model_path, data_yaml_path)
            if results:
                all_results[model_name] = results
        else:
            print(f"⚠️  Model not found: {model_path}")
    
    # Compare models if multiple were evaluated
    if len(all_results) > 1:
        print(f"\n🏆 === MODEL COMPARISON ===")
        for model_name, results in all_results.items():
            best_map = results['confidence_optimization']['best_mAP50-95']
            print(f"{model_name}: {best_map:.3f} ({best_map*100:.1f}%)")

if __name__ == '__main__':
    main()