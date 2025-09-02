from ultralytics import YOLO
import os

def main():
    # --- CONFIGURATION ---
    # Path to your best-performing model from the previous run
    model_path = 'runs/waste_detection_yolov8_large_single_class/weights/best.pt'
    
    # Path to your data configuration file
    data_yaml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset', 'data.yaml'))
    
    # Directory to save the new fine-tuning run
    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'runs'))
    
    # --- SCRIPT START ---
    print(f"🚀 Starting fine-tuning for model: {model_path}")
    
    # Load your best model
    model = YOLO(model_path)
    
    # Fine-tune the model for 20 more epochs with a lower learning rate
    model.train(
        data=data_yaml_path,
        epochs=20,          # The new total number of epochs (50 + 20)
        lr0=0.0001,         # Learning rate 10x smaller for fine-tuning
        
        # It's important to include the other key parameters from your last run
        imgsz=640,
        batch=4,            # Keep the batch size the same
        project=project_path,
        name='garbage_detection_finetuned',
        optimizer='AdamW',
        patience=10         # You can use a lower patience for fine-tuning
    )
    
    print("\n✅ Fine-tuning complete!")
    print(f"📈 Check for improvements in: {project_path}/garbage_detection_finetuned")

if __name__ == '__main__':
    main()