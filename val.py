from ultralytics import YOLO

# 1. Load your trained model
model = YOLO('weights.pt')

# 2. Run validation (it will automatically find the paths in data.yaml)
results = model.val(data='data.yaml')

print("Validation complete! Check the 'runs' folder.")