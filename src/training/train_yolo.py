from ultralytics import YOLO

def main():
    # YOLOv8n pretrained on COCO as lightweight baseline model
    # for fast training on small datasets and transfer learning benefitrs
    model = YOLO("yolov8n.pt", task="detect")
    
    print("Model loaded: yolov8n.pt")

    try:
        print("Loading dataset from: data/yolo/dataset.yaml")

        results = model.train(
            data="data/yolo/dataset.yaml",
            epochs=50,
            imgsz=416,
            batch=8
        )
        print("Training completed")
        print(results)

    except Exception as e:
        print("Training failed with error")

if __name__ == "__main__":
    main()