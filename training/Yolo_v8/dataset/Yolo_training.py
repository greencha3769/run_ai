from ultralytics import YOLO

def main():
    model = YOLO("yolov8s.pt")

    model.train(
        data=r"C:\Product_Anomaly_Detection\training\Yolo_v8\dataset\data.yaml",
        # 학습
        epochs=150,          # 300장이므로 100보다 조금 길게
        patience=30,

        # 입력
        imgsz=640,
        batch=4,

        # 최적화
        optimizer="auto",
        lr0=0.005,
        cos_lr=True,

        # 증강
        close_mosaic=15,     # 마지막 15 epoch는 Mosaic 끄기

        # 시스템
        device=0,
        workers=4,
        cache=False,

        # 저장
        project="runs",
        name="milkis_detect_s",
        exist_ok=True,
    )

if __name__ == "__main__":
    main()