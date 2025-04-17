
# # print(SETTINGS['datasets_dir'])
# # import torch
# # print("GPU Available:", torch.cuda.is_available())




# from ultralytics import YOLO

# def main():
#     # model = YOLO('yolov8n.pt')
#     # model.train(data='data/cards/data.yaml', epochs=100, imgsz=640)


#     model = YOLO("runs/detect/train3/weights/last.pt")  # or adjust path like train2/ if needed
#     model.train(resume=True)

# if __name__ == '__main__':
#     main()


# from ultralytics import YOLO

# def main():
#     # model = YOLO("yolov8x.pt")  # YOLOv8-X — highest capacity model

#     # model.train(
#     #     data="data/cards/data.yaml",  # your dataset
#     #     epochs=200,                   # train long for best performance
#     #     imgsz=640,                    # standard YOLO image size
#     #     batch=16,                     # safe for 16GB GPU (try 32 if you want to push it)
#     #     workers=8,                    # for faster data loading
#     #     device=0,                     # use GPU 0
#     #     augment=True,                 # enable built-in data augmentation
#     #     close_mosaic=10,              # turn off mosaic after 10 epochs (helps stability)
#     #     patience=50,                  # early stopping patience (optional)
#     #     name="cards_x_large",         # custom run name
#     #     save=True,                    # save best.pt and last.pt
#     #     verbose=True                  # log all training steps
#     # )



#     model = YOLO("runs/detect/train3/weights/last.pt")  # or adjust path like train2/ if needed
#     # model.train(resume=True, epochs =200)
#     model.train(data="data/cards/data.yaml", epochs=200, batch=16, imgsz=640, name="train3_continue")

from ultralytics import YOLO

def main():
    # Load a mid-size model (better generalization than yolov8x with less overfitting)
    model = YOLO("yolov8m.pt")

    # Train the model with robust settings for your ~2700 image card dataset
    model.train(
        data="data/cards/data.yaml",         # Your dataset config file
        epochs=200,                          # Enough time to converge
        imgsz=640,                           # Standard YOLO image size
        batch=16,                            # Safe for 16GB GPU (adjust if needed)
        name="cards_yolov8m_augmented",      # Name of the run folder

        # 📦 Strong Augmentations
        mosaic=1.0,                          # Combine 4 images into 1 (fakes multi-card scenes)
        scale=0.7,                           # Random zoom in/out
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,   # Simulate different lighting
        fliplr=0.5,                          # Flip cards horizontally
        flipud=0.0,                          # Don't flip vertically (cards stay upright)
        copy_paste=0.1,                      # Simulate card stacking
        erasing=0.3,                         # Random occlusion / partial visibility
        auto_augment="randaugment",         # Extra built-in randomness

        # 🧠 Overfitting Prevention
        close_mosaic=10,                     # Turn off mosaic after 10 epochs for stability
        patience=50,                         # Stop early if no improvement
        freeze=None,                         # Optional: freeze backbone for transfer learning

        # 🧰 Hardware Settings
        device=0,                            # GPU index
        workers=8,                           # Dataloader threads

        # 📊 Logging & Saving
        save=True,
        save_period=-1,                      # Only save best and last
        plots=True,                          # Save loss + metrics curves
        verbose=True
    )

if __name__ == "__main__":
    main()



