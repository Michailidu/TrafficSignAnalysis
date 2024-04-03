from ultralytics import YOLO


def run():
    test_path = r"data\images\train\full"
    epochs = 300
    batch_size = 128
    patience = 16
    seed = 16
    name_train = "exp1train"
    name_test = "exp1test"

    # 8k dataset
    model = YOLO("yolov9c.yaml")
    model.train(data="config_8k.yaml", epochs=epochs, batch=batch_size, project="8k", name=name_train,
                deterministic=True, seed=seed, profile=True, plots=True, patience=patience)
    model.export(format="onnx")
    model.predict(source=test_path, save=True, save_txt=True, save_conf=True, project="8k", name=name_test)

    # 14k dataset
    model.train(data="config_14k.yaml", epochs=epochs, batch=batch_size, project="14k", name=name_train,
                deterministic=True, seed=seed, profile=True, plots=True, patience=patience)
    model.export(format="onnx")
    model.predict(source=test_path, save=True, save_txt=True, save_conf=True, project="14k", name=name_test)

    # full dataset
    model.train(data="config_full.yaml", epochs=epochs, batch=batch_size, project="full", name=name_train,
                deterministic=True, seed=seed, profile=True, plots=True, patience=patience)
    model.export(format="onnx")
    model.predict(source=test_path, save=True, save_txt=True, save_conf=True, project="full", name=name_test)


if __name__ == "__main__":
    run()
