from datetime import datetime

from ultralytics import YOLO


def run_model(model, config_name, project):
    test_path = r"data\images\test"
    epochs = 300
    batch_size = 128
    patience = 12
    seed = 16
    name_train = "exp1train"
    name_test = "exp1test"

    start = datetime.now()
    model.train(data="config_8k.yaml", epochs=epochs, batch=batch_size, project="8k", name=name_train,
                deterministic=True, seed=seed, profile=True, plots=True, patience=patience)
    end = datetime.now()

    with open("time.txt", "a") as file:
        file.write(f"Project: {project}, Config: {config_name}, Epochs: {epochs}, Time: {end - start}\n")

    model.export(format="onnx")
    model.predict(source=test_path, save=True, save_txt=True, save_conf=True, project=project, name=name_test)
    return model


def run():
    model = YOLO("yolov9c.pt")
    run_model(model, "config_8k.yaml", "8k")

    run_model(model, "config_14k.yaml", "14k")

    run_model(model, "config_full.yaml", "full")


if __name__ == "__main__":
    run()
