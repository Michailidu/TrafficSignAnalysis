from datetime import datetime

import click
from ultralytics import YOLO


def run_model(model, config_name, project, epochs, imgsz, batch_size):
    test_path = r"datasets\data\images\test"
    seed = 16

    name_train = "exp1train"
    name_test = "exp1test"

    start = datetime.now()
    model.train(data=config_name, epochs=epochs, batch=batch_size, project=project, name=name_train,
                seed=seed, profile=True, plots=True, imgsz=imgsz)
    end = datetime.now()

    with open("time.txt", "a") as file:
        file.write(f"Project: {project}, Config: {config_name}, Epochs: {epochs}, Time: {end - start}\n")

    model.export(format="onnx")
    model.predict(source=test_path, imgsz=imgsz, save=True, save_txt=True, save_conf=True, project=project,
                  name=name_test)
    return model


@click.command()
@click.option('--model-type', default="yolov9c.pt", help='Model type')
@click.option('--batch-size', default=16, help='Batch size', type=int)
def run(model_type, batch_size):
    configs = ["datasets/config_8k.yaml", "datasets/config_14k.yaml", "datasets/config_full.yaml"]
    epochs = [30, 10, 10]
    img_sizes = [640, 860, 1280]

    for img_size in img_sizes:
        for config in configs:
            model = YOLO(model_type)
            total_epochs = 0
            for epoch_count in epochs:
                total_epochs += epoch_count
                run_model(model, config, f'{config.split(".")[0]}_{total_epochs}_{img_size}', epoch_count, img_size,
                          batch_size)


if __name__ == "__main__":
    run()
