from datetime import datetime

import click
from ultralytics import YOLO


def run_model(model, config_name, project, epochs, imgsz, batch_size, name):
    seed = 16

    start = datetime.now()
    model.train(data=config_name, epochs=epochs, batch=batch_size, project=project, name=name,
                seed=seed, profile=True, plots=True, imgsz=imgsz, workers=1)
    end = datetime.now()

    with open("time.txt", "a") as file:
        file.write(f"Project: {project}, Config: {config_name}, Epochs: {epochs}, Time: {end - start}\n")

    model.export(format="onnx", imgsz=imgsz)
    return model


@click.command()
@click.argument('config-path', type=click.Path(exists=True))
@click.option('--model-type', default="yolov8n.pt", help='Model type')
@click.option('--batch-size', default=8, help='Batch size', type=int)
@click.option('--epochs', default=30, help='Number of epochs', type=int)
@click.option('--img-size', default=640, help='Image size', type=int)
@click.option('--name', default="run", help='Name of the run')
def run(config_path, model_type, batch_size, epochs, img_size, name):
    model = YOLO(model_type)
    run_model(model,
              config_path,
              f'{config_path.split(".")[0]}_{epochs}_{img_size}',
              epochs,
              img_size,
              batch_size,
              name)


if __name__ == "__main__":
    run()
