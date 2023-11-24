import argparse
import os
import time
from tqdm import tqdm
import yaml
import numpy as np
import torch
import yolov5
from ultralytics import YOLO
import supervision as sv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompareModel():
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(**kwargs)
        self._validate_inputs()

    @classmethod
    def from_yaml(cls, yaml_path):
        with open(yaml_path, 'r') as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)

        return cls(**yaml_data)

    def _validate_inputs(self):
        if not self.model_name in ["yolov5", "yolov8"]:
            raise ValueError("Invalid model name. Please use 'yolov5' or 'yolov8'.")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("Model could not found in given location. Please check your model path.")
        # TODO: More validation will be added later on

    def perform_yolo_model(self):
        logger.info(f"Performing the {self.model_name} model.")
        self.model = yolov5.load(self.model_path, device="cuda:0") if self.model_name == "yolov5" else YOLO(self.model_path)
        if not self.calculate_map:
            images = [os.path.join(self.image_path, image) for image in os.listdir(self.image_path)]
            start_time = time.time()
            for image in tqdm(images):
                predictions = self.model(image, size=self.image_size) if self.model_name == "yolov5" else self.model.predict(
                    image, save=self.write_results, imgsz=self.image_size, conf=self.conf_thresh,
                    project = self.project_folder_name, name = self.project_name, device="cuda" if torch.cuda.is_available() else "cpu")[0]
            end_time = time.time()
            logger.info(f"Total Time Spent on Prediction: {end_time - start_time} seconds.")

        elif self.calculate_map:
            start_time = time.time()
            self._calculate_map()
            end_time = time.time()
            logger.info(f"Total Time Spent on Both Prediction and Calculation: {end_time - start_time} seconds.")

    def _calculate_map(self):
        try:
            ground_truth_dataset = sv.DetectionDataset.from_yolo(
                images_directory_path=self.image_path,
                annotations_directory_path=self.annotation_path,
                data_yaml_path=self.yaml_path,
                force_masks=self.act_mask
            )

            def _callback(image: np.array) -> sv.Detections:
                results = self.model(image, size=self.image_size) if self.model_name == "yolov5" else self.model.predict(
                    image, save=self.write_results, imgsz=self.image_size, conf=self.conf_thresh, verbose = self.verbose,
                    device="cuda" if torch.cuda.is_available() else "cpu")[0]
                return sv.Detections.from_ultralytics(results)

            mean_average_precision = sv.MeanAveragePrecision.benchmark(dataset=ground_truth_dataset, callback=_callback)
            logger.info(f"\nmAP50: {mean_average_precision.map50} \nmAP75: {mean_average_precision.map75} \nmAP50-95: {mean_average_precision.map50_95}")

        except ImportError:
            logger.error("Please clone the supervision repository and install it by writing 'pip install -e .'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Computer Vision models.")
    parser.add_argument('--config', '--cfg', help="Path to the configuration file")
    args = parser.parse_args()

    if args.config:
        try:
            calculated_mean_average_precision = CompareModel.from_yaml(args.config)
            calculated_mean_average_precision.perform_yolo_model()
        except Exception as e:
            print(f"An error occurred while initializing the CompareModel: {e}")
    else:
        print("Please provide a configuration file using the '--config' option.")
