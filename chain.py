import os
import yaml
import time
from collections import deque

import cv2
from ultralytics import YOLO
import streamlit as st


class Chain:
    def __init__(
        self, input_video_path: str,
        results_folder: str,
        model_weights: str,
        class_names_yaml: str,
        device: str = 'cpu',
        log_step_frames: int = 20,
        streamlit_log_flag: bool = False,
    ):
        self.model = YOLO(model_weights).to(device=device)

        with open(class_names_yaml, 'r') as f:
            yaml_dict = yaml.safe_load(f)
        self.class_mapping = yaml_dict['names']

        self.source = cv2.VideoCapture(input_video_path)

        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        self.out_path = os.path.join(
            results_folder, os.path.basename(input_video_path),
        )

        width = int(self.source.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.source.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.out_stream = cv2.VideoWriter(
            self.out_path, cv2.VideoWriter_fourcc(*'h256'),
            self.source.get(cv2.CAP_PROP_FPS), (width, height),
        )

        self.log_step_frames = log_step_frames
        self.curr_log_line = None

        self.model_inference_time = deque([], maxlen=self.log_step_frames)
        self.streamlit_flag = streamlit_log_flag

    def infer(self, conf_thr: float = 0.5):
        frame_count = 0

        while True:
            ret, frame = self.source.read()
        
            if not ret:
                break
            
            start_time = time.time()
            # Run YOLOv8 inference on the frame
            result = self.model.predict(frame, conf=conf_thr, verbose=False)[0]
            inference_time = time.time() - start_time

            self.model_inference_time.append(inference_time)

            if (frame_count + 1) % self.log_step_frames == 0:
                model_fps = len(self.model_inference_time) / sum(self.model_inference_time)
                road_signs_count = len(result)
                self.curr_log_line = f'At frame #{frame_count} detected {road_signs_count} road signs, model fps on last {self.log_step_frames} frames: {model_fps:.3f}'
                
                if self.streamlit_flag:
                    st.text(self.curr_log_line)

            for cls_id, custom_label in self.class_mapping.items():
                if cls_id in result.names: # check if the class id is in the results
                    result.names[cls_id] = custom_label # replace the class name with the custom label

            # Visualize the results on the frame
            rendered_frame = result.plot()

            self.out_stream.write(rendered_frame)
            frame_count += 1

        self.out_stream.release()

        return self.out_path


if __name__ == '__main__':
    video_path = os.path.join('test_clips', 'snow_falls.mp4')
    model_path = os.path.join('model_info', 'yolov8m.pt')
    class_names_info = os.path.join('model_info', 'class_labels.yaml')

    chain = Chain(video_path, 'results', model_path, class_names_info)
    out_path = chain.infer()
