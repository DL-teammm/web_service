import os
import yaml
import time
from collections import deque

import cv2
from ultralytics import YOLO
import streamlit as st


class Chain:
    def __init__(
        self,
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

        self.out_videos_folder = os.path.join(results_folder, 'videos')
        self.out_logs_folder = os.path.join(results_folder, 'logs')

        for path in [self.out_videos_folder, self.out_logs_folder]:
            if not os.path.exists(path):
                os.makedirs(path)

        self.log_step_frames = log_step_frames
        self.curr_log_line = None

        self.streamlit_flag = streamlit_log_flag

    def infer(self, input_video_path: str, conf_thr: float = 0.5):
        source = cv2.VideoCapture(input_video_path)
        video_label = os.path.basename(input_video_path).split('.')[0]

        out_path = os.path.join(
            self.out_videos_folder, f'{video_label}.mp4',
        )

        width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_stream = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*'h256'),
            source.get(cv2.CAP_PROP_FPS), (width, height),
        )

        log_file = os.path.join(self.out_logs_folder, f'{video_label}.txt')

        model_inference_time = deque([], maxlen=self.log_step_frames)

        frame_count = 0

        while True:
            ret, frame = source.read()
        
            if not ret:
                break
            
            start_time = time.time()
            # Run YOLOv8 inference on the frame
            result = self.model.predict(frame, conf=conf_thr, verbose=False)[0]
            inference_time = time.time() - start_time

            model_inference_time.append(inference_time)

            if (frame_count + 1) % self.log_step_frames == 0:
                model_fps = len(model_inference_time) / sum(model_inference_time)
                road_signs_count = len(result)
                self.curr_log_line = f'At frame #{frame_count} detected {road_signs_count} road signs, model fps on last {self.log_step_frames} frames: {model_fps:.3f}'
                
                if self.streamlit_flag:
                    st.text(self.curr_log_line)

                open_file_mode = 'w' if frame_count + 1 == self.log_step_frames else 'a'

                with open(log_file, open_file_mode) as f:
                    f.write(self.curr_log_line + '\n')

            for cls_id, custom_label in self.class_mapping.items():
                if cls_id in result.names: # check if the class id is in the results
                    result.names[cls_id] = custom_label # replace the class name with the custom label

            # Visualize the results on the frame
            rendered_frame = result.plot()

            out_stream.write(rendered_frame)
            frame_count += 1

        out_stream.release()

        return out_path


if __name__ == '__main__':
    video_path = os.path.join('test_clips', 'snow_falls.mp4')
    model_path = os.path.join('model_info', 'yolov8m.pt')
    class_names_info = os.path.join('model_info', 'class_labels.yaml')

    chain = Chain(video_path, 'results', model_path, class_names_info)
    out_path = chain.infer()
