import os
import streamlit as st

from chain import Chain


def intro_info(clips_path: str = 'test_clips') -> str:
    test_labels = [
        video_name.replace('.mp4', '') for video_name in os.listdir(clips_path)
    ]

    video_label = st.selectbox(
        'Please, choose test clip:',
        tuple(test_labels),
        index=None,
        placeholder="Select clip name...",
    )

    st.write('You selected:', video_label)

    video_path = os.path.join(
        clips_path, f'{video_label}.mp4',
        ) if video_label is not None else None
    
    device = st.selectbox(
        'Please, choose inference device:',
        ('cpu', 'cuda'),
        index=0,
        placeholder="Select device...",
    )

    st.write('You selected:', device)
    
    return video_path, device


def show_chain_results(
    input_video_path: str,
    model_weights: str,
    class_names_yaml: str,
    device: str,
    results_folder: str = 'results',
):
    chain = Chain(
        input_video_path,
        results_folder,
        model_weights,
        class_names_yaml,
        device=device,
        streamlit_log_flag=True,
    )

    with st.spinner('Chain inference...'):
        out_path = chain.infer()

    st.header('Results:')
    st.text(f'Rendered video saved at: {out_path}')

    video_file = open(out_path, 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)


def build_page(
    model_info_folder: str = 'model_info',
    class_labels_yaml_name: str = 'class_labels.yaml',
    weights_filename: str = 'yolov8m.pt',
):
    st.title('Road signs recognition project')

    with st.sidebar:
        video_path, device = intro_info()

    if video_path is not None:
        model_weights = os.path.join(
            model_info_folder, weights_filename,
        )
        classes_yaml = os.path.join(
            model_info_folder, class_labels_yaml_name,
        )
        show_chain_results(video_path, model_weights, classes_yaml, device)


if __name__ == '__main__':
    build_page()
