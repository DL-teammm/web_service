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
    chain: Chain,
    input_video_path: str,
):
    with st.spinner('Chain inference...'):
        out_path = chain.infer(input_video_path)

    st.header('Results:')
    st.text(f'Rendered video saved at: {out_path}')

    video_file = open(out_path, 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)


def build_page(
    model_info_folder: str = 'model_info',
    class_labels_yaml_name: str = 'class_labels.yaml',
    weights_filename: str = 'best.pt',
    results_folder: str = 'results',
):
    st.title('Road signs recognition project')

    with st.sidebar:
        video_path, device = intro_info()

    model_weights = os.path.join(
        model_info_folder, weights_filename,
    )
    classes_yaml = os.path.join(
        model_info_folder, class_labels_yaml_name,
    )

    chain = Chain(
        results_folder,
        model_weights,
        classes_yaml,
        device=device,
        streamlit_log_flag=True,
    )

    if video_path is not None:
        show_chain_results(chain, video_path)


if __name__ == '__main__':
    build_page()
