import os
import glob
from tqdm import tqdm
from scenedetect import detect, ContentDetector
import pandas as pd
from torchmetrics import StructuralSimilarityIndexMeasure
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
)
import json
from scene_match import delete_tiny_scenes

mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]

def get_borders(trailer_path, movie_path):
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    trailer_clip = EncodedVideo.from_path(trailer_path).get_clip(start_sec=0.0, end_sec=999.0)
    movie_clip = EncodedVideo.from_path(movie_path).get_clip(start_sec=0.0, end_sec=999.0)
    transform = ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                NormalizeVideo(mean, std),
            ]
        ),
    )

    trailer_clip = transform(trailer_clip)["video"]
    movie_clip = transform(movie_clip)["video"]

    best_simmularity = -1
    start = -1
    for frame_start in range(movie_clip.shape[1]):
        frame_end = min(frame_start + 4, movie_clip.shape[1])
        simmularity = ssim(trailer_clip[:, :frame_end - frame_start, :, :], movie_clip[:, frame_start:frame_end, :, :])
        if best_simmularity < simmularity:
            best_simmularity = simmularity
            start = frame_start

    end = min(start + trailer_clip.shape[1], movie_clip.shape[1])
    return start, end


def match_final(in_folder, out_folder, trailer_clips_count, movie_clips_count, movie_scenes):
    table = pd.DataFrame(columns=['start_frame', 'end_frame', 'is_in_trailer', 'place','start_best', 'end_best', 'insertion_next'],
                         index=range(0, movie_clips_count))

    for scene_index in range(len(movie_scenes)):
        table.at[scene_index, 'start_frame'] = movie_scenes[scene_index][0].get_frames()
        table.at[scene_index, 'end_frame'] = movie_scenes[scene_index][1].get_frames()

    previous = -1
    for directory in os.listdir(in_folder):
        trailer_clip = glob.glob(f'{in_folder}{os.sep}{directory}{os.sep}trailer*')[0]
        movie_clip = [f for f in glob.glob(f'{in_folder}{os.sep}{directory}{os.sep}*') if "trailer" not in f]
        if len(movie_clip) != 0:
            movie_clip = movie_clip[0]
        elif previous > 0:
            table.at[previous, 'insertion_next'] = int(directory)
            continue
        else:
            continue

        borders = get_borders(trailer_clip, movie_clip)
        scene_index = int(movie_clip.split(f'{os.sep}')[-1].split('-')[0])
        place = float(directory) / trailer_clips_count
        previous = scene_index
        table.at[scene_index, 'is_in_trailer'] = 1
        table.at[scene_index, 'place'] = place
        table.at[scene_index, 'start_best'] = borders[0]
        table.at[scene_index, 'end_best'] = borders[1]
        table.at[scene_index, 'insertion_next'] = -1

    table.to_csv(f'{out_folder}{os.sep}match.csv')


matched_path = 'matched'
movies_path = 'compressed'
json_path = 'films'
if not os.path.exists('final'):
    os.mkdir('final')
for directory in tqdm(os.listdir(matched_path)):
    in_folder = f'{matched_path}{os.sep}{directory}'
    out_folder = f'final{os.sep}{directory}'
    if not os.path.isdir(in_folder) or os.path.isdir(out_folder):
        print(directory)
        continue
    match_folder = f'{matched_path}{os.sep}{directory}'

    try:
        with open(f'{json_path}{os.sep}{directory}{os.sep}data.json', 'rb') as f:
            data = json.load(f)
        movie_threshold = data['movie_threshold']
        trailer_threshold = data['trailer_threshold']
    except:
        continue

    trailer_name = glob.glob(f'{movies_path}{os.sep}{directory}{os.sep}trailer.*')[0]
    movie_name = glob.glob(f'{movies_path}{os.sep}{directory}{os.sep}movie.*')[0]
    trailer_clips = detect(trailer_name, ContentDetector(threshold=trailer_threshold, min_scene_len=2))
    movie_clips = detect(movie_name, ContentDetector(threshold=movie_threshold, min_scene_len=2))
    frames_per_second = 5
    delete_tiny_scenes(trailer_clips, frames_per_second)
    delete_tiny_scenes(movie_clips, frames_per_second)
    movie_clips_count = len(movie_clips)
    trailer_clips_count = len(trailer_clips)
    print(movie_clips_count)
    os.mkdir(out_folder)

    match_final(in_folder, out_folder, trailer_clips_count, movie_clips_count, movie_clips)
