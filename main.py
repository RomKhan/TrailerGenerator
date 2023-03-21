import os
import sys
import cv2
import json
from tqdm import tqdm
from PIL import Image
import requests
from Enums import WorkType, Status, ModelType
from comress import compress
import pandas as pd
from scene_match import delete_tiny_scenes
from scenedetect import detect, ContentDetector
import numpy as np
import shelve
import gzip
import torch
import clip
from torchvision import transforms
from CLIPWrapperEval import CLIPWrapperEval
from model import ShotSelect
import ffmpeg
import scenedetect

frames_per_second = 5
frames_per_scene = 1


def get_clip(scene, cap, transform):
    shots = np.array([])
    assert cap.isOpened() != False, "Error opening video stream or file"
    end = scene[1]
    start = scene[0]
    central = np.linspace(0, end - start, num=2 + frames_per_scene) + start
    central = central.astype(np.int64)
    for current in central:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current)
        ret, frame = cap.read()
        if ret == True:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = transform(rgb)
            rgb = np.expand_dims(rgb, axis=0)
            if shots.shape[0] == 0:
                shots = rgb
            else:
                shots = np.concatenate((shots, rgb), axis=0)
        else:
            break
    return shots


def saveClipEmbedding(path_movie, path_csv, transform, wrapper, shelve_path, device):
    shelve_file = shelve.open(shelve_path, 'c')
    key = 0

    cap = cv2.VideoCapture(path_movie)
    df = pd.read_csv(path_csv)

    for i in range(df.shape[0]):
        pics = get_clip([df.at[i, 'start_frame'],
                         df.at[i, 'end_frame']],
                        cap,
                        transform)
        with torch.no_grad():
            embeddings = wrapper(torch.Tensor(pics).to(device)).cpu().to(torch.float32)
        for embedding in embeddings:
            data = {'pic_embedding': embedding,
                    'shot_number_global': i}
            shelve_file[f'{key}'] = data
            key += 1
    cap.release()
    shelve_file['capacity'] = key
    shelve_file['movie'] = path_movie
    shelve_file.close()


def saveTextEmbedding(path_movie, wrapper, shelve_path, device, json_path):
    shelve_file = shelve.open(shelve_path, 'c')

    with open(json_path) as f:
        data = json.load(f)
        synopsis = data['synopsis']
        summaries = data['Summaries']
        ganres = data['ganre']

    synopsis_sentences = synopsis.replace(',', '.').replace('!', '.').replace('\'', '.').replace('"', '.').replace('?', '.').split(".")
    summaries_sentences = summaries.replace(',', '.').replace('!', '.').replace('\'', '.').replace('"', '.').replace('?', '.').split(".")
    while "" in synopsis_sentences:
        synopsis_sentences.remove("")
    while "" in summaries_sentences:
        summaries_sentences.remove("")

    with torch.no_grad():
        synopsis_embedding = wrapper.forward_text(clip.tokenize(synopsis_sentences).to(device))
        summaries_embedding = wrapper.forward_text(clip.tokenize(summaries_sentences).to(device))
        ganres_embedding = wrapper.forward_text(clip.tokenize(ganres).to(device))

        data = {'synopsis_embedding': synopsis_embedding,
                'summaries_embedding': summaries_embedding,
                'ganres_embedding': ganres_embedding}
        shelve_file[path_movie] = data

    shelve_file.close()


def getUserEmbeddings(device, json_path):
    with open(json_path) as f:
        data = json.load(f)
        if 'user' not in data:
            return False, None
        user = data['user']

    user_sentences = user.replace(',', '.').replace('!', '.').replace('\'', '.').replace('"', '.').replace('?', '.').split(".")
    while "" in user_sentences:
        user_sentences.remove("")

    clip_model = CLIPWrapperEval(device)
    with torch.no_grad():
        user_embedding = clip_model.forward_text(clip.tokenize(user_sentences).to(device))

    return True, user_embedding


def scenes_detect(json_path, movie_path):
    try:
        with open(json_path, 'rb') as f:
            data = json.load(f)
        movie_threshold = data['movie_threshold']
    except:
        movie_threshold = 40

    movie_clips = detect(movie_path, ContentDetector(threshold=movie_threshold, min_scene_len=2))
    frames_per_second = 5
    delete_tiny_scenes(movie_clips, frames_per_second)
    movie_clips_count = len(movie_clips)
    table = pd.DataFrame(columns=['start_frame', 'end_frame'],
                         index=range(0, movie_clips_count))
    for scene_index in range(len(movie_clips)):
        table.at[scene_index, 'start_frame'] = movie_clips[scene_index][0].get_frames()
        table.at[scene_index, 'end_frame'] = movie_clips[scene_index][1].get_frames()

    table.to_csv(f'temp/scenes.csv')


def save_trailer(df, shots_idx, out_name, video_path):
    probe = ffmpeg.probe(video_path)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    fps = int(video_info['r_frame_rate'].split('/')[0]) / int(video_info['r_frame_rate'].split('/')[1])
    in_file = ffmpeg.input(video_path)
    trims = []


    for i in range(len(shots_idx)):
        start = df.at[shots_idx[i].item(), 'start_frame'] / fps
        end = df.at[shots_idx[i].item(), 'end_frame'] / fps
        vid = (
            in_file
                .trim(start=start, end=end)
                .setpts('PTS-STARTPTS')
        )
        aud = (
            in_file
                .filter_('atrim', start=start, end=end)
                .filter_('asetpts', 'PTS-STARTPTS')
        )

        trims.append(vid)
        trims.append(aud)


    joined = ffmpeg.concat(
        *trims, v=1,
        a=1, unsafe=True).node
    out = ffmpeg.output(joined[0], joined[1], f"{out_name}.mp4")
    out.run()


def convert_shots(compressed_frame_per_second, df, shots_idx):
    for i in shots_idx:
        df.at[i.item(), 'start_frame'] *= compressed_frame_per_second
        df.at[i.item(), 'end_frame'] *= compressed_frame_per_second
        df.at[i.item(), 'end_frame'] -= 4


def main():
    work_type = WorkType(int(sys.argv[1]))
    is_compressed = False if int(sys.argv[2]) == 0 else True
    is_scene_detected = False if int(sys.argv[3]) == 0 else True
    is_embeddings_created = False if int(sys.argv[4]) == 0 else True
    modelType = ModelType(int(sys.argv[5]))
    count_of_shots = int(sys.argv[6])
    device = sys.argv[7]
    movie = f'in{os.sep}{sys.argv[8]}'
    json_path = f'in{os.sep}{sys.argv[9]}'

    path_csv = 'temp/scenes.csv'
    path_movie = 'temp/movie.avi'
    shelve_image_path = 'temp/data_image'
    shelve_text_path = 'temp/data_text'
    side = 224

    if not os.path.exists('temp'):
        os.mkdir('temp')

    if work_type == WorkType.BYLINK:
        print('Not supported')
        return 0
    elif work_type != WorkType.BYPATH:
        print('incorrect work type')
        return 0

    if not is_compressed and not is_embeddings_created:
        print('compressing start')
        cap_movie = cv2.VideoCapture(movie)
        compress(cap_movie, path_movie, side)
        print('successful compressed')

    if not is_scene_detected and not is_embeddings_created:
        print('shots detecting start')
        scenes_detect(json_path, 'temp/movie.avi')
        print('shots detecting end')

    if not is_embeddings_created:
        clip_model = CLIPWrapperEval(device)
        transform = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.Resize((224, 224)),
            clip_model.preprocess
        ])

        print('image embeddings creating start')
        saveClipEmbedding(path_movie, path_csv, transform, clip_model, shelve_image_path, device)
        print('image embeddings creating end')

        print('text embeddings creating start')
        saveTextEmbedding(path_movie, clip_model, shelve_text_path, device, json_path)
        print('end embeddings creating end')

    is_user_description, embeddings = getUserEmbeddings(device=device, json_path=json_path)

    print('shots selecting start')
    model_api = ShotSelect(device, shelve_image_path, shelve_text_path)

    if modelType == ModelType.COSINUSDISTKMEANS:
        if is_user_description:
            print('User derscription not supported')
        best_shots_idx = model_api.cosinus_dist_KMeans(count_of_shots)
    elif modelType == ModelType.COSINUSDISTWITHBATCHNORM:
        if is_user_description:
            print('User derscription not supported')
        best_shots_idx = model_api.cosinus_dist_with_bathcnorm(count_of_shots)
    elif modelType == ModelType.COSINUSDIST:
        if not is_user_description:
            best_shots_idx = model_api.cosinus_dist(count_of_shots)
        else:
            best_shots_idx = model_api.cosinus_dist_with_custom_description(embeddings, count_of_shots)
    else:
        print('unknown model type')
        return 0
    print('shots selecting end')

    print('video saving start')
    df_scenes = pd.read_csv(path_csv)
    convert_shots(5, df_scenes, best_shots_idx)
    save_trailer(df_scenes, best_shots_idx, 'trailer', movie)
    print('video saving end')


if __name__ == "__main__":
    main()
