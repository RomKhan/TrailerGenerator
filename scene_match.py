import numpy as np
import scenedetect
from scenedetect import detect, ContentDetector
from FramesDataset import FramesDataset
from CLIPWrapper import CLIPWrapper
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from PIL import Image
import os
import glob


def delete_tiny_scenes(scene_list, frames_per_second):
    delta = scenedetect.FrameTimecode(timecode="00:00:00.750", fps=frames_per_second)
    scene_list_mock = list(scene_list)
    for i in range(len(scene_list_mock)):
        if scene_list_mock[i][1] - scene_list_mock[i][0] < delta:
            scene_list.remove(scene_list_mock[i])


def extract_frames(scenes, frames_per_scene=1):
    central_frames = []
    for scene in scenes:
        start = scene[0].get_frames()
        end = scene[1].get_frames()
        # extract two frame from interval
        central = np.linspace(0, end - start, num=2 + frames_per_scene) + start
        central_frames += list((central[1:-1].astype(int)))
    return (central_frames)


def much_embeddings(scene_list_trailer, frames_per_scene_trailer, ds_movie, distances, k=3):
    _, nearest = torch.topk(distances, k=k, largest=False)
    dict_indexes = {}
    for i in range(len(scene_list_trailer)):
        scenes = {}
        for j in range(frames_per_scene_trailer):
            for num in nearest[i * frames_per_scene_trailer + j]:
                if ds_movie[num][1] not in scenes:
                    scenes[ds_movie[num][1]] = [0, 0]
                scenes[ds_movie[num][1]][0] += 1
                scenes[ds_movie[num][1]][1] += distances[i][num]

        dict_indexes[i] = sorted(scenes.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True)[0]
    return dict_indexes


def match(movie_path, trailer_path):
    frames_per_second = 5

    # Разбиение на сцены
    scene_list_trailer = detect(trailer_path, ContentDetector(threshold=28.0, min_scene_len=2))
    scene_list_movie = detect(movie_path, ContentDetector(threshold=28.0, min_scene_len=2))

    # Удаление маленьких сцен
    delete_tiny_scenes(scene_list_trailer, frames_per_second)
    delete_tiny_scenes(scene_list_movie, frames_per_second)

    # Определение номеров кадров, которые пойдут в датасет
    frames_per_scene_trailer = 4
    frames_per_scene_movie = 8
    central_frames_trailer = extract_frames(scene_list_trailer, frames_per_scene_trailer)
    central_frames_movie = extract_frames(scene_list_movie, frames_per_scene_movie)

    # Создаем датасеты
    ds_trailer = FramesDataset(trailer_path, central_frames_trailer, frames_per_scene_trailer)
    ds_movie = FramesDataset(movie_path, central_frames_movie, frames_per_scene_movie)

    # Оборачиваем в даталодеры
    dl_trailer = DataLoader(ds_trailer, batch_size=4, shuffle=False)
    dl_movie = DataLoader(ds_movie, batch_size=4, shuffle=False)

    # Создаем модель
    clip_model = CLIPWrapper("cpu")

    transform = transforms.Compose([
        lambda x: Image.fromarray(x),  # to PIL
        transforms.Resize((224, 224)),
        clip_model.preprocess
    ])
    ds_trailer.transforms = transform
    ds_movie.transforms = transform

    device = "cuda"
    model = CLIPWrapper(device)

    # Получение эмбеддингов
    embeddings_trailer = torch.tensor([])
    for frame in dl_trailer:
        with torch.no_grad():
            emb = model(frame[0].to(device))
            embeddings_trailer = torch.cat((embeddings_trailer, emb.detach().cpu()))

    embeddings_movie = torch.tensor([])
    for frame in dl_movie:
        with torch.no_grad():
            emb = model(frame[0].to(device))
            embeddings_movie = torch.cat((embeddings_movie, emb.detach().cpu()))

    # Расчет косинусоидного расстояния
    cosine_similarity = embeddings_trailer.matmul(embeddings_movie.T)  # scalar product [-1 .. 1]
    cosine_distances = (1 - cosine_similarity) / 2  # [-1 ..1 ] -> [0 .. 1]
    cosine_distances.fill_diagonal_(1)

    # Получаем сцены, в которых модель больше всего уверена
    dict_indexes = much_embeddings(scene_list_trailer=scene_list_trailer,
                                   frames_per_scene_trailer=frames_per_scene_trailer,
                                   ds_movie=ds_movie,
                                   distances=cosine_distances,
                                   k=3)
    return dict_indexes, (scene_list_trailer, scene_list_movie)


def save_matching(dict_indexes, trailer_path, movie_path, scene_list_trailer, scene_list_movie, out_folder):
    for i in range(len(dict_indexes)):
        os.mkdir(f'{out_folder}{os.sep}{i}')
        os.chdir(f'{out_folder}{os.sep}{i}')
        scenedetect.video_splitter.split_video_ffmpeg(f'..{os.sep}..{os.sep}..{os.sep}' + trailer_path, [scene_list_trailer[i]])
        scenedetect.video_splitter.split_video_ffmpeg(f'..{os.sep}..{os.sep}..{os.sep}' + movie_path, [scene_list_movie[dict_indexes[i][0]]],
                                                      video_name=f'{dict_indexes[i][0]}')
        os.chdir(f'..{os.sep}..{os.sep}..')


movies_path = f'compressed'
if not os.path.exists('matched'):
    os.mkdir('matched')
for directory in tqdm(os.listdir(movies_path)):
    if not os.path.isdir(f'{movies_path}{os.sep}{directory}'):
        continue

    out_folder = f'matched{os.sep}{directory}'
    if os.path.isdir(out_folder):
        continue
    os.mkdir(out_folder)

    trailer_name = glob.glob(f'{movies_path}{os.sep}{directory}{os.sep}trailer.*')[0]
    movie_name = glob.glob(f'{movies_path}{os.sep}{directory}{os.sep}movie.*')[0]
    data = match(movie_name, trailer_name)

    save_matching(data[0], trailer_name, movie_name, data[1][0], data[1][1], out_folder)
