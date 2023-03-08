import shelve
import torch
from sklearn.metrics import f1_score
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from PicsDataset import PicsDataset


class SceneClassifier(torch.nn.Module):
    def __init__(self, scene_count):
        super().__init__()
        self.activation = nn.GELU()
        self.layer1 = nn.Linear(768, 128)
        self.norm1 = nn.BatchNorm1d(128)
        self.layer2 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.5)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.targets = np.array([])
        self.preds = np.array([])
        self.scene_count = scene_count
        self.scenes_pred = np.full((scene_count, 2), 0)
        self.scenes_target = np.full((scene_count), -1)
        self.f1 = 0

    def _forward(self, x):
        features = self.layer1(x)
        features = self.norm1(features)
        features = self.activation(features)
        features = self.dropout(features)
        pred = self.layer2(features)
        return pred

    def forward(self, pics, target=None, scene_numbers=None):
        output = self._forward(pics)
        loss = -1

        if target is not None:
            loss = self.loss_func(output, target)

            target = target.cpu()
            pred = torch.argmax(output, dim=-1).cpu()
            scene_numbers = scene_numbers.cpu()
            self.preds = np.concatenate((self.preds, pred.detach().numpy()), 0)
            self.targets = np.concatenate((self.targets, target.detach().numpy()), 0)
            for i in range(len(scene_numbers)):
                self.scenes_pred[scene_numbers[i]][pred[i]] += 1
                self.scenes_target[scene_numbers[i]] = target[i]

            self.f1 = f1_score(self.targets, self.preds)

        return loss, output

    def get_f1(self, reset=False):
        if reset:
            self.targets = np.array([])
            self.preds = np.array([])
        return self.f1

    def get_f1_for_scenes(self, reset=False):
        if reset:
            self.scenes_pred = np.full((self.scene_count, 2), 0)
            self.scenes_target = np.full((self.scene_count), -1)
            return

        mask = self.scenes_target != -1
        pred = self.scenes_pred[mask]
        target = self.scenes_target[mask]
        pred = torch.argmax(torch.Tensor(pred), dim=-1)
        f1 = f1_score(target, pred)

        return f1


class ShotSelect():
    def __init__(self, device, shelve_image_path, shelve_text_path):
        self.shelve_image_path = shelve_image_path
        self.shelve_text_path = shelve_text_path
        self.device = device

    def predict_linear(self, shots_count=20):
        ds = PicsDataset(path=self.shelve_image_path)
        dl = DataLoader(ds, batch_size=256, shuffle=True)
        model = SceneClassifier(0)
        #model.load_state_dict(torch.load('weights'))
        model.eval()

        scenes = torch.zeros(ds.get_scenes_count())
        for batch in dl:
            embedding, scene_local_i = batch
            _, output = model(embedding)
            pred = torch.max(output.cpu(), dim=-1)[0]
            pred = torch.nn.functional.softmax(pred, dim=0)
            for i in range(len(batch)):
                scenes[scene_local_i[i]] += pred[i].max()

        best_values, best_brames_idx = scenes.topk(shots_count, dim=0)
        return torch.sort(best_brames_idx)[0]


    def cosinus_dist_KMeans(self, shots_count=20, k=3):
        shelve_image = shelve.open(self.shelve_image_path, 'r')
        shelve_text = shelve.open(self.shelve_text_path, 'r')
        movie = shelve_image['movie']
        synopsis_embedding = shelve_text[movie]['synopsis_embedding'].to(self.device).to(torch.float32)
        clip_cosine_simularity = torch.zeros((shelve_image['capacity'], len(synopsis_embedding)))
        for i in range(shelve_image['capacity']):
            pic_embedding = shelve_image[f'{i}']['pic_embedding'].to(self.device)
            clip_cosine_simularity[i] = synopsis_embedding.matmul(pic_embedding.T)

        scenes_pred = torch.zeros(shelve_image[f'{i-1}']['shot_number_global'])
        best_values, best_brames_idx = clip_cosine_simularity.T.topk(k, dim=1)
        for i in range(len(best_brames_idx)):
            for j in range(k):
                scenes_pred[shelve_image[f'{best_brames_idx[i][j]}']['shot_number_global']] += best_values[i][j]

        best_values, best_brames_idx = scenes_pred.topk(shots_count, dim=0)
        return torch.sort(best_brames_idx)[0]


    def get_movie_frames(self, shelve_image):
        pics = []
        shot_number_global = []
        pred_shot_number_global = -1
        for key in range(shelve_image["capacity"]):
            frame = shelve_image[f'{key}']
            pics.append(frame['pic_embedding'])
            shot_number_global.append(frame['shot_number_global'])
            if pred_shot_number_global != frame['shot_number_global']:
                pred_shot_number_global = frame['shot_number_global']

        return torch.stack(pics), shot_number_global


    def cosinus_dist(self, shots_count=20):
        shelve_image = shelve.open(self.shelve_image_path, 'r')
        shelve_text = shelve.open(self.shelve_text_path, 'r')
        movie = shelve_image['movie']
        synopsis_embedding = shelve_text[movie]['synopsis_embedding'].to(self.device).to(torch.float32)
        pics, shot_number_global = self.get_movie_frames(shelve_image)
        clip_cosine_simularity = synopsis_embedding.matmul(nn.functional.normalize(pics).T.to(self.device))
        max_dist, _ = clip_cosine_simularity.max(dim=0)

        scenes_pred = torch.zeros(len(set(shot_number_global)))
        for i in range(len(max_dist)):
            scenes_pred[shot_number_global[i]] += max_dist[i]

        best_values, best_brames_idx = scenes_pred.topk(shots_count, dim=0)
        return torch.sort(best_brames_idx)[0]


    def cosinus_dist_with_bathcnorm(self, shots_count=20):
        shelve_image = shelve.open(self.shelve_image_path, 'r')
        shelve_text = shelve.open(self.shelve_text_path, 'r')
        movie = shelve_image['movie']
        synopsis_embedding = shelve_text[movie]['synopsis_embedding'].to(self.device).to(torch.float32)
        pic_embeddings = []

        for i in range(shelve_image['capacity']):
            pic_embeddings.append(shelve_image[f'{i}']['pic_embedding'].to(self.device))
        pic_embeddings = torch.stack(pic_embeddings)

        clip_cosine_similaity = synopsis_embedding.matmul(nn.functional.normalize(pic_embeddings.T)).T
        max_dist, _ = clip_cosine_similaity.max(dim=0)

        scenes_pred = torch.zeros(shelve_image[f'{i-1}']['shot_number_global'])
        for i in range(len(max_dist)):
            scenes_pred[shelve_image[f'{i}']['shot_number_global']] += max_dist[i]

        best_values, best_brames_idx = scenes_pred.topk(shots_count, dim=0)
        return torch.sort(best_brames_idx)[0]
