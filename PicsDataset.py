import shelve
from torch.utils.data import Dataset


class PicsDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.shelve_path = shelve.open(path, 'r')
        self.lenght = self.shelve_path['capacity']

    def __getitem__(self, n):
        data = self.shelve_path[f'{n}']
        embedding = data['pic_embedding']
        scene_local_i = data['shot_number_global']

        return embedding, scene_local_i

    def __len__(self):
        return self.lenght

    def __del__(self):
        self.shelve_path.close()

    def get_scenes_count(self):
        capacity = self.shelve_path['capacity']
        return self.shelve_path[f'{capacity-1}']['shot_number_global'] + 1

    def get_target(self):
        return self.shelve_path['target']

