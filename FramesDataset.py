from torch.utils.data import Dataset
import cv2


class FramesDataset(Dataset):
    def __init__(self, path, frame_nums, frames_per_scene, transforms=None):
        super().__init__()
        self.data = []
        self.scenes = []
        current_scene = 0
        self.transforms = transforms
        cap = cv2.VideoCapture(path)
        assert cap.isOpened() != False, "Error opening video stream or file"
        for frame_number in frame_nums:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if ret == True:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.data.append(rgb)
                self.scenes.append(current_scene // frames_per_scene)
                current_scene += 1
            else:
                break
        cap.release()

    def __getitem__(self, n):
        out = self.data[n]
        scene = self.scenes[n]
        if self.transforms:
            out = self.transforms(out)
        return out, scene

    def __len__(self):
        return len(self.data)