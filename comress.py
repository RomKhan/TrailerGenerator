import cv2
import os
import glob
from tqdm import tqdm

side = 224
frames_per_second = 5


def compress(cap, out_name):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_name, fourcc, frames_per_second, (side, side))
    sec = 0
    while True:
        ret, frame = cap.read()
        if ret:
            if sec % (round(cap.get(cv2.CAP_PROP_FPS) / frames_per_second)) == 0:
                b = cv2.resize(frame, (side, side), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
                out.write(b)
            sec += 1
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


movies_path = f'films'
if not os.path.exists('compressed'):
    os.mkdir('compressed')
for directory in tqdm(os.listdir(movies_path)):
    if not os.path.isdir(f'{movies_path}{os.sep}{directory}'):
        continue
    trailer_name = glob.glob(f'{movies_path}{os.sep}{directory}{os.sep}trailer.*')[0]
    movie_name = glob.glob(f'{movies_path}{os.sep}{directory}{os.sep}movie.*')[0]
    try:
        cap_trailer = cv2.VideoCapture(trailer_name)
        cap_movie = cv2.VideoCapture(movie_name)
        out_folder = f'compressed{os.sep}{directory}'
        if not os.path.exists(out_folder):
            os.mkdir(out_folder)
        compress(cap_trailer, out_folder + os.sep + 'trailer.avi')
        compress(cap_movie, out_folder + os.sep + 'movie.avi')
    except:
        print(directory)
