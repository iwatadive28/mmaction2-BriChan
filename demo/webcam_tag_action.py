# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time
from collections import deque
from operator import itemgetter
from threading import Thread

import cv2
import numpy as np
import torch
from mmengine import Config, DictAction
from mmengine.dataset import Compose, pseudo_collate
from mmaction.apis import init_recognizer
from mmaction.utils import get_str_type

# For recognize anything
import torchvision.transforms as transforms
import datetime
import os
import shutil
import glob
import pygame.mixer

from PIL import Image
from ram.models import ram
from ram import inference_ram

EXCLUED_STEPS = [
    'OpenCVInit', 'OpenCVDecode', 'DecordInit', 'DecordDecode', 'PyAVInit',
    'PyAVDecode', 'RawFrameDecode'
]

# WARNUNG_THING  = ['baseball bat', 'dagger', 'gun', 'knife', 'spear', 'rifle', 'saw', 'scissors', 'screwdriver', 'shears', 'throw']
WARNING_THING  = ['baseball bat', 'gun', 'knife', 'spear', 'rifle', 'saw', 'scissors', 'screwdriver', 'shears', 'throw']
WARNING_ACTION = ['Throw', 'Hitting', 'collide']

# Prepare recognize anything
SIZE       = 384
PRETRAINED = os.getenv("BRIQUE_BASE_PATH") + '/demo/pretrained/ram_swin_large_14m.pth'

tDevice   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.Resize((SIZE, SIZE)),transforms.ToTensor(), normalize])

tagModel = ram(pretrained = PRETRAINED, image_size = SIZE, vit = 'swin_l')
tagModel.eval()
tagModel = tagModel.to(tDevice)

# Initialize rec folder
REC_PATH = os.getenv("BRIQUE_BASE_PATH") + '/image/rec/'
shutil.rmtree(REC_PATH)
os.mkdir(REC_PATH)

CAMERA_W      = 640
CAMERA_H      = 480

fps           = 20
recSec        = 5
recFrame      = fps * recSec
warningCount  = 0
recCount      = 0
recFrameCount = 0
recFlag       = False
nowStr        = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

pygame.mixer.init(frequency = 44100)
pygame.mixer.music.load(os.getenv("BRIQUE_BASE_PATH") + '/sound/abunaiyokiwotsukete_01.wav')

frame = None

# Create movie method
def create_movie(timeStr: str):
    output = os.getenv("BRIQUE_BASE_PATH") + '/movie/' + timeStr + '_warn_video.mp4'
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    outfh  = cv2.VideoWriter(output, fourcc, fps, (CAMERA_W, CAMERA_H))

    for photo_name in sorted(glob.glob(REC_PATH + '/*.jpg'), key = os.path.getmtime):
        im = cv2.imread(photo_name)
        outfh.write(im)

    outfh.release()

# Recognize anything from image
def tagGenerate():
    global frame, nowStr, recFlag, recFrameCount, warningCount

    while True:
        bgr_image = cv2.resize(frame, (SIZE, SIZE))
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        image     = transform(Image.fromarray(rgb_image)).unsqueeze(0).to(tDevice)
        tags      = inference_ram(image, tagModel)
        print("Image Tags: ", tags[0])

        # Check warning item
        if 0 == warningCount:
            for i in WARNING_THING:
                if i in tags[0]:
                    pygame.mixer.music.play(1)
                    nowStr        = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
                    recFlag       = True
                    recFrameCount = recFrame
                    warningCount  = recFrame
                    cv2.imwrite(os.getenv("BRIQUE_BASE_PATH") + 'image/photo/' + nowStr + '_warn.jpg', cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
                    break

def parse_args():
    parser = argparse.ArgumentParser(description = 'Recognize Anything and MMAction2 webcam')

    parser.add_argument('config',     help = 'test config file path')
    parser.add_argument('checkpoint', help = 'checkpoint file/url')
    parser.add_argument('label',      help = 'label file')

    parser.add_argument(
        '--device',
        type    = str,
        default = 'cuda:0',
        help    = 'CPU/CUDA device option')
    parser.add_argument(
        '--camera-id',
        type    = int,
        default = 0,
        help    = 'camera device id')
    parser.add_argument(
        '--threshold',
        type    = float,
        default = 0.01,
        help    = 'recognition score threshold')
    parser.add_argument(
        '--average-size',
        type    = int,
        default = 1,
        help    = 'number of latest clips to be averaged for prediction')
    parser.add_argument(
        '--drawing-fps',
        type    = int,
        default = 20,
        help    = 'Set upper bound FPS value of the output drawing')
    parser.add_argument(
        '--inference-fps',
        type    = int,
        default = 4,
        help    = 'Set upper bound FPS value of model inference')
    parser.add_argument(
        '--cfg-options',
        nargs   = '+',
        action  = DictAction,
        default = {},
        help    = 'override some settings in the used config, the key-value pair '
                  'in xxx = yyy format will be merged into config file. For example, '
                  "'--cfg-options model.backbone.depth = 18 model.backbone.with_cp = True'")
    args = parser.parse_args()
    assert args.drawing_fps >= 0 and args.inference_fps >= 0, \
        'upper bound FPS value of drawing and inference should be set as ' \
        'positive number, or zero for no limit'
    return args


def show_results():
    print('Press "Esc", "q" or "Q" to exit')

    global frame, nowStr, recFlag, warningCount, recFrameCount, recCount

    text_info = {}
    cur_time  = time.time()
    prevTxt   = ''

    while True:
        msg = 'Waiting for action ...'

        _, frame = camera.read()
        frame_queue.append(np.array(frame[:, :, ::-1]))

        if len(result_queue) != 0:
            text_info = {}
            results   = result_queue.popleft()
            for i, result in enumerate(results):
                selected_label, score = result
                if score < threshold:
                    break
                location = (0, 40 + i * 20)
                text = selected_label + ': ' + str(round(score * 100, 2))
                text_info[location] = text
        elif len(text_info) != 0:
            if prevTxt != text_info:
                print('Action    : ' + str(text_info))
                prevTxt = text_info

            # Check warning action
            if 0 == warningCount:
                for i in WARNING_ACTION:
                    if i in str(text_info):
                        pygame.mixer.music.play(1)
                        nowStr        = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
                        recFlag       = True
                        recFrameCount = recFrame
                        warningCount  = recFrame
                        break

        cv2.imshow('camera', frame)
        ch = cv2.waitKey(1)

        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            camera.release()
            cv2.destroyAllWindows()
            break

        # Record a frame image
        wFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(REC_PATH + '/frame' + str(recCount).zfill(3) + '.jpg', cv2.cvtColor(wFrame, cv2.COLOR_BGR2RGB))

        # Countdown waning count
        if 0 < warningCount:
            warningCount -= 1

        if 0 < recFrameCount and True == recFlag:
            recFrameCount -=1

        # Save movie if event occurs
        if 0 == recFrameCount and True == recFlag:
            recFlag = False
            # Save photo
            print(nowStr)
            create_movie(nowStr)

        # Increase rec ount
        recCount += 1
        recCount %= (recFrame * 2)

        if drawing_fps > 0:
            # add a limiter for actual drawing fps <= drawing_fps
            sleep_time = 1 / drawing_fps - (time.time() - cur_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            cur_time = time.time()


def inference():
    score_cache = deque()
    scores_sum  = 0
    cur_time    = time.time()

    while True:
        cur_windows = []

        while len(cur_windows) == 0:
            if len(frame_queue) == sample_length:
                cur_windows = list(np.array(frame_queue))
                if data['img_shape'] is None:
                    data['img_shape'] = frame_queue.popleft().shape[:2]

        cur_data         = data.copy()
        cur_data['imgs'] = cur_windows
        cur_data         = test_pipeline(cur_data)
        cur_data         = pseudo_collate([cur_data])

        # Forward the model
        with torch.no_grad():
            result = model.test_step(cur_data)[0]
        scores = result.pred_score.tolist()
        scores = np.array(scores)
        score_cache.append(scores)
        scores_sum += scores

        if len(score_cache) == average_size:
            scores_avg          = scores_sum / average_size
            num_selected_labels = min(len(label), 5)

            score_tuples = tuple(zip(label, scores_avg))
            score_sorted = sorted(
                score_tuples, key=itemgetter(1), reverse=True)
            results = score_sorted[:num_selected_labels]

            result_queue.append(results)
            scores_sum -= score_cache.popleft()

            if inference_fps > 0:
                # add a limiter for actual inference fps <= inference_fps
                sleep_time = 1 / inference_fps - (time.time() - cur_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                cur_time = time.time()


def main():
    global average_size, threshold, drawing_fps, inference_fps, \
           device, model, camera, data, label, sample_length, \
           test_pipeline, frame_queue, result_queue, \
           frame, fps

    args          =  parse_args()
    average_size  = args.average_size
    threshold     = args.threshold
    drawing_fps   = args.drawing_fps
    fps           = args.drawing_fps
    inference_fps = args.inference_fps

    device = torch.device(args.device)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Build the recognizer from a config file and checkpoint file/url
    model  = init_recognizer(cfg, args.checkpoint, device = args.device)
    camera = cv2.VideoCapture(args.camera_id)
    data   = dict(img_shape = None, modality = 'RGB', label = -1)

    _, frame = camera.read()

    with open(args.label, 'r') as f:
        label = [line.strip() for line in f]

    # prepare test pipeline from non-camera pipeline
    cfg           = model.cfg
    sample_length = 0
    pipeline      = cfg.test_pipeline
    pipeline_     = pipeline.copy()

    for step in pipeline:
        if 'SampleFrames' in get_str_type(step['type']):
            sample_length     = step['clip_len'] * step['num_clips']
            data['num_clips'] = step['num_clips']
            data['clip_len']  = step['clip_len']
            pipeline_.remove(step)
        if get_str_type(step['type']) in EXCLUED_STEPS:
            # remove step to decode frames
            pipeline_.remove(step)
    test_pipeline = Compose(pipeline_)

    assert sample_length > 0

    try:
        frame_queue  = deque(maxlen = sample_length)
        result_queue = deque(maxlen = 1)

        pw = Thread(target = show_results, args = (), daemon = True)
        pr = Thread(target = inference,    args = (), daemon = True)
        pt = Thread(target = tagGenerate,  args = (), daemon = True)
        pw.start()
        pr.start()
        pt.start()
        pw.join()

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
