import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import concurrent.futures
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from loguru import logger

from src.utils.image import encode_image


def get_duration(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    cap.release()
    return duration


def extract_frame_at_timestamp(video_path: str, timestamp: float) -> Optional[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ret, frame = cap.read()
    cap.release()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    return None


def compute_frame_similarity(frame1: Any, frame2: Any) -> float:
    if isinstance(frame1, Image.Image):
        frame1 = cv2.cvtColor(np.array(frame1), cv2.COLOR_RGB2BGR)
    if isinstance(frame2, Image.Image):
        frame2 = cv2.cvtColor(np.array(frame2), cv2.COLOR_RGB2BGR)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = np.sum(gray1 != gray2)
    total_pixels = gray1.shape[0] * gray1.shape[1]
    similarity = 1 - (diff / total_pixels)
    return similarity


def find_loading_complete_time(
    video_path: str,
    start_time: float,
    end_time: float,
    video_start_timestamp: float,
    sample_interval: float = 0.1,
    similarity_threshold: float = 0.99,
):
    start_time_relative = start_time - video_start_timestamp
    final_time = max(start_time_relative, end_time - video_start_timestamp - 0.5)
    current_time = max(start_time_relative, final_time)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None
    cap.set(cv2.CAP_PROP_POS_MSEC, final_time * 1000)
    ret, final_frame = cap.read()
    if not ret:
        cap.release()
        return None, None
    while current_time >= start_time_relative:
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        similarity = compute_frame_similarity(frame, final_frame)
        if similarity < similarity_threshold:
            cap.release()
            return current_time + sample_interval, final_time
        current_time -= sample_interval
    cap.release()
    return start_time_relative, final_time


def find_terminate_time(
    video_path: str,
    start_time: float,
    end_time: float,
    video_start_timestamp: float,
    sample_interval: float = 0.1,
    similarity_threshold: float = 0.99,
):
    start_time_relative = start_time - video_start_timestamp
    final_time = max(start_time_relative, end_time - video_start_timestamp)
    if start_time_relative == final_time:
        return start_time_relative, final_time
    current_time = min(start_time_relative, final_time)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time_relative * 1000)
    ret, start_frame = cap.read()
    if not ret:
        cap.release()
        return None, None
    while current_time <= final_time:
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        similarity = compute_frame_similarity(frame, start_frame)
        if similarity < similarity_threshold:
            cap.release()
            return current_time + sample_interval, current_time
        current_time += sample_interval
    cap.release()
    return start_time_relative, final_time


def process_single_directory(basedir: str, episode_dir: str, load_image: bool) -> Optional[Dict[str, Any]]:
    if episode_dir.startswith(".DS_Store"):
        return None
    raw_traj: Dict[str, Any] = {"episode_id": episode_dir}
    task_name_path = os.path.join(basedir, episode_dir, "task_name.json")
    try:
        with open(task_name_path, encoding="utf-8-sig") as f:
            content = f.read().strip()
            if not content:
                return None
            taskname = json.loads(content)["task_name"]
            raw_traj["task_name"] = taskname
    except Exception:
        return None

    metadata_path = os.path.join(basedir, episode_dir, "metadata.json")
    try:
        with open(metadata_path, encoding="utf-8-sig") as f:
            metadata = json.load(f)
            raw_traj["metadata"] = metadata
    except Exception:
        return None

    vis_events_path = os.path.join(basedir, episode_dir, "reduced_events_vis.jsonl")
    complete_events_path = os.path.join(basedir, episode_dir, "reduced_events_complete.jsonl")

    try:
        video_name = [f for f in os.listdir(os.path.join(basedir, episode_dir)) if f.endswith(".mp4")][0]
        video_path = os.path.join(basedir, episode_dir, video_name)
    except Exception:
        return None

    try:
        events: List[Dict[str, Any]] = []
        with open(complete_events_path, encoding="utf-8-sig") as f:
            complete_events = [json.loads(line) for line in f if line.strip()]

        with open(vis_events_path, encoding="utf-8-sig") as f:
            num_lines = sum(1 for _ in f)

            print(f"length of complete_events : {len(complete_events)}")
            print(f"num_lines : {num_lines}")

            if num_lines != len(complete_events):
                print("*"*10 + "Mismatch!!" + "*"*10)
                return None

        last_time_stamp = None
        with open(vis_events_path, encoding="utf-8-sig") as f:
            for index, line in enumerate(f):
                if not line.strip():
                    continue
                event = json.loads(line)
                if event["description"] != complete_events[index]["description"]:
                    event["description"] = complete_events[index]["description"]
                if "\n" in event["description"]:
                    event["description"] = event["description"].split("\n")[0]
                if (
                    "click" in complete_events[index]["action"].lower()
                    or "mouse_press" in complete_events[index]["action"].lower()
                    or "click" in complete_events[index]["description"].lower()
                ) and "(" not in event["description"]:
                    event["description"] = (
                        event["description"]
                        + f" ({complete_events[index]['coordinate']['x']}, {complete_events[index]['coordinate']['y']})"
                    )
                elif "scroll" in complete_events[index]["action"].lower():
                    event["trace"] = complete_events[index]["trace"]

                video_length = get_duration(video_path)
                if complete_events[index]["action"].lower() in ["click", "mouse_press", "drag"] and "pre_move" in complete_events[index]:
                    timestamp, _ = find_loading_complete_time(
                        video_path,
                        start_time=complete_events[index]["pre_move"]["start_time"],
                        end_time=event["start_time"],
                        video_start_timestamp=raw_traj["metadata"]["video_start_timestamp"],
                    )
                else:
                    timestamp = max(0.01, event["start_time"] - raw_traj["metadata"]["video_start_timestamp"] - 0.5)

                if timestamp >= video_length:
                    timestamp = max(0.0, video_length - 0.01)

                try:
                    frame = extract_frame_at_timestamp(video_path, timestamp)
                    if frame and load_image:
                        event["frame"] = f"data:image/png;base64,{encode_image(frame)}"
                    else:
                        event["frame"] = None
                except Exception:
                    event["frame"] = None

                last_time_stamp = event["end_time"]
                event["axtree"] = None
                events.append(event)

        video_length = get_duration(video_path)
        if last_time_stamp is not None:
            if video_length + raw_traj["metadata"]["video_start_timestamp"] > last_time_stamp:
                terminate_time_stamp, _ = find_terminate_time(
                    video_path,
                    start_time=last_time_stamp,
                    end_time=video_length + raw_traj["metadata"]["video_start_timestamp"],
                    video_start_timestamp=raw_traj["metadata"]["video_start_timestamp"],
                )
            else:
                terminate_time_stamp = max(0.0, video_length - 0.01)
            try:
                terminate_frame = extract_frame_at_timestamp(video_path, terminate_time_stamp)
                terminate_event = {
                    "action": "terminate",
                    "description": "terminate the task",
                    "end_time": terminate_time_stamp,
                    "id": len(events),
                    "start_time": terminate_time_stamp,
                    "target": None,
                    "time_stamp": terminate_time_stamp,
                    "frame": f"data:image/png;base64,{encode_image(terminate_frame)}" if (terminate_frame and load_image) else None,
                    "axtree": None,
                }
                events.append(terminate_event)
            except Exception:
                pass

        raw_traj["events"] = events
        if len(events) == 0:
            return None
        return raw_traj
    except Exception as e:
        print(e)
        return None


def get_raw_examples(basedir: str, num_samples: int = -1, load_image: bool = True) -> List[Dict[str, Any]]:
    directories = os.listdir(basedir)
    if num_samples != -1:
        directories = directories[:num_samples]
    process_dir = lambda d: process_single_directory(basedir, d, load_image)
    raw: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = list(tqdm(executor.map(process_dir, directories), total=len(directories), desc="Processing directories"))
        print("="*10 + "futures" + "="*10)
        print(len(futures))
        raw = [result for result in futures if result is not None]
    return raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_raw", type=str, help="Output file path for raw samples (.json or directory)")
    parser.add_argument("--num_samples", "-n", type=int, default=-1, help="Number of samples to extract (-1=all)")
    parser.add_argument("--raw_dir", type=str, default="datasets/raw", help="Directory containing per-episode raw folders")
    parser.add_argument("--no-image", action="store_true", help="Do not embed base64 images in events")
    args = parser.parse_args()

    os.makedirs(os.path.join("datasets"), exist_ok=True)
    raw_examples = get_raw_examples(args.raw_dir, num_samples=args.num_samples, load_image=not args.no_image)

    print(f"sample_raw : {args.sample_raw}")
    print(f"raw_dir : {args.raw_dir}")

    if args.sample_raw.endswith(".json"):
        with open(args.sample_raw, "w", encoding="utf-8") as f:
            json.dump(list(raw_examples), f, indent=2)
    else:
        Path(args.sample_raw).mkdir(parents=True, exist_ok=True)
        fetched_raw = {file.split(".json")[0] for file in os.listdir(args.sample_raw)}
        for raw_example in raw_examples:
            episode_id = raw_example["episode_id"]
            if episode_id in fetched_raw:
                continue
            with open(os.path.join(args.sample_raw, f"{episode_id}.json"), "w", encoding="utf-8") as f:
                json.dump(raw_example, f, indent=2)


if __name__ == "__main__":
    main()


