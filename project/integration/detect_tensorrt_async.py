    #!/usr/bin/env python3
"""
detect_tensorrt_async.py

- Multi-GPU (one worker per GPU) video detection
- Each worker uses an async pipeline (decoder thread -> inference in batches -> writer thread)
- Uses TensorRT engine if found (yolov8n.engine), otherwise falls back to yolov8n.pt
- Writes per-process CSVs and a final merged CSV
"""

import os
import sys
import math
import time
import queue
import threading
import multiprocessing as mp
from multiprocessing import Manager
from pathlib import Path

import cv2
import pandas as pd
import torch
from ultralytics import YOLO

# ---------------- CONFIG ----------------
VIDEO_FOLDER = "./downloads"
OUTPUT_FOLDER = "./output_videos"
WORK_FOLDER = "./workers"
DETECTIONS_CSV = "detections.csv"
SUMMARY_CSV = "file_summary.csv"

YOLO_PT = "yolov8n.pt"
YOLO_ENGINE = "yolov8n.engine"  # put your TensorRT engine here if available
IMGSZ = 480                 # input size for model (longest side)
FPS_TO_PROCESS = 4          # frames per second to sample
BATCH_SIZE = 64             # batch size for inference (tune up for A100)
CONF_THRESH = 0.25          # minimum conf to keep
QUEUE_MAXSIZE = 256         # max frames in decode->inference queue
WRITER_QUEUE_MAXSIZE = 512  # max items in writer queue

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(WORK_FOLDER, exist_ok=True)

# ---------------- Helpers ----------------
def list_videos(folder):
    exts = (".mp4", ".avi", ".mov", ".mkv")
    return sorted([str(p) for p in Path(folder).iterdir() if p.suffix.lower() in exts])

def chunk_list(lst, n_chunks):
    # split list into n_chunks (nearly even)
    return [lst[i::n_chunks] for i in range(n_chunks)]

# ---------------- Worker Process ----------------
def worker_process(gpu_id, videos, return_dict):
    """
    Worker runs on a single GPU (gpu_id).
    It creates a decode thread and a writer thread; inference happens in the main thread in batches.
    return_dict is a multiprocessing.Manager().list-like or dict where we append results.
    """
    # pin process to GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = f"cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[W{gpu_id}] Starting worker on device {device} with {len(videos)} videos")

    # load model (prefer TensorRT engine)
    model_path = YOLO_ENGINE if Path(YOLO_ENGINE).exists() else YOLO_PT
    try:
        model = YOLO(model_path)
        model.to(device)
        # convert to half if CUDA available and engine not already FP16
        if torch.cuda.is_available():
            try:
                model.model.half()
            except Exception:
                pass
        print(f"[W{gpu_id}] Loaded model: {model_path}")
    except Exception as e:
        print(f"[W{gpu_id}] ERROR loading model {model_path}: {e}")
        return

    # thread-safe queues
    decode_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)    # holds tuples (video_file, frame_number, orig_frame, resized_frame)
    writer_queue = queue.Queue(maxsize=WRITER_QUEUE_MAXSIZE)  # holds tuples (video_file, frame_number, annotated_frame)
    detections_rows = []  # local list, will push to return_dict at end
    file_summaries = []

    # decoder thread: reads frames, samples interval, pushes to decode_queue
    def decoder_thread_fn(video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                file_summaries.append({"video": os.path.basename(video_path), "status": "cannot_open"})
                return
            orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            interval = max(1, int(round(orig_fps / FPS_TO_PROCESS)))
            frame_idx = 0
            pushed = 0
            # read frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if (frame_idx % interval) != 0:
                    continue
                # prepare resized frame for inference (keep aspect ratio)
                h, w = frame.shape[:2]
                scale = IMGSZ / max(h, w)
                if scale < 1.0:
                    new_w, new_h = int(w * scale), int(h * scale)
                    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                else:
                    resized = frame
                # push to queue
                while True:
                    try:
                        decode_queue.put((video_path, frame_idx, frame, resized), timeout=2)
                        pushed += 1
                        break
                    except queue.Full:
                        continue
            cap.release()
            file_summaries.append({"video": os.path.basename(video_path), "status": "decoded", "pushed_frames": pushed})
        except Exception as e:
            file_summaries.append({"video": os.path.basename(video_path), "status": "decode_error", "message": str(e)})

    # writer thread: consume annotated frames and write output video per source video
    def writer_thread_fn():
        writers = {}  # map video_filename -> (VideoWriter, fps)
        frames_written = {}  # counters
        while True:
            item = writer_queue.get()
            if item is None:
                break
            video_file, frame_num, annotated = item
            base = os.path.basename(video_file)
            out_name = f"detected_{Path(base).stem}.mp4"
            out_path = os.path.join(OUTPUT_FOLDER, out_name)
            # lazy create writer with annotated frame size
            if out_name not in writers:
                h, w = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, FPS_TO_PROCESS, (w, h))
                writers[out_name] = writer
                frames_written[out_name] = 0
            writers[out_name].write(annotated)
            frames_written[out_name] += 1
        # release writers and produce summary rows
        for out_name, writer in writers.items():
            writer.release()
            file_summaries.append({"video": out_name, "status": "written", "frames": frames_written.get(out_name, 0)})

    # start writer thread
    writer_thread = threading.Thread(target=writer_thread_fn, daemon=True)
    writer_thread.start()

    # Process each video: start a decoder thread per video, but ensure decode_queue consumed by main loop
    for v in videos:
        # spawn decoder thread, wait for it to finish decoding this video, while consuming decode_queue
        dec_thread = threading.Thread(target=decoder_thread_fn, args=(v,), daemon=True)
        dec_thread.start()

        # inference loop for this video's frames: consume decode_queue, batch and run inference
        local_detections = 0
        batch_frames = []
        batch_meta = []  # list of tuples (video_path, frame_num, orig_frame)
        while True:
            # exit condition: decoder thread finished and queue empty
            if not dec_thread.is_alive() and decode_queue.empty():
                # process leftover batch then break
                if batch_frames:
                    results = None
                    try:
                        # run batch inference (Ultralytics accepts list of frames)
                        if torch.cuda.is_available():
                            with torch.cuda.amp.autocast():
                                results = model(batch_frames, imgsz=IMGSZ, conf=CONF_THRESH)
                        else:
                            results = model(batch_frames, imgsz=IMGSZ, conf=CONF_THRESH)
                    except Exception as e:
                        print(f"[W{gpu_id}] inference exception: {e}")
                        results = [None] * len(batch_frames)
                    # handle results
                    for res_idx, res in enumerate(results):
                        meta = batch_meta[res_idx]
                        vid_path, fnum, orig_frame = meta
                        if res is None or getattr(res, "boxes", None) is None or len(res.boxes) == 0:
                            continue
                        # draw and push writer item
                        annotated = res.plot() if hasattr(res, "plot") else orig_frame
                        try:
                            writer_queue.put((vid_path, fnum, annotated), timeout=5)
                        except queue.Full:
                            # drop frame if writer overloaded
                            pass
                        # record detections
                        for box, conf, cls in zip(res.boxes.xyxy, res.boxes.conf, res.boxes.cls):
                            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                            try:
                                obj = model.names[int(cls)]
                            except Exception:
                                obj = str(int(cls))
                            detections_rows.append({
                                "video": os.path.basename(vid_path),
                                "output_video": f"detected_{Path(os.path.basename(vid_path)).stem}.mp4",
                                "frame": int(fnum),
                                "object": obj,
                                "confidence": float(conf),
                                "x1": x1, "y1": y1, "x2": x2, "y2": y2
                            })
                            local_detections += 1
                    batch_frames = []
                    batch_meta = []
                break

            # consume one item from decode_queue
            try:
                item = decode_queue.get(timeout=2)
            except queue.Empty:
                continue
            vid_path, fnum, orig_frame, resized = item
            batch_frames.append(resized)
            batch_meta.append((vid_path, fnum, orig_frame))

            if len(batch_frames) >= BATCH_SIZE:
                # run inference on batch_frames
                try:
                    if torch.cuda.is_available():
                        with torch.cuda.amp.autocast():
                            results = model(batch_frames, imgsz=IMGSZ, conf=CONF_THRESH)
                    else:
                        results = model(batch_frames, imgsz=IMGSZ, conf=CONF_THRESH)
                except Exception as e:
                    print(f"[W{gpu_id}] inference exception: {e}")
                    results = [None] * len(batch_frames)
                # handle results
                for res_idx, res in enumerate(results):
                    vid_path, fnum, orig_frame = batch_meta[res_idx]
                    if res is None or getattr(res, "boxes", None) is None or len(res.boxes) == 0:
                        continue
                    annotated = res.plot() if hasattr(res, "plot") else orig_frame
                    try:
                        writer_queue.put((vid_path, fnum, annotated), timeout=5)
                    except queue.Full:
                        pass
                    # record detections
                    for box, conf, cls in zip(res.boxes.xyxy, res.boxes.conf, res.boxes.cls):
                        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                        try:
                            obj = model.names[int(cls)]
                        except Exception:
                            obj = str(int(cls))
                        detections_rows.append({
                            "video": os.path.basename(vid_path),
                            "output_video": f"detected_{Path(os.path.basename(vid_path)).stem}.mp4",
                            "frame": int(fnum),
                            "object": obj,
                            "confidence": float(conf),
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2
                        })
                        local_detections += 1
                batch_frames = []
                batch_meta = []

        # decoder for this video finished and all frames processed
        print(f"[W{gpu_id}] Done video {os.path.basename(v)}: detections={local_detections}")
        file_summaries.append({"video": os.path.basename(v), "detections": local_detections})

    # signal writer thread to finish
    writer_queue.put(None)
    writer_thread.join()

    # put results into shared return_dict (multiprocessing.Manager list/dict)
    return_dict[f"worker_{gpu_id}_detections"] = detections_rows
    return_dict[f"worker_{gpu_id}_summary"] = file_summaries
    print(f"[W{gpu_id}] Worker finished. detections_rows={len(detections_rows)}")
    return

# ---------------- Main ----------------
def main():
    mp.set_start_method("spawn", force=True)
    video_files = list_videos(VIDEO_FOLDER)
    if not video_files:
        print("No videos found. Exiting.")
        return

    ngpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    ngpus = max(1, ngpus)
    print(f"Found {ngpus} GPUs (torch.cuda.is_available={torch.cuda.is_available()})")
    shards = chunk_list(video_files, ngpus)

    mgr = Manager()
    return_dict = mgr.dict()

    procs = []
    for gpu_id in range(ngpus):
        vids = shards[gpu_id]
        if not vids:
            # skip empty shard
            continue
        p = mp.Process(target=worker_process, args=(gpu_id, vids, return_dict), daemon=False)
        p.start()
        procs.append(p)

    # wait
    for p in procs:
        p.join()

    # merge results
    all_rows = []
    all_summaries = []
    for k, v in return_dict.items():
        if k.endswith("_detections"):
            all_rows.extend(v)
        elif k.endswith("_summary"):
            all_summaries.extend(v)

    # save merged csvs
    if all_rows:
        pd.DataFrame(all_rows).to_csv(DETECTIONS_CSV, index=False)
    else:
        pd.DataFrame(columns=["video","output_video","frame","object","confidence","x1","y1","x2","y2"]).to_csv(DETECTIONS_CSV, index=False)

    pd.DataFrame(all_summaries).to_csv(SUMMARY_CSV, index=False)
    print(f"Saved detections -> {DETECTIONS_CSV}, summary -> {SUMMARY_CSV}")

if __name__ == "__main__":
    t_start = time.time()
    main()
    t_end = time.time()
    total_time = t_end - t_start
    print(f"✅ Total processing time: {total_time:.2f} sec for all videos") 