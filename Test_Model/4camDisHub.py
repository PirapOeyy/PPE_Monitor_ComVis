import argparse
import json
from typing import Dict
import cv2
import numpy as np
import time
from tqdm import tqdm
from ultralytics import YOLO
import supervision as sv
import torch   
import threading
import GPUtil     
from connection.connection_minio import connection_minio
from connection.connection_mqtt import connection_mqtt
from controllers.DetectionController import store_detection
from PIL import Image
import pandas as pd
from datetime import datetime
import io
import sys
import concurrent.futures
import os

count_data = []
fps_data = [] 
average_data = []

# Global GPU performance lists
gpu_loads = []
gpu_memory_usages = []
gpu_temperatures = []


minio_client, bucket_name, folder_name = connection_minio()
mqtt_client, topic = connection_mqtt()

def log_gpu_performance():
    while True:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            gpu_loads.append(gpu.load * 100)  # Load as percentage
            gpu_memory_usages.append(gpu.memoryUsed)  # Memory in MB
            gpu_temperatures.append(gpu.temperature)  # Temperature in Celsius
        time.sleep(1)  # Log every second
        
def get_minio_object_url(minio_client, bucket_name, object_name):
    url = minio_client.presigned_get_object(bucket_name, object_name)
    return url

def save_frame_to_minio(frame, count, folder_name):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_byte_arr = io.BytesIO()
    img_pil.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    folder_name = folder_name.rstrip('/')
    minio_object_name = f'{folder_name}/frame_{count}.jpg'

    minio_client.put_object(
        bucket_name,
        minio_object_name,
        img_byte_arr,
        len(img_byte_arr.getvalue()),
        content_type="image/jpeg"
    )

    return minio_object_name


CLASS_NAMES = {
    0: 'AUP',
    1: 'BB',
    2: 'BS',
    3: 'KTB',
    4: 'MP',
    5: 'SM',
    6: 'TB',
    7: 'TR',
    8: 'TS'
}

def save_frame_to_minio_async(frame, count, folder_name):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(save_frame_to_minio, frame, count, folder_name)
        return future.result()

def intersects_line(bbox, p1, p2):
    """Check if a bounding box intersects with a line segment."""
    x1, y1, x2, y2 = bbox
    x3, y3 = p1
    x4, y4 = p2

    # Check if the bounding box and line segment intersect
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def intersect(A, B, C, D):
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    return intersect((x1, y1), (x2, y2), p1, p2)

class DetectionsManager:
    def __init__(self, roi_data) -> None:
        self.tracker_id_to_zone_id: Dict[int, int] = {}
        self.counts: Dict[str, Dict[str, Dict[str, int]]] = {
            "AUP": {zone: {direction: 0 for direction in roi_data} for zone in roi_data},
            "BB": {zone: {direction: 0 for direction in roi_data} for zone in roi_data},
            "BS": {zone: {direction: 0 for direction in roi_data} for zone in roi_data},
            "KTB": {zone: {direction: 0 for direction in roi_data} for zone in roi_data},
            "MP": {zone: {direction: 0 for direction in roi_data} for zone in roi_data},
            "SM": {zone: {direction: 0 for direction in roi_data} for zone in roi_data},
            "TB": {zone: {direction: 0 for direction in roi_data} for zone in roi_data},
            "TR": {zone: {direction: 0 for direction in roi_data} for zone in roi_data},
            "TS": {zone: {direction: 0 for direction in roi_data} for zone in roi_data}
        }
        self.roi_data = roi_data
        self.confidence_scores: Dict[str, list] = {class_name: [] for class_name in CLASS_NAMES.values()}
    def update(self, detections_all: sv.Detections, id_simpang: int) -> sv.Detections:
        movement_data =[]
        for i, (class_id, tracker_id,confidence) in enumerate(zip(detections_all.class_id, detections_all.tracker_id, detections_all.confidence)):
            class_name = CLASS_NAMES[class_id]
            self.confidence_scores[class_name].append(confidence)
            if tracker_id in self.tracker_id_to_zone_id:
                prev_zone = self.tracker_id_to_zone_id[tracker_id]
                for direction, (p1, p2) in roi_data.items():
                    # Check if the vehicle crosses the line in the current direction
                    if intersects_line(detections_all.xyxy[i], p1, p2):  # Using xyxy for bounding box
                        if prev_zone != direction:  # Avoid counting when vehicle returns to the same zone
                            self.counts[class_name][prev_zone][direction] += 1
                            movement_data.append({
                                "class_name": class_name,
                                "dari_arah": prev_zone,
                                "ke_arah": direction
                            })
                            origin = self.tracker_origin.get(tracker_id, prev_zone)
                            store_detection(
                                id_simpang, "P", class_name, direction,
                                datetime.utcnow().isoformat(),
                                update_direction=True,
                                base_direction=origin  
                            )

                        self.tracker_id_to_zone_id[tracker_id] = direction
                        break
            else:
                for direction, (p1, p2) in roi_data.items():
                    # Check if the vehicle starts from a zone
                    if intersects_line(detections_all.xyxy[i], p1, p2):  # Using xyxy for bounding box
                        self.tracker_id_to_zone_id[tracker_id] = direction
                        store_detection(
                            id_simpang, "P", class_name, direction, 
                            datetime.utcnow().isoformat(),
                            update_direction=False  
                        )
                        
                        movement_data.append({
                            "class_name": class_name,
                            "dari_arah": direction, 
                            "ke_arah": None
                        })
                        break
        return detections_all
    def calculate_average_confidence(self) -> Dict[str, float]:
        # Calculate average confidence per class
        average_confidence = {}
        for class_name, scores in self.confidence_scores.items():
            if scores:
                average_confidence[class_name] = sum(scores) / len(scores)
            else:
                average_confidence[class_name] = 0.0  # No detections, so average is 0
        return average_confidence

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B",  "#3C76D1"])
# COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])


def load_roi(file_path):
    try:
        roi_data = {}
        with open(file_path, "r") as f:
            for line in f:
                if line.strip(): 
                    key, value = line.strip().split(":", 1) 
                    roi_data[key.strip().strip('"')] = eval(value.strip().strip(','))
        return roi_data
    except Exception as e:
        print(f"Error loading ROI file {file_path}: {e}")
        return {}
    
ROI_COLORS = {
    2: { #Prambanan
        "east": (255, 0, 0),  # Red
        "south": (0, 0, 255),  # Blue
        "west": (0, 255, 0)  # Green
    },
    3: {  # Demen Glagah
        "east": (0, 255, 0),  # Green
        "south": (0, 0, 255),  # Blue
        "west": (255, 0, 0)  # Red
    },
    4: {  # Piyungan
        "north": (0, 255, 0),  # Green
        "east": (0, 255, 255),  # Cyan
        "west": (0, 0, 139)  # Dark Blue
    },
    5: {  # Tempel
        "east": (0, 255, 0),  # Green
        "south": (0, 0, 255),  # Blue
        "north": (0, 255, 255)  # Cyan
    }
}

class VideoProcessor:
    def __init__(
        self,
        source_weights_path: str,
        source_video_path: str,
        roi_path: str,
        target_video_path: str = None,
        confidence_threshold: float = 0.7,
        iou_threshold: float = 0.7,
        id_simpang: int = 0
    ) -> None:
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.source_video_path = source_video_path
        self.roi_data = load_roi(roi_path)
        self.target_video_path = target_video_path
        self.id_simpang = id_simpang

        # Check for GPU availability and load the model to the GPU if possible
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(source_weights_path).to(device)

        self.roi_data = load_roi(roi_path)

        self.tracker = sv.ByteTrack()

        self.video_info = sv.VideoInfo.from_video_path(source_video_path)

        self.bounding_box_annotator = sv.BoundingBoxAnnotator(color=COLORS)
        self.label_annotator = sv.LabelAnnotator(
            color=COLORS, text_color=sv.Color.BLACK
        )
        self.trace_annotator = sv.TraceAnnotator(
            color=COLORS, position=sv.Position.CENTER, trace_length=100, thickness=2
        )
        self.detections_manager = DetectionsManager(self.roi_data)

        self.frame_confidences = []
    
    def upload_frame_and_send_mqtt(self, frame, frame_index, detections, fps, timestamp):
        try:
            # Upload frame to Minio
            minio_object_name = f'{folder_name}/frame_{frame_index}.jpg'
            _, buffer = cv2.imencode('.jpg', frame)
            io_buf = io.BytesIO(buffer)
            
            minio_client.put_object(
                bucket_name,
                minio_object_name,
                io_buf,
                len(buffer),
                content_type="image/jpeg"
            )
            
            # Get presigned URL
            url = minio_client.presigned_get_object(bucket_name, minio_object_name)
            
            # Prepare detection data for MQTT
            detection_data = {
                "id_simpang": self.id_simpang,
                "count": frame_index,
                "timestamp": datetime.utcnow().isoformat(),
                "fps": round(fps, 2),
                "image_url": url,
                "detections": []
            }
            
            # Extract movement data from detections
            if hasattr(detections, 'class_id') and len(detections.class_id) > 0:
                class_counts = {}
                for class_id in detections.class_id:
                    class_name = CLASS_NAMES[class_id]
                    if class_name in class_counts:
                        class_counts[class_name] += 1
                    else:
                        class_counts[class_name] = 1
                    
                    # Add to detections list - note this is simplified
                    # In the original code, it tracked movements between regions
                    detection_data["detections"].append({
                        "class": class_name
                    })
            
            # Send MQTT message
            mqtt_client.publish(topic, json.dumps(detection_data))
            
            # Cleanup old frames (keep only the most recent 30)
            self.cleanup_old_frames()
            
        except Exception as e:
            print(f"Error in upload_frame_and_send_mqtt: {e}")
    
    def cleanup_old_frames(self):
        try:
            objects = minio_client.list_objects(
                bucket_name,
                prefix=folder_name,
                recursive=True
            )
            
            sorted_objects = sorted(objects, key=lambda x: x.last_modified)
            
            if len(sorted_objects) > 30:
                delete_count = len(sorted_objects) - 30
                for i in range(delete_count):
                    minio_client.remove_object(bucket_name, sorted_objects[i].object_name)
        except Exception as e:
            print(f"Cleanup error: {e}")

    def process_video(self):
        # For RTSP streams, use cv2.VideoCapture directly instead of sv.get_video_frames_generator
        if self.source_video_path.startswith('rtsp://'):
            print("Connecting to RTSP stream...")
            cap = cv2.VideoCapture(self.source_video_path)
            # Set buffer size and timeouts for RTSP
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Reduce buffer size
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|reconnect;1|reconnect_delay_max;2000"
            
            if not cap.isOpened():
                print(f"Error: Unable to open video source {self.source_video_path}")
                return
                
            frame_index = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to receive frame. Attempting to reconnect...")
                    time.sleep(1)
                    # Reopen the connection
                    cap.release()
                    cap = cv2.VideoCapture(self.source_video_path)
                    continue
                
                start_time = time.time()
                annotated_frame, frame_confidence_summary, detections = self.process_frame(frame, frame_index)
                
                self.frame_confidences.append(frame_confidence_summary)
                
                end_time = time.time()
                fps = 1 / (end_time - start_time)
                timestamp = frame_index / (cap.get(cv2.CAP_PROP_FPS) or 25.0)  # Default to 25 fps if not available
                
                self.upload_frame_and_send_mqtt(annotated_frame, frame_index, detections, fps, timestamp)
                
                cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                if self.target_video_path:
                    # Write to video file (you'd need to initialize a VideoWriter)
                    pass
                
                cv2.imshow("Processed Video", annotated_frame)
                frame_index += 1
                
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                
            cap.release()
            cv2.destroyAllWindows()
        else:
            frame_generator = sv.get_video_frames_generator(
                source_path=self.source_video_path
            )
            for frame_index, frame in enumerate(tqdm(frame_generator, total=self.video_info.total_frames)):
                start_time = time.time()
                annotated_frame, frame_confidence_summary, detections = self.process_frame(frame, frame_index)
                
                self.frame_confidences.append(frame_confidence_summary)
                
                # Calculate FPS
                end_time = time.time()
                fps = 1 / (end_time - start_time)
                timestamp = frame_index / self.video_info.fps
                fps_data.append([round(fps,2),round(timestamp,2)])
                self.upload_frame_and_send_mqtt(annotated_frame, frame_index, detections, fps, timestamp)

                # print(f"FPS: {fps:.2f},timestamps: {timestamp:.2f}")
                # Display FPS on the frame
                cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                cv2.imshow("Processed Video", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()
        
    def draw_cardinal_lines(self, frame: np.ndarray) -> np.ndarray:
        scale_factor = 0.8 
        colors = ROI_COLORS.get(self.id_simpang, {})
        for direction, (p1, p2) in self.roi_data.items():
            p1_scaled = (int(p1[0] * scale_factor), int(p1[1] * scale_factor))
            p2_scaled = (int(p2[0] * scale_factor), int(p2[1] * scale_factor))
            cv2.line(frame, p1_scaled, p2_scaled, colors.get(direction, (255, 255, 255)), 2)
        return frame
        
    def annotate_frame(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        annotated_frame = frame.copy()
        
        # Draw cardinal lines
        annotated_frame = self.draw_cardinal_lines(annotated_frame)

    
        labels = [f"{CLASS_NAMES[class_id]} #{tracker_id} {confidence:.2f}" for tracker_id, class_id, confidence in zip(detections.tracker_id, detections.class_id, detections.confidence)]
        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.bounding_box_annotator.annotate(
            annotated_frame, detections
        )
        annotated_frame = self.label_annotator.annotate(
            annotated_frame, detections, labels
        )

        return annotated_frame


    def process_frame(self, frame: np.ndarray, frame_index: int) -> (np.ndarray, dict, sv.Detections):
        results = self.model(
            frame, verbose=False, conf=self.conf_threshold, iou=self.iou_threshold
        )[0]
        total_count = len(results.boxes)
        count_data.append(total_count)
        # print(total_count)
        detections = sv.Detections.from_ultralytics(results)
        detections = self.tracker.update_with_detections(detections)

        detections = self.detections_manager.update(
            detections,
            self.id_simpang
        )
        frame_confidence = {class_name: [] for class_name in CLASS_NAMES.values()}
        for class_id, confidence in zip(detections.class_id, detections.confidence):
            class_name = CLASS_NAMES[class_id]
            frame_confidence[class_name].append(confidence)
        
        average_class_confidence = {
            class_name: (sum(confidences) / len(confidences) if confidences else 0.0)
            for class_name, confidences in frame_confidence.items()
        }
        
        all_confidences = [conf for confidences in frame_confidence.values() for conf in confidences]
        average_all_classes = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
        average_data.append(average_all_classes)
        frame_confidence_summary = {
            "frame": frame_index + 1,
            "confidence": average_class_confidence,
            "average_all_classes": average_all_classes
        }
        return self.annotate_frame(frame, detections), frame_confidence_summary
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Traffic Flow Analysis with YOLO and ByteTrack"
    )

    parser.add_argument(
        "--source_weights_path",
        required=True,
        help="Path to the source weights file",
        type=str,
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--target_video_path",
        default=None,
        help="Path to the target video file (output)",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.7,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--roi_path",
        required=True,
        help="Path to roi roi_piyungan.txt, roi_demen.txt, roi_prambanan.txt, roi_tempel.txt",
        type=str,
    )
    parser.add_argument(
        "--iou_threshold", default=0.7, help="IOU threshold for the model", type=float
    )

    parser.add_argument(
        "--id_simpang",
        default=0,
        help="ID Simpang (e.g., 2 for Prambanan, 3 for DemenGlagah, 4 for Piyungan, 5 for Tempel)",
        type=int,
    )


    args = parser.parse_args()
        # Start GPU logging thread
    gpu_logging_thread = threading.Thread(target=log_gpu_performance, daemon=True)
    gpu_logging_thread.start()
    
    processor = VideoProcessor(
        source_weights_path=args.source_weights_path,
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
        roi_path=args.roi_path,
        id_simpang=args.id_simpang,

    )
    processor.process_video()

        # Save GPU performance metrics to JSON
    with open("load_gpu.json", "w") as f:
        json.dump(gpu_loads, f, indent=4)
    with open("memory_gpu.json", "w") as f:
        json.dump(gpu_memory_usages, f, indent=4)
    with open("temp_gpu.json", "w") as f:
        json.dump(gpu_temperatures, f, indent=4)