from ultralytics import YOLO
import cv2
import numpy as np
import csv

object_detector = YOLO('yolov8n.pt')

input_video_path = r"C:\Users\prabh\OneDrive\Desktop\IMG_1503.MOV"
video_capture = cv2.VideoCapture(input_video_path)

fps = video_capture.get(cv2.CAP_PROP_FPS)

output_video_path = r"Videos/Output/output.mp4"

region_of_interest_Coordinates = np.array([[22, 154], [1602, 170], [1630, 1066], [42, 1054]], dtype=np.int32)

people_entry_timestamps = {}

time_spent_in_queue = []

csv_filename = "queue_time.csv"

csv_file = open(csv_filename, 'w', newline='')

csv_writer = csv.writer(csv_file)

frame_count = 0

person_ids_present_in_video = []

video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (video_width, video_height))

while video_capture.isOpened():

    success, frame = video_capture.read()

    if success:

        detection_results = object_detector.track(frame, persist=True)

        detected_boxes = detection_results[0].boxes.xyxy.cpu()
        track_ids = detection_results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(detected_boxes, track_ids):
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            if cv2.pointPolygonTest(region_of_interest_Coordinates, (x_center, y_center), False) > 0:
                if str(track_id) not in people_entry_timestamps:
                    people_entry_timestamps[str(track_id)] = str(frame_count)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Person ID: " + str(track_id), (x1, y1 - 5),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (20, 0, 255), 2)

            else:
                if str(track_id) in people_entry_timestamps:
                    exit_timestamp = frame_count
                    entry_timestamp = people_entry_timestamps[str(track_id)]

                    time_spent = (exit_timestamp - int(entry_timestamp)) / fps
                    time_spent_in_queue.append(time_spent)

                    csv_writer.writerow(
                        ["Time spent by person " + str(track_id) + " in line is " + str(time_spent)])
                    person_ids_present_in_video.append(str(track_id))

                    people_entry_timestamps.pop(str(track_id))

        cv2.drawContours(frame, [region_of_interest_Coordinates], -1, (255, 0, 0), 3)

        output_video.write(frame)

        frame_count += 1

    else:
        break

if time_spent_in_queue:
    average = sum(time_spent_in_queue) / len(set(person_ids_present_in_video))
    print("Average of list: ", round(average, 3))

    csv_writer.writerow(["Average time spent in line is " + str(round(average, 3))])

video_capture.release()
output_video.release()
csv_file.close()

print(f"Output video saved at: {output_video_path}"
