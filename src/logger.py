import csv
from datetime import datetime, timedelta

csv_file = "face_logs.csv"

detection_tracker = {}
last_logged_times = {}

#csv file
try:
    with open(csv_file, mode='x', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Timestamp"])
except FileExistsError:
    pass

#logging data
def log_to_csv(name):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, timestamp])

def should_log(name, frame_time):
    global detection_tracker, last_logged_times

    # updating detection 
    if name not in detection_tracker:
        detection_tracker[name] = frame_time
    else:
        detection_duration = frame_time - detection_tracker[name]
        if detection_duration.total_seconds() >= 3: 
            if name not in last_logged_times or frame_time - last_logged_times[name] >= timedelta(minutes=30):
                last_logged_times[name] = frame_time
                detection_tracker.pop(name, None)  # reset tracking
                return True

    return False
