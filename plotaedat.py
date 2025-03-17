import aedat
import numpy as np
import matplotlib.pyplot as plt
import cv2
path = "/home/martin-barry/Downloads/processed_dataset/train/run0/ung_78_06/ung_78_06_c0001/dvs.aedat4"
# Initialize the decoder
decoder = aedat.Decoder(path)
print(decoder.id_to_stream())

# Initialize lists for spike visualization
time_stamps = []
event_counts = []

time_window = 100  # Number of packets to display at a time

for i, packet in enumerate(decoder):
    print(packet["stream_id"], end=": ")
    if "events" in packet:
        events =  packet["events"]
        def events_to_img(events):
            img = np.zeros((260, 346), dtype=np.uint8)
            for event in events:
                ts, x, y, p = event
                print(x, y, p, ts)
                img[y, x] = 255
            return img
        img = events_to_img(events)
        cv2.imshow("Spike Plot", img)
        cv2.waitKey(100)

cv2.destroyAllWindows()