import cv2
import math

cap = cv2.VideoCapture('C:/Users/pyeonmu/Desktop/test/data/111.mp4')
tracker = cv2.TrackerKCF_create()

# Select initial bounding box for the player
ret, frame = cap.read()
bbox = cv2.selectROI(frame, False)



# Initialize the tracker
tracker.init(frame, bbox)

# Initialize variables for tracking speed
prev_pos = (bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2)
prev_time = 0

while True:
    ret, frame = cap.read()

    # Update the tracker and get the new bounding box
    success, bbox = tracker.update(frame)

    if success:
        # Draw the bounding box around the player
        (x, y, w, h) = [int(i) for i in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Calculate the current position of the player
        current_pos = (x + w/2, y + h/2)

        # Calculate the Euclidean distance between the current and previous positions
        distance = math.sqrt((current_pos[0] - prev_pos[0])**2 + (current_pos[1] - prev_pos[1])**2)

        # Calculate the time elapsed since the previous frame
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
        time_elapsed = (current_time - prev_time) / 1000

        # Calculate the speed of the player in pixels per second
        speed = distance / time_elapsed

        # Display the speed on the frame
        cv2.putText(frame, "Speed: {:.2f} pixels/sec".format(speed), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Update the previous position and time
        prev_pos = current_pos
        prev_time = current_time
    else:
        # Player not found, break out of loop
        break

    # Display the frame with the player bounding box and speed
    cv2.imshow('Player Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
