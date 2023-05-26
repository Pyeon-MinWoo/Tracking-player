import cv2
import math

cap = cv2.VideoCapture('C:/Users/pyeonmu/Desktop/test/data/111.mp4')
tracker = cv2.TrackerKCF_create() 

# Select initial bounding box for the player
# 플레이어의 초기 경계 상자 선택
ret, frame = cap.read()
bbox = cv2.selectROI(frame, False)



# Initialize the tracker
# 추적기 초기화
tracker.init(frame, bbox)

# Initialize variables for tracking speed
# 추적 속도에 대한 변수 초기화
prev_pos = (bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2)
prev_time = 0

while True:
    ret, frame = cap.read()

    # Update the tracker and get the new bounding box
    # 트래커를 업데이트하고 새 경계 상자 가져오기
    success, bbox = tracker.update(frame)

    if success:
        # Draw the bounding box around the player
        # 플레이어 주위에 경계 상자를 그립니다.
        (x, y, w, h) = [int(i) for i in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Calculate the current position of the player
        # 플레이어의 현재 위치 계산
        current_pos = (x + w/2, y + h/2)

        # Calculate the Euclidean distance between the current and previous positions
        # 현재 위치와 이전 위치 사이의 유클리드 거리 계산
        distance = math.sqrt((current_pos[0] - prev_pos[0])**2 + (current_pos[1] - prev_pos[1])**2)

        # Calculate the time elapsed since the previous frame
        # 이전 프레임 이후 경과된 시간 계산
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
        time_elapsed = (current_time - prev_time) / 1000

        # Calculate the speed of the player in pixels per second
        # 플레이어의 속도를 초당 픽셀 단위로 계산합니다.
        speed = distance / time_elapsed

        # Display the speed on the frame
        # 프레임에 속도 표시
        cv2.putText(frame, "Speed: {:.2f} pixels/sec".format(speed), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Update the previous position and time
        # 이전 위치 및 시간 업데이트
        prev_pos = current_pos
        prev_time = current_time
    else:
        # Player not found, break out of loop
        # 플레이어를 찾을 수 없습니다. 루프에서 벗어납니다.
        break

    # Display the frame with the player bounding box and speed
    # 플레이어 경계 상자 및 속도로 프레임 표시
    cv2.imshow('Player Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()
