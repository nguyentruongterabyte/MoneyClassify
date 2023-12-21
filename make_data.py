import cv2
import os

# Label: 000000 là ko cầm tiền, còn lại là các mệnh giá
label = "005000"

cap = cv2.VideoCapture(0)

# Biến đếm, để chỉ lưu dữ liệu sau khoảng 60 frame, tránh lúc đầu chưa kịp cầm tiền lên
i = 0
while True:
    # Capture frame-by-frame
    #
    i += 1
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.resize(frame, dsize=None, fx=0.3, fy=0.3)

    # Hiển thị
    cv2.imshow('frame', frame)

    # Lưu dữ liệu từ frame 60 đến 1060
    if 60 <= i <= 1260:
        print("Số ảnh capture = ", i-60)
        # Tạo thư mục nếu chưa có
        if not os.path.exists('data/' + str(label)):
            os.mkdir('data/' + str(label))

        cv2.imwrite('data/' + str(label) + "/" + str(i) + ".png", frame)
    elif i > 1260:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
