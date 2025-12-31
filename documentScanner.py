import cv2
import numpy as np
import utilsScanner
import os

########################################################################
webCamFeed = False
pathImage = "UserForms/test1.png"
cap = cv2.VideoCapture(0)
cap.set(10, 160)
heightImg = 640
widthImg = 480
########################################################################

if not os.path.exists("Scanned"):
    os.makedirs("Scanned")

utilsScanner.initializeTrackbars()
count = 0


def preprocess_image(img):
    # Tiền xử lý ảnh để cải thiện việc phát hiện biên
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE cải thiện độ tương phản
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_gray = clahe.apply(img_gray)

    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)

    return img_gray, img_blur


def enhance_document_edges(img_threshold):
    # Tăng cường các cạnh của tài liệu
    kernel = np.ones((5, 5), np.uint8)

    # Đóng các khoảng trống nhỏ trong edges
    img_closing = cv2.morphologyEx(img_threshold, cv2.MORPH_CLOSE, kernel)

    # Giãn nở để làm dày edges
    img_dilate = cv2.dilate(img_closing, kernel, iterations=2)

    # Ăn mòn để làm mỏng edges
    img_erode = cv2.erode(img_dilate, kernel, iterations=1)

    return img_erode


try:
    while True:
        if webCamFeed:
            success, img = cap.read()
            if not success:
                print("Không thể đọc frame từ webcam")
                break
        else:
            img = cv2.imread(pathImage)
            if img is None:
                print(f"Không thể đọc ảnh từ {pathImage}")
                break

        # Resize ảnh
        img = cv2.resize(img, (widthImg, heightImg))
        img_blank = np.zeros((heightImg, widthImg, 3), np.uint8)

        # Tiền xử lý ảnh
        img_gray, img_blur = preprocess_image(img)

        # Lấy giá trị threshold từ trackbar
        thres = utilsScanner.valTrackbars()

        # Phát hiện biên với Canny
        img_threshold = cv2.Canny(img_blur, thres[0], thres[1])

        # Tăng cường edges
        img_threshold = enhance_document_edges(img_threshold)

        # Tìm contours
        img_contours = img.copy()
        img_big_contour = img.copy()

        contours, hierarchy = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Chỉ vẽ contours nếu tìm thấy
        if len(contours) > 0:
            cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)

        # Tìm contour lớn nhất (tài liệu)
        biggest, max_area = utilsScanner.biggestContour(contours)

        if biggest.size != 0:
            biggest = utilsScanner.reorder(biggest)

            img_big_contour = utilsScanner.drawRectangle(img_big_contour, biggest, 2)

            # Thực hiện perspective transform
            pts1 = np.float32(biggest)
            pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            img_warp_colored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

            # Cắt bớt viền
            img_warp_colored = img_warp_colored[5:img_warp_colored.shape[0] - 5,
                               5:img_warp_colored.shape[1] - 5]
            img_warp_colored = cv2.resize(img_warp_colored, (widthImg, heightImg))

            # Xử lý ảnh scan để có văn bản rõ ràng
            img_warp_gray = cv2.cvtColor(img_warp_colored, cv2.COLOR_BGR2GRAY)

            # Adaptive threshold với các tham số được điều chỉnh
            img_adaptive_thresh = cv2.adaptiveThreshold(
                img_warp_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            img_adaptive_thresh = cv2.bitwise_not(img_adaptive_thresh)
            img_adaptive_thresh = cv2.medianBlur(img_adaptive_thresh, 3)

            # Mảng hiển thị
            imageArray = ([img, img_gray, img_threshold, img_contours],
                          [img_big_contour, img_warp_colored, img_warp_gray, img_adaptive_thresh])
        else:
            imageArray = ([img, img_gray, img_threshold, img_contours],
                          [img_blank, img_blank, img_blank, img_blank])

        # Hiển thị kết quả
        labels = [["Original", "Gray", "Threshold", "Contours"],
                  ["Biggest Contour", "Warp Perspective", "Warp Gray", "Adaptive Threshold"]]

        stacked_image = utilsScanner.stackImages(imageArray, 0.62, labels)
        cv2.imshow("Document Scanner", stacked_image)

        # Xử lý sự kiện bàn phím
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            if 'img_warp_colored' in locals():
                filename = f"Scanned/myImage{count}.png"
                cv2.imwrite(filename, img_warp_colored)
                print(f"Đã lưu ảnh: {filename}")

                # Hiển thị thông báo
                cv2.rectangle(stacked_image,
                              (int(stacked_image.shape[1] / 2) - 200, int(stacked_image.shape[0] / 2) - 50),
                              (int(stacked_image.shape[1] / 2) + 200, int(stacked_image.shape[0] / 2) + 50),
                              (0, 255, 0), cv2.FILLED)
                cv2.putText(stacked_image, "Scan Saved!",
                            (int(stacked_image.shape[1] / 2) - 150, int(stacked_image.shape[0] / 2) + 15),
                            cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('Document Scanner', stacked_image)
                cv2.waitKey(1000)
                count += 1
            else:
                print("Không thể lưu: Chưa tìm thấy tài liệu!")

        elif key == ord('q'):
            break

except Exception as e:
    print(f"Lỗi: {e}")

finally:
    if webCamFeed:
        cap.release()
    cv2.destroyAllWindows()