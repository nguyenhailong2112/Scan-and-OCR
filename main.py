import cv2
import numpy as np
import pytesseract
import os
import re

# Config
per = 25
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# ROI cho ảnh GỐC 810x1080
roi_original = [[(36, 544), (388, 620), 'text', 'Name'],
                [(428, 544), (768, 622), 'text', 'Phone'],
                [(40, 702), (382, 774), 'text', 'Email'],
                [(422, 702), (768, 772), 'text', 'ID'],
                [(40, 850), (384, 920), 'text', 'Class'],
                [(424, 846), (766, 926), 'text', 'Address']]


def preprocess_for_ocr(img_crop):
    """Tiền xử lý ảnh để cải thiện OCR"""
    if len(img_crop.shape) == 3:
        img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_crop

    # Tăng độ tương phản
    img_gray = cv2.equalizeHist(img_gray)

    # Làm sạch ảnh trước khi threshold
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

    # Adaptive threshold với parameters được tối ưu
    img_thresh = cv2.adaptiveThreshold(
        img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 8
    )

    # Loại bỏ nhiễu với morphological operations
    kernel = np.ones((2, 2), np.uint8)
    img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
    img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)

    return img_thresh


def clean_ocr_text(text, field_type):
    """Làm sạch text OCR với xử lý đặc biệt cho tiếng Việt"""
    # Xóa khoảng trắng thừa
    text_clean = ' '.join(text.split())

    # Xử lý đặc biệt cho từng loại trường
    if field_type == 'Name':
        # Giữ chữ cái, khoảng trắng và ký tự tiếng Việt
        text_clean = re.sub(r'[^a-zA-ZÀ-ỹ\s]', '', text_clean)
        # Sửa lỗi thường gặp trong tiếng Việt
        text_clean = re.sub(r'Nguy n', 'Nguyen', text_clean)
        text_clean = re.sub(r'Nguyn', 'Nguyen', text_clean)

    elif field_type == 'Phone':
        # Giữ số và các ký tự phone cần thiết
        text_clean = re.sub(r'[^0-9\s+\-()]', '', text_clean)

    elif field_type == 'Email':
        # Giữ ký tự email hợp lệ
        text_clean = re.sub(r'[^a-zA-Z0-9@._\-]', '', text_clean)

    elif field_type == 'ID':
        # Chỉ giữ số
        text_clean = re.sub(r'[^0-9]', '', text_clean)

    elif field_type == 'Address':
        # Giữ chữ cái, số, khoảng trắng và ký tự đặc biệt thông dụng trong địa chỉ
        text_clean = re.sub(r'[^a-zA-ZÀ-ỹ0-9\s,.-]', '', text_clean)
        # Sửa lỗi thường gặp
        text_clean = re.sub(r'HaN[i1l6]', 'HaNoi', text_clean)
        text_clean = re.sub(r'HN[i1l6]', 'Hanoi', text_clean)
        text_clean = re.sub(r'HaN[i1l6]', 'HaNoi', text_clean)

    elif field_type == 'Class':
        # Giữ chữ cái, số và khoảng trắng
        text_clean = re.sub(r'[^a-zA-Z0-9\s]', '', text_clean)

    # Xóa khoảng trắng thừa sau khi xử lý
    text_clean = ' '.join(text_clean.split())

    return text_clean.strip()


# Đọc ảnh mẫu
imgQ = cv2.imread('Query.png')
if imgQ is None:
    print("Lỗi: Không thể đọc file Query.png")
    exit()

h, w, c = imgQ.shape
print(f"Kích thước ảnh mẫu: {w}x{h}")

# Khởi tạo ORB
orb = cv2.ORB_create(5000)
kp1, des1 = orb.detectAndCompute(imgQ, None)

# Hiển thị ảnh mẫu với keypoints (resized để xem)
imgQ_display = cv2.resize(imgQ, (w // 2, h // 2))
cv2.imshow('Query Image', imgQ_display)

# Xử lý từng form
path = 'UserForms'
myPicList = os.listdir(path)
print(f"Tìm thấy {len(myPicList)} form(s): {myPicList}")

for j, y in enumerate(myPicList):
    print(f"\n--- Xử lý form {j + 1}: {y} ---")

    # Đọc ảnh form
    img_path = os.path.join(path, y)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Lỗi: Không thể đọc file {img_path}")
        continue

    # Tạo bản copy để xử lý - GIỮ NGUYÊN KÍCH THƯỚC GỐC
    img_processing = img.copy()

    # Detect keypoints trên ảnh form
    kp2, des2 = orb.detectAndCompute(img_processing, None)

    if des2 is None:
        print(f"Không tìm thấy keypoints trong {y}")
        continue

    # Feature matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des1, des2)

    # QUAN TRỌNG: Chuyển matches thành list nếu là tuple
    if isinstance(matches, tuple):
        matches = list(matches)

    if len(matches) == 0:
        print(f"Không tìm thấy matches trong {y}")
        continue

    # SẮP XẾP MATCHES - GIỜ ĐÃ AN TOÀN
    matches.sort(key=lambda x: x.distance)
    good = matches[:int(len(matches) * (per / 100))]

    print(f"Tổng matches: {len(matches)}, Good matches: {len(good)}")

    # Vẽ matches để debug (resized để hiển thị)
    img_match = cv2.drawMatches(imgQ, kp1, img_processing, kp2, good[:500], None, flags=2)
    img_match_display = cv2.resize(img_match, (w, h // 2))
    cv2.imshow(f'Matches {y}', img_match_display)

    # Tìm Homography
    if len(good) > 4:
        srcPoints = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dstPoints = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(dstPoints, srcPoints, cv2.RANSAC, 5.0)

        if M is not None:
            # Warp ảnh form về ảnh mẫu
            img_warped = cv2.warpPerspective(img_processing, M, (w, h))

            # Hiển thị ảnh đã warp (resized)
            img_warped_display = cv2.resize(img_warped, (w // 2, h // 2))
            cv2.imshow(f'Warped {y}', img_warped_display)

            # Tạo ảnh để hiển thị kết quả
            img_show = img_warped.copy()
            img_mask = np.zeros_like(img_show)

            myData = []
            print(f'Trích xuất dữ liệu từ form {j + 1}...')

            # ĐẦU TIÊN: XỬ LÝ OCR TẤT CẢ CÁC TRƯỜNG
            for x, r in enumerate(roi_original):
                # Cắt vùng ROI từ ảnh GỐC - SỬ DỤNG ROI GỐC
                img_crop = img_warped[r[0][1]:r[1][1], r[0][0]:r[1][0]]

                # Kiểm tra xem vùng crop có dữ liệu không
                if img_crop.size == 0:
                    print(f"Vùng ROI {r[3]} trống, bỏ qua...")
                    myData.append('')
                    continue

                # Tiền xử lý ảnh để cải thiện OCR
                img_crop_processed = preprocess_for_ocr(img_crop)

                # Hiển thị ảnh đã xử lý (resized)
                img_crop_display = cv2.resize(img_crop_processed, (200, 50))
                cv2.imshow(f'ROI {r[3]}', img_crop_display)

                if r[2] == 'text':
                    # Cấu hình Tesseract tối ưu cho từng loại trường
                    if r[3] in ['Name', 'Class', 'Address']:
                        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠ-ỹ0123456789\s'
                    elif r[3] == 'Phone':
                        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789+-\s()'
                    elif r[3] == 'ID':
                        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
                    elif r[3] == 'Email':
                        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789@.-_'
                    else:
                        custom_config = r'--oem 3 --psm 7'

                    text = pytesseract.image_to_string(img_crop_processed, config=custom_config)
                    text_clean = clean_ocr_text(text, r[3])
                    print(f'{r[3]}: "{text_clean}"')
                    myData.append(text_clean)
                else:
                    myData.append('')

            # QUAN TRỌNG: VẼ TEXT SAU KHI ĐÃ CÓ TẤT CẢ DỮ LIỆU
            # Tạo một bản copy riêng để vẽ text mà không bị ảnh hưởng bởi mask
            img_result = img_warped.copy()

            # Vẽ mask cho tất cả ROI
            for x, r in enumerate(roi_original):
                cv2.rectangle(img_mask, (r[0][0], r[0][1]), (r[1][0], r[1][1]), (0, 255, 0), cv2.FILLED)

            # Kết hợp mask với ảnh gốc
            img_result = cv2.addWeighted(img_warped, 0.7, img_mask, 0.3, 0)

            # Vẽ text lên ảnh kết quả
            for x, r in enumerate(roi_original):
                if x < len(myData) and myData[x]:
                    # Vị trí vẽ text - điều chỉnh để hiển thị rõ ràng
                    text_x = r[0][0] + 5
                    text_y = r[0][1] - 5

                    # Nếu vị trí quá cao, vẽ bên trong ROI
                    if text_y < 20:
                        text_y = r[0][1] + 25

                    # Vẽ nền cho text để dễ đọc
                    text_size = cv2.getTextSize(str(myData[x]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(img_result,
                                  (text_x - 2, text_y - text_size[1] - 2),
                                  (text_x + text_size[0] + 2, text_y + 2),
                                  (255, 255, 255), cv2.FILLED)

                    # Vẽ text
                    cv2.putText(img_result, str(myData[x]), (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Lưu dữ liệu
            with open('DataOutput.csv', 'a+', encoding='utf-8') as f:
                f.write(f"{y},")
                for data in myData:
                    f.write(f"{data},")
                f.write("\n")

            print(f"Dữ liệu trích xuất: {myData}")

            # Hiển thị kết quả (resized) - SỬ DỤNG IMG_RESULT
            img_show_display = cv2.resize(img_result, (w // 2, h // 2))
            cv2.imshow(f'Result {y}', img_show_display)

        else:
            print(f"Không thể tìm Homography cho {y}")
    else:
        print(f"Không đủ matches tốt cho {y} (cần >4, có {len(good)})")

    # Chờ phím để tiếp tục
    print("Nhấn phím bất kỳ để tiếp tục, 'q' để thoát...")
    key = cv2.waitKey(0)
    if key == ord('q'):
        break

    # Đóng các cửa sổ tạm
    cv2.destroyWindow(f'Matches {y}')
    cv2.destroyWindow(f'Warped {y}')
    cv2.destroyWindow(f'Result {y}')
    for i in range(6):
        cv2.destroyWindow(f'ROI {roi_original[i][3]}')

cv2.destroyAllWindows()
print("\nHoàn thành xử lý tất cả forms!")