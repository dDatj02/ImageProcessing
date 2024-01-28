import cv2
import numpy as np

# Đọc ảnh đầu vào
image1 = cv2.imread("input1.jpg")
input_image = cv2.imread('input2.png', 0)
input_image1 = cv2.imread("input2.png")

###############Programing 1a###############
# Chuyển ảnh sang không gian màu HSV
hsv_image = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)

# Định nghĩa các ngưỡng màu cho từng ngôi sao (ví dụ màu vàng, màu xanh, đỏ, vv.)
lower_thresholds = [np.array([25, 50, 70]),  # Màu vàng
                    np.array([60, 100, 100]),  # Màu xanh lá
                    np.array([159, 50, 70]),   # Màu hồng
                    np.array([90, 50, 70]),  # Màu xanh dương
                    np.array([5, 160, 180]),  # Màu cam
                    np.array([129, 50, 70]),   # Màu tím
                    # Thêm các ngưỡng cho các màu khác nếu cần
                   ]

upper_thresholds = [np.array([35, 255, 255]),  # Màu vàng
                    np.array([100, 255, 255]),  # Màu xanh
                    np.array([180, 255, 255]),   # Màu hồng
                    np.array([128, 255, 255]),  # Màu xanh dương
                    np.array([24, 255, 255]),  # Màu cam
                    np.array([158, 255, 255]),   # Màu tím
                    # Thêm các ngưỡng cho các màu khác nếu cần
                   ]

# Trích xuất từng ngôi sao với màu tương ứng và lưu thành các tệp riêng lẻ
for i in range(len(lower_thresholds)):
    mask = cv2.inRange(hsv_image, lower_thresholds[i], upper_thresholds[i])
    extracted_star = cv2.bitwise_and(image1, image1, mask=mask)
    output_file = f'output_star_{i}.jpg'
    cv2.imwrite(output_file, extracted_star)




###############Programing 1b###############
# Chuyển đổi ảnh sang không gian màu thang độ xám
gray_image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

# Sử dụng ngưỡng để vẽ lại viền trắng thành màu đen
_, thresholded_image = cv2.threshold(gray_image, 220, 0, cv2.THRESH_TOZERO_INV)

# Loại bỏ nhiễu (optional)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
opened_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel, iterations=2)

# Lưu kết quả hình ảnh đầu ra
output_file = 'output_image.jpg'
cv2.imwrite(output_file, opened_image)


###############Programing 1d###############
_, thresholded_image = cv2.threshold(gray_image, 200, 0, cv2.THRESH_TOZERO_INV)
_, thresholded_image = cv2.threshold(thresholded_image, 160, 255, cv2.THRESH_BINARY)


# Loại bỏ nhiễu (optional)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
opened_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel, iterations=2)

# Lưu kết quả hình ảnh đầu ra
output_file = 'output_image_1.jpg'
cv2.imwrite(output_file, opened_image)




###############Programing 2###############
# Áp dụng ngưỡng nhị phân để trích xuất các chữ số
_, thresh = cv2.threshold(input_image, 127, 255, cv2.THRESH_BINARY_INV)

# Tìm đường viền của các chữ số riêng lẻ
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Vẽ hình chữ nhật xung quanh mỗi đường viền chữ số
for contour in contours:
    # Kiểm tra xem đường viền có đủ lớn để trở thành một chữ số không
    if cv2.contourArea(contour) > 100:
        # Tính hình chữ nhật giới hạn của đường viền
        x, y, w, h = cv2.boundingRect(contour)

        # Vẽ một hình chữ nhật xung quanh hình chữ nhật giới hạn
        cv2.rectangle(input_image1, (x, y), (x + w, y + h), (0, 255, 0), 2)


# Lưu kết quả hình ảnh đầu ra 
cv2.imwrite('output_image_2.jpg', input_image1)