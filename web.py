import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import streamlit as st

import streamlit as st
import base64





# Load the pre-trained model
model = tf.keras.models.load_model('braintumor.h5')

# Tiêu đề trang web
st.title("Phân loại u não")

# Cho phép người dùng tải lên một hình ảnh
uploaded_file = st.file_uploader("Chọn một hình MRI não định dạng JPG...", type="jpg")

if uploaded_file is not None:
    # Đọc hình ảnh từ đối tượng file_uploader sử dụng OpenCV thay vì PIL
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Hiển thị hình ảnh đã chọn
    st.image(img, caption="Hình ảnh đã tải lên.", use_column_width=True)

    # Tiền xử lý hình ảnh để dự đoán
    img_array = cv2.resize(img, (150, 150))
    img_array = img_array.reshape(1, 150, 150, 3)

    # Thực hiện dự đoán
    predictions = model.predict(img_array)

    # Lấy nhãn lớp dự đoán
    class_labels = ["Glioma tumor", "Meningioma tumor", "No tumor", "Pituitary tumor"]
    predicted_class = np.argmax(predictions)
    st.write(predictions)
    st.write(predicted_class)

    # Hiển thị kết quả
    st.write(f"Loại dự đoán: {class_labels[predicted_class]}")
