import streamlit as st
import numpy as np
import onnxruntime as ort 
import requests
from PIL import Image
import io
import os
import time
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
import paho.mqtt.client as mqtt



# สร้างแอป Flask สำหรับรับภาพจาก HTTP ภายนอก
app = Flask(__name__)
CORS(app)


# ตัวแปร global สำหรับเก็บภาพล่าสุดที่รับเข้ามา
latest_image = None

# MQTT broker details
broker = "d8229ac5fefe43a9a7c09fabb5f30929.s1.eu.hivemq.cloud"
port = 8883
username = "python"
password = "123456789"
topic = "esp32cam/servo"

# Create an MQTT client instance
client = mqtt.Client()
client.tls_set()  # ใช้ TLS สำหรับการเชื่อมต่อที่ปลอดภัย
client.username_pw_set(username, password)
client.connect(broker, port)

@app.route('/api/send-image', methods=['POST'])
def receive_image():
    global latest_image
    # รับข้อมูลภาพจากคำขอ POST
    file = request.files['image']  # ดึงไฟล์จากคำขอ
    if file:
        # อ่านภาพจากไฟล์และแปลงเป็นไบต์
        image_bytes = file.read()  
        latest_image = Image.open(io.BytesIO(image_bytes))  # เปิดไฟล์ภาพจากไบต์
        return jsonify({"status": "Image received"}), 200
    return jsonify({"status": "No image received"}), 400

# รัน Flask Server ใน Thread แยกจาก Streamlit
def run_flask():
    app.run(port=5000)

# เริ่ม Flask ใน Thread ใหม่
threading.Thread(target=run_flask, daemon=True).start()

# ลิงก์ดาวน์โหลดโมเดล ONNX สำหรับการจำแนกรถและมอเตอร์ไซค์
vehicle_model_url = 'https://www.dropbox.com/scl/fi/zlmm6k6u96qgemddm4bzn/vehicle_classification.onnx?rlkey=pvcrm0bv2vxczmhou6bfqoa9h&st=4pplshvw&dl=1'
helmet_model_url = 'https://www.dropbox.com/scl/fi/gd8djpwcr9itx3nkxgjbr/helmet_detection_model.onnx?rlkey=f5p5ezg76wdicvcuzfw4kzuub&st=bpc3tiuu&dl=1'

# ดาวน์โหลดและบันทึกโมเดลชั่วคราวในเครื่อง
def download_model(url, model_name):
    response = requests.get(url)
    with open(model_name, 'wb') as f:
        f.write(response.content)

# ดาวน์โหลดโมเดล
download_model(vehicle_model_url, 'temp_vehicle_classification_model.onnx')
download_model(helmet_model_url, 'temp_helmet_detection_model.onnx')

# โหลดโมเดลจากไฟล์ชั่วคราวโดยใช้ ONNX Runtime
ort_session_vehicle = ort.InferenceSession('temp_vehicle_classification_model.onnx')
ort_session_helmet = ort.InferenceSession('temp_helmet_detection_model.onnx')

class_names_vehicle = ['Motorcycle', 'Car']
class_names_helmet = ['With Helmet', 'Without Helmet']

def prepare_image(img, img_width=150, img_height=150):
    img = img.resize((img_width, img_height))  # ปรับขนาดภาพ
    img_array = np.array(img).astype(np.float32)  # แปลงเป็น float32
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # แปลงเป็น array แบบ 0-1
    return img_array

# ฟังก์ชันสำหรับตรวจสอบรถหรือมอเตอร์ไซค์
def classify_vehicle(img):
    inputs = {ort_session_vehicle.get_inputs()[0].name: img}
    outputs = ort_session_vehicle.run(None, inputs)
    prediction = outputs[0]
    predicted_vehicle_class = np.argmax(prediction)
    vehicle_confidence = np.max(prediction)
    return predicted_vehicle_class, vehicle_confidence

# ฟังก์ชันสำหรับตรวจสอบหมวกกันน็อคโดยใช้ ONNX
def classify_helmet(img):
    inputs = {ort_session_helmet.get_inputs()[0].name: img}  # แก้ไขให้ใช้ ort_session_helmet
    outputs = ort_session_helmet.run(None, inputs)
    
    # วิเคราะห์ผลลัพธ์
    prediction = outputs[0]
    predicted_helmet_class = np.argmax(prediction)
    helmet_confidence = np.max(prediction)
    return predicted_helmet_class, helmet_confidence

def main():
    st.title("Helmet Detection")
    check = False
    # ตัวแปรเพื่อเช็คว่ามีการทำนายหรือไม่
    latest_image_cache = None  # ตัวแปรใหม่เพื่อตรวจสอบภาพล่าสุด
    wait_message_shown = False  # ตัวแปรใหม่เพื่อตรวจสอบการแสดงข้อความ

    # สร้าง placeholder สำหรับแสดงผล
    image_placeholder = st.empty()
    result_placeholder = st.empty()
    imagehelmet_placeholder = st.empty()
    resulthelmet_placeholder = st.empty()
    wait_message_shown_placeholder = st.empty()
    pass_placeholder = st.empty()
    client.publish(topic, "camerastart")

    while True:  # ลูปไม่สิ้นสุดสำหรับตรวจสอบภาพใหม่
        # เช็คว่ามีภาพล่าสุดหรือไม่
        if check == False :
            if latest_image is not None:
                # ตรวจสอบว่าภาพใหม่หรือไม่
                if latest_image_cache != latest_image:  # เช็คว่ามีการเปลี่ยนแปลงหรือไม่
                    latest_image_cache = latest_image  # อัปเดตภาพล่าสุดที่แคชไว้

                    # เคลียร์พื้นที่การแสดงภาพเก่า
                    image_placeholder.empty()  # เคลียร์พื้นที่การแสดงภาพเก่า
                    result_placeholder.empty()  # เคลียร์พื้นที่การแสดงผลการทำนายเก่า
                    imagehelmet_placeholder.empty()
                    resulthelmet_placeholder.empty()
                    pass_placeholder.empty()
                    

                    # แสดงภาพล่าสุด
                    image_placeholder.image(latest_image, caption="Captured Image", use_column_width=True)

                    # เตรียมภาพเพื่อใช้กับโมเดล
                    img_array = prepare_image(latest_image)

                    # ตรวจสอบรถ
                    predicted_vehicle_class, vehicle_confidence = classify_vehicle(img_array)
                    vehicle_result_text = f"Vehicle Prediction: {class_names_vehicle[predicted_vehicle_class]} with confidence {vehicle_confidence * 100:.2f}%"
                    result_placeholder.write(vehicle_result_text)
                    
                        
                    # ถ้าผลลัพธ์เป็นมอเตอร์ไซค์ ให้ถ่ายภาพอีกครั้งเพื่อตรวจสอบหมวกกันน็อค
                    if predicted_vehicle_class == 0:  # 0 หมายถึงมอเตอร์ไซค์
                        check = True
                    # ถ้าผลลัพธ์เป็นมอเตอร์ไซค์ ให้ถ่ายภาพอีกครั้งเพื่อตรวจสอบหมวกกันน็อค
                    elif predicted_vehicle_class == 1:
                        client.publish(topic, "pass")
                        pass_placeholder.markdown("<p style='color: green;'>Pass</p>", unsafe_allow_html=True)
                        check = False
            time.sleep(1)  # รอ 1 วินาทีก่อนตรวจสอบอีกครั้ง
        elif check == True :
            client.publish(topic, "camerastart")
            wait_message_shown_placeholder.write("Waiting for pictures of helmets...")
            if latest_image is not None:
                # ตรวจสอบว่าภาพใหม่หรือไม่
                if latest_image_cache != latest_image:  # เช็คว่ามีการเปลี่ยนแปลงหรือไม่
                    latest_image_cache = latest_image  # อัปเดตภาพล่าสุดที่แคชไว้
                    wait_message_shown_placeholder.empty()
                    # แสดงภาพล่าสุด
                    imagehelmet_placeholder.image(latest_image, caption="Capture Helmet Image", use_column_width=True)

                    # เตรียมภาพเพื่อใช้กับโมเดล
                    img_array = prepare_image(latest_image)
                    predicted_helmet_class, helmet_confidence = classify_helmet(img_array)
                    if helmet_confidence < 0.6:  # ถ้าความมั่นใจต่ำกว่า 60%
                        client.publish(topic, "no_pass")
                        pass_placeholder.markdown("<p style='color: red;'>Error: Confidence is too low</p>", unsafe_allow_html=True)
                    else:
                        if predicted_helmet_class == 0:  # 0 หมายถึง With Helmet
                            client.publish(topic, "pass")
                            helmet_result_text = f"Helmet Prediction: {class_names_helmet[predicted_helmet_class]} with confidence {helmet_confidence * 100:.2f}%"
                            resulthelmet_placeholder.write(helmet_result_text)
                            pass_placeholder.markdown("<p style='color: green;'>Pass</p>", unsafe_allow_html=True)
                        elif predicted_helmet_class == 1:
                            client.publish(topic, "no_pass")
                            helmet_result_text = f"Helmet Prediction: {class_names_helmet[predicted_helmet_class]} with confidence {helmet_confidence * 100:.2f}%"
                            resulthelmet_placeholder.write(helmet_result_text)
                            pass_placeholder.markdown("<p style='color: red;'>No-Pass</p>", unsafe_allow_html=True)
                    check = False
            time.sleep(1)  # รอ 1 วินาทีก่อนตรวจสอบอีกครั้ง
            

    # ลบไฟล์โมเดลชั่วคราวเมื่อเลิกใช้งาน
    if os.path.exists('temp_vehicle_classification_model.onnx'):
        os.remove('temp_vehicle_classification_model.onnx')
    if os.path.exists('temp_helmet_detection_model.onnx'):
        os.remove('temp_helmet_detection_model.onnx')

if __name__ == "__main__":
    main()