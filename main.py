from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import base64
import os
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

# โฟลเดอร์เก็บรูปพนักงาน (ชื่อไฟล์ต้องเป็น USER_ID.jpg)
DB_PATH = "img"
if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)


@app.route('/verify-deepface', methods=['POST'])
def verify_deepface():
    temp_filename = "" # สร้างตัวแปรไว้ก่อนกัน Error ตอนลบไฟล์
    try:
        # --- เปลี่ยนมาใช้ request.json ทั้งหมด ---
        data = request.json
        if not data:
            return jsonify({"status": "fail", "message": "No JSON payload received"}), 400

        user_id = data.get('userId')
        img_raw = data.get('image') # ใน payload JS ใช้ชื่อ 'image'

        if not user_id or not img_raw:
            return jsonify({"status": "fail", "message": "Missing userId or image"}), 400

        # 1. จัดการ Base64
        if "," in img_raw:
            img_b64 = img_raw.split(",")[1]
        else:
            img_b64 = img_raw

        decoded_data = base64.b64decode(img_b64)
        np_data = np.frombuffer(decoded_data, np.uint8)
        img_from_camera = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

        if img_from_camera is None:
            return jsonify({"status": "fail", "message": "Invalid image data"}), 400

        # 2. ค้นหารูปต้นฉบับ
        reference_img_path = os.path.join(DB_PATH, f"{user_id}.png")
        if not os.path.exists(reference_img_path):
            return jsonify({"verified": False, "message": f"User {user_id} not registered"}), 404

        # 3. บันทึกรูปชั่วคราวเพื่อใช้กับ DeepFace
        temp_filename = f"temp_{user_id}.jpg"
        cv2.imwrite(temp_filename, img_from_camera)

        # 4. ใช้ DeepFace.verify (แม่นยำที่สุดสำหรับการเช็ค ID)
        rdfs = DeepFace.find(
                img_path=temp_filename,
                db_path=DB_PATH,
                model_name="VGG-Face",
                detector_backend="opencv", 
                enforce_detection=True,
                silent=True,
                align=True
            )

        # ลบไฟล์ชั่วคราวหลังใช้งาน
        if len(rdfs) > 0 and not rdfs[0].empty:
                # เอาผลลัพธ์แรกที่ใกล้เคียงที่สุด
                match = rdfs[0].iloc[0]
                distance = match['distance']
                
                # Threshold ปกติของ VGG-Face คือประมาณ 0.40
                if distance < 0.40:


                    if os.path.exists(temp_filename): os.remove(temp_filename)

                    return jsonify({
                        'status': 'success', 
                        'message': 'ลงเวลาสำเร็จ ✅',
                        'distance': float(distance)
                    })

            # กรณีไม่เจอ
                if os.path.exists(temp_filename): os.remove(temp_filename)
                return jsonify({'status': 'fail', 'message': '❌ ไม่พบข้อมูล', 'distance': 0})
        
    except Exception as e:
        print("Error:", str(e))
        if temp_filename and os.path.exists(temp_filename):
            os.remove(temp_filename)
        return jsonify({"status": "error", "error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=9000, threaded=True)