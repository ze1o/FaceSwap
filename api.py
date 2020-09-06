import os
from pathlib import Path
import base64

from flask import Flask, request, jsonify
import numpy as np
import cv2

from face_detection import select_face
from face_swap import face_swap

global dst_img_info
dst_img_info = dict()

def generate_dst_img_info():
    info = dict()

    dst_img_dir = Path("imgs/src")
    for img_path in dst_img_dir.glob("*.jpg"):
        path = os.path.abspath(img_path)
        img = cv2.imread(path)
        points, shape, face = select_face(img)
        img_name = os.path.basename(img_path).split(".")[0]
        img_info = dict(points=points, shape=shape, face=face, img=img)
        info[img_name] = img_info

    return info

app = Flask(__name__)

@app.route('/swap-face', methods=['POST'])
def swap_face():
    try:
        bin_img = request.data
        arr_img = np.fromstring(bin_img, np.uint8)
        src_img = cv2.imdecode(arr_img, cv2.IMREAD_COLOR)

        gender = request.args["gender"]
        character = request.args["character"]

        # Select src face
        src_points, src_shape, src_face = select_face(src_img, choose=False)

        dst_name = gender + "-" + character
        dst = dst_img_info[dst_name]
        dst_points = dst["points"]
        dst_shape = dst["shape"]
        dst_face = dst["face"]
        dst_img = dst["img"]

        out_img = face_swap(src_face, dst_face, src_points, dst_points, dst_shape, dst_img, correct_color=True, warp_2d=False)

        _, img = cv2.imencode(".jpg", out_img)
        return jsonify(img=str(base64.b64encode(img),'utf-8'), error=False)
    except Exception:
        return jsonify(error=True)

if __name__ == '__main__':
    dst_img_info = generate_dst_img_info()
    app.run(debug=True,host='0.0.0.0', port=8080)