import io
import json
import cv2
import base64


def numpy2base64(image):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    _, buffer = cv2.imencode('.jpg', image, encode_param)
    b64 = base64.b64encode(buffer).decode('utf-8')
    return b64


def pil2base64(image):
    image = corp_image(image)
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return b64


def load_json(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding {f}: {e}")
    return data


def corp_image(image):
    target_size = (667, 500)
    image = image.copy()
    width, height = image.size
    left = (width - target_size[0]) // 2
    top = (height - target_size[1]) // 2
    right = left + target_size[0]
    bottom = top + target_size[1]
    image = image.crop((left, top, right, bottom))
    return image