
from flask import Flask
from flask import Response
from flask import request
from flask import send_file
import werkzeug
import os
import time, datetime
import numpy as np
import cv2
import json
from scipy import misc
import imageio
import base64

from warpgan import WarpGAN
from align.detect_align import detect_align

class GANnetworks:
    def __init__(self, isAligned, num_styles):
        self.warpgan_dir = "./warpgan_pretrained/warpgan_pretrained"
        self.isAligned = isAligned
        self.num_styles = num_styles
        self.warpGAN = self.load_warpGAN()

    def load_warpGAN(self):
        network = WarpGAN()
        network.load_model(self.warpgan_dir)
        return network

    def generate_cartoon(self, img):

        if not self.isAligned:
            s = time.time()
            img = detect_align(img)
            e = time.time()
            print("detect time cost ", e - s, "   s")
            if img is None:
                print("no face in img ******")
                return
        img = (img - 127.5) / 128.0

        images = np.tile(img[None], [self.num_styles, 1, 1, 1])
        scales = 1.0 * np.ones((self.num_styles))
        styles = np.random.normal(0., 1., (self.num_styles, self.warpGAN.input_style.shape[1].value))

        start = time.time()
        output = self.warpGAN.generate_BA(images, scales, 16, styles=styles)
        output = 0.5 * output + 0.5
        end = time.time()
        print("generate caricatue time cost: ", end - start, "   s.")
        return output

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Just disables the warning, doesn't enable AVX/FMA
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
num_styles = 4
warpGAN = GANnetworks(isAligned=False, num_styles=num_styles)


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def handle_request():
    return "Flask Server & Android are Working Successfully"

# 客户端上传图片
@app.route('/upload', methods=['POST', 'GET'])
def get_face_img():
    imagefile = request.files['image']
    operation = werkzeug.utils.secure_filename(imagefile.filename)
    image_array = imagefile.read()
    image = np.asarray(bytearray(image_array), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image_file = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S.jpg')
    print("operation is ", operation, " save file to: : ", image_file)
    cv2.imwrite(os.path.join("./image", image_file), image)
    print("接收完成")

    ## 生成漫画图片
    outputs = warpGAN.generate_cartoon(image)
    for i in range(num_styles):
        outdir = os.path.join("image", image_file[:-4])
        imageio.imwrite(outdir + '_{}.jpg'.format(i), outputs[i])

        # cv2.imshow("img ", output[i])
        print("生成漫画图片，", i)
    ## 返回给客户端
    outjson = {}
    outjson["num_styles"] = num_styles
    for i in range(num_styles):
        filename = os.path.join("image", image_file[:-4] + '_{}.jpg'.format(i))
        print(filename)
        if not os.path.exists(filename):
            print("error , image not exist ???????????????????? ")
        with open(filename, 'rb') as fp:
            img_bytes = base64.b64encode(fp.read())
            outjson[str(i)] = img_bytes.decode('utf-8')
    json_data = json.dumps(outjson)

    return Response(json_data, mimetype="application/json")

app.run(host="0.0.0.0", port=8999, debug=True)