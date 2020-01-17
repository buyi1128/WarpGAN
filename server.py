from socketserver import StreamRequestHandler, TCPServer
from socket import error as SocketError
import errno
import datetime
import time
import os
import base64
import numpy as np
import cv2
from scipy import misc

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
num_styles = 1
warpGAN = GANnetworks(isAligned=False, num_styles=num_styles)
# image_file = "image/2019-11-29_13_32_50.jpg"
# image = misc.imread(image_file, mode='RGB')
# outputs = warpGAN.generate_cartoon(image)
# for i in range(4):
#     # outdir = os.path.join("image", image_file[:-4])
#     misc.imsave(image_file[:-4] + '_{}.jpg'.format(i), outputs[i])

class MyTCPHandler(StreamRequestHandler):

    def handle(self):
        op = ""
        imglens = 0
        lens = 0
        datalist = []
        print("op: ", op, "  image1: ", len(datalist))
        try:
            while (imglens == 0 or lens < imglens):
                data = self.request.recv(10240)  # 拿到客户端发送的数据
                if not data or len(data) == 0:
                    break
                # print(data)
                if len(op) == 0:
                    op = data.decode("utf-8")
                    msg = "OK".encode("utf-8")
                    self.request.send(msg)
                    print("op : ", op)
                elif len(op) > 0 and imglens == 0:
                    print("客户端传送图片大小： ", data)
                    imglens = data.decode("utf-8")
                    imglens = int(imglens)
                    print("imglens: ", imglens, "  lens: ", lens)
                else:
                    datalist.extend(data)
                    lens = lens + len(data)
                    # print("处理中...", "imglens: ", imglens, "  lens: ", lens)
            if len(datalist) > 0 and lens == imglens:
                image = np.asarray(bytearray(datalist), dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                image_file = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S.jpg')
                print("save file to: : ", image_file)
                cv2.imwrite(os.path.join("./image", image_file), image)
                print("接收完成")

                ## 生成漫画图片
                outputs = warpGAN.generate_cartoon(image)
                for i in range(num_styles):
                    outdir = os.path.join("image", image_file[:-4])
                    misc.imsave(outdir + '_{}.jpg'.format(i), outputs[i])
                    # cv2.imshow("img ", output[i])
                ## 返回给客户端
                img_encode = cv2.imencode(".jpg", outputs[0])[1]
                data_encode = np.array(img_encode)
                img_data = data_encode.tostring()
                lenImg = str(len(img_data)) + "\t"
                print("发送图片大小:", lenImg)

                print("??? ", type(img_encode), img_encode.shape, type(data_encode), data_encode.shape)
                cv2.imwrite("./image/temp.jpg", outputs[0])

                self.request.send(lenImg.encode("utf-8"))
                self.request.send(img_data)

            else:
                print("%%%%%%%%%%%% error image is none or break!")
            self.request.send("done".encode("utf-8"))
        except Exception as e:
            print(e)
            print(self.client_address, "error : 连接断开")
        finally:
            self.request.close()  # 异常之后，关闭连接

    # before handle,连接建立：
    def setup(self):
        now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n\ntime: ", now_time, "  连接建立：", self.client_address)

    # finish run  after handle
    def finish(self):
        print("释放连接")

if __name__ == "__main__":
    from threading import Thread

    try:
        NWORKERS = 16
        TCPServer.allow_reuse_address = True
        serv = TCPServer(('', 8999), MyTCPHandler)
        for n in range(NWORKERS):
            t = Thread(target=serv.serve_forever)
            t.daemon = True
            t.start()
        serv.serve_forever()
    except Exception as e:
        print("exit: ", e)