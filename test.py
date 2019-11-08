import os
import argparse
from scipy import misc
import scipy.io as sio
import cv2
import time
import numpy as np
import tensorflow.python.framework.dtypes
from warpgan import WarpGAN
from align.detect_align import detect_align

# Parse aguements
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", help="The path to the pretrained model",
                        type=str, default="./warpgan_pretrained/warpgan_pretrained")
parser.add_argument("--input", help="The path to the aligned image",
                        type=str, default="./data/oriImgs")
parser.add_argument("--output", help="The prefix path to the output file, subfix will be added for different styles.",
                        type=str, default="./data/result")
parser.add_argument("--num_styles", help="The number of images to generate with different styles",
                        type=int, default=5)
parser.add_argument("--scale", help="The path to the input directory",
                        type=float, default=1.0)
parser.add_argument("--aligned", help="Set true if the input face is already normalized",
                        action='store_true', default=False)


args = parser.parse_args()
print("args ", args)


if __name__ == '__main__':

    network = WarpGAN()
    network.load_model(args.model_dir)

    for name in os.listdir(args.input):
        imgfile = os.path.join(args.input, name)
        img = misc.imread(imgfile, mode='RGB')

        if not args.aligned:
            s = time.time()
            img = detect_align(img)
            e = time.time()
            print("detect time cost ", e - s, "   s")
            if img is None:
                print("detect failed *********** ", imgfile)
                continue
            cv2.imshow("img ", img)
            # cv2.waitKey(0)

        img = (img - 127.5) / 128.0

        images = np.tile(img[None], [args.num_styles, 1, 1, 1])
        scales = args.scale * np.ones((args.num_styles))
        styles = np.random.normal(0., 1., (args.num_styles, network.input_style.shape[1].value))

        start = time.time()
        output = network.generate_BA(images, scales, 16, styles=styles)
        output = 0.5*output + 0.5
        end = time.time()
        print("generate caricatue time cost: ", end  - start, "   s.")

        for i in range(args.num_styles):
            outdir = os.path.join(args.output, name[:-4])
            misc.imsave(outdir + '_{}.jpg'.format(i), output[i])
            cv2.imshow("img ", output[i])
            # cv2.waitKey(0)
        break


