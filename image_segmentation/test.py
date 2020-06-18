from keras_segmentation.pretrained import pspnet_50_ADE_20K , pspnet_101_cityscapes, pspnet_101_voc12
import argparse

model = pspnet_50_ADE_20K()

parser = argparse.ArgumentParser()
parser.add_argument('-image_path','--image_path', help='provide image_path', required=True)
args = parser.parse_args()

out = model.predict_segmentation(
    inp=args.image_path,
    out_fname="out.png"
)