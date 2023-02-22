import os
from sys import path
import argparse

parser = argparse.ArgumentParser(description="Overlay Label Map On Top Of An Image.")
parser.add_argument("input_image")
parser.add_argument("label_map")
parser.add_argument("output_image")
args = parser.parse_args()

def main():
    conf = configure()
    model = Unet_3D(tf.Session(), conf)
    generate_h5.build_h5_dataset(conf.raw_data_dir,conf.data_dir)
    getattr(model, 'predict')()
    return 'DONE!!'

if __name__ == '__main__':
    # configure which gpu or cpu to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tf.app.run()


if not os.path.exists(output_image_path):
    os.makedirs(output_image_path)

## antsRegistrationSyNQuick # s: rigid + affine + deformable syn (3 stages)
command1 = 'antsRegistrationSyNQuick.sh -d 3' + \
' -m ' + input_image_path                     + \
' -f ' + template                             + \
' -t ' + 's'                                  + \
' -o ' + output_image_path                    + \
' -n ' + '16'
# print(command1)
# os.system(command1)