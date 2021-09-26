# # from __future__ import absolute_import
# # from __future__ import division
# # from __future__ import print_function
# import argparse
# import sys
# import time
# from models.gazenet import GazeNet

# import torch
# import torch.nn as nn
# import models


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print('Device', device)

# # def load_model_weight(model, checkpoint):
# #     state_dict = checkpoint['model_state_dict']
# #     # strip prefix of state_dict
# #     if list(state_dict.keys())[0].startswith('module.'):
# #         state_dict = {k[7:]: v for k, v in checkpoint['model_state_dict'].items()}

# #     model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

# #     # check loaded parameters and created model parameters
# #     for k in state_dict:
# #         if k in model_state_dict:
# #             if state_dict[k].shape != model_state_dict[k].shape:
# #                 print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(
# #                     k, model_state_dict[k].shape, state_dict[k].shape))
# #                 state_dict[k] = model_state_dict[k]
# #         else:
# #             print('Drop parameter {}.'.format(k))
# #     for k in model_state_dict:
# #         if not (k in state_dict):
# #             print('No param {}.'.format(k))
# #             state_dict[k] = model_state_dict[k]
# #     model.load_state_dict(state_dict, strict=False)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     # parser.add_argument('--weights', type=str, default="./checkpoint/snapshot/checkpoint_epoch_1.pth.tar", help='weights path')  # from yolov5/models/
#     parser.add_argument('--weights', type=str, default="./models/weights/gazenet.pth", help='weights path')  # from yolov5/models/

#     parser.add_argument('--img-size', nargs='+', type=int, default=[112, 112], help='image size')  # height, width
#     parser.add_argument('--batch-size', type=int, default=1, help='batch size')
#     opt = parser.parse_args()
#     opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand

 
#     print("=====> load pytorch checkpoint...")
#     checkpoint = torch.load(opt.weights, map_location=torch.device('cpu')) 
#     # net = GazeNet().to(device)


#     net = GazeNet(device)

#     net.load_state_dict(checkpoint[GazeNet(device)])

#     img = torch.zeros(1, 3, *opt.img_size).to(device)
#     print(img.shape)
#     landmarks, gaze = net.forward(img)
#     # f = opt.weights.replace('.pth.tar', '.onnx')  # filename

#     f = opt.weights.replace('.pth', '.onnx')  # filename
#     torch.onnx.export(net, img, f,export_params=True, verbose=False, opset_version=12, input_names=['inputs'])
#     # # ONNX export
#     try:
#         import onnx
#         from onnxsim import simplify

#         print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
#         # f = opt.weights.replace('.pth.tar', '.onnx')  # filename
#         f = opt.weights.replace('.pth', '.onnx')  # filename


#         torch.onnx.export(net, img, f, verbose=False, opset_version=11, input_names=['images'],
#                           output_names=['output'])

#         # Checks
#         onnx_model = onnx.load(f)  # load onnx model
#         model_simp, check = simplify(onnx_model)
#         assert check, "Simplified ONNX model could not be validated"
#         onnx.save(model_simp, f)
#         print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
#         print('ONNX export success, saved as %s' % f)
#     except Exception as e:
#         print('ONNX export failure: %s' % e)



"""
This code is used to convert the pytorch models into an onnx format models.
"""
import torch.onnx

from models.gazenet import GazeNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device', device)

input_img_size = 112  # define input size

model_path = "./models/weights/gazenet.pth"

checkpoint = torch.load(model_path)
net = GazeNet(device)
net.load_state_dict(checkpoint)
net.eval()
net.to("cuda")

model_name = model_path.split("/")[-1].split(".")[0]
model_path = f"models/onnx/{model_name}.onnx"

dummy_input = torch.randn(1, 3, 112, 112).to("cuda")

torch.onnx.export(net, dummy_input, model_path, export_params=True, 
verbose=False, input_names=['input'], output_names=['pose', 'landms'])



# torch_out = torch.onnx._export(net, inputs, output_onnx, export_params=True, verbose=False,
#                                input_names=input_names, output_names=output_names)