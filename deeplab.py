import colorsys
import copy
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from nets.deeplabv3_plus import DeepLab
from utils.utils import cvtColor, preprocess_input, resize_image, show_config


class DeeplabV3(object):
    _defaults = {
        "model_path":'/home/robot/Code/logs/UDTIRI/1basic.pth',
        "num_classes": 2,
        "backbone": "mobilenetv2",
        "input_shape": [512, 512],
        "downsample_factor": 8,
        "mix_type": 0,
        "cuda": False,
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        if self.num_classes <= 21:
            self.colors = [
                (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
                (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
                (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
                (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                (0, 64, 128), (128, 64, 128)
            ]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (
                int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)
            ), self.colors))

        self.generate()
        show_config(**self._defaults)

    def generate(self, onnx=False):
        self.net = DeepLab(
            num_classes=self.num_classes,
            backbone=self.backbone,
            downsample_factor=self.downsample_factor,
            pretrained=False
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.model_path, map_location=device)
        self.net.load_state_dict(state_dict, strict=False)
        self.net = self.net.eval()
        print(f'{self.model_path} model, and classes loaded.')

        if not onnx and self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    def get_segmentation_mask(self, image):
        image = cvtColor(image)
        orininal_h, orininal_w = np.array(image).shape[:2]
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(
            np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0
        )

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr = pr[
                int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)
            ]
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1).astype(np.uint8)

        return pr

    def get_red_black_image(self, image, pothole_class_index=1, color=(113, 0, 0)):
        pr = self.get_segmentation_mask(image)
        h, w = pr.shape
        result = np.zeros((h, w, 3), dtype=np.uint8)
        result[pr == pothole_class_index] = color
        return Image.fromarray(result)

    def detect_image(self, image, count=False, name_classes=None):
        image = cvtColor(image)
        old_img = copy.deepcopy(image)
        orininal_h, orininal_w = np.array(image).shape[:2]
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(
            np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0
        )

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr = pr[
                int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)
            ]
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)

        if count:
            classes_nums = np.zeros([self.num_classes])
            total_points_num = orininal_h * orininal_w
            for i in range(self.num_classes):
                num = np.sum(pr == i)
                classes_nums[i] = num

        if self.mix_type == 0:
            seg_img = np.reshape(
                np.array(self.colors, np.uint8)[np.reshape(pr, [-1])],
                [orininal_h, orininal_w, -1]
            )
            image = Image.fromarray(seg_img)
            image = Image.blend(old_img, image, 0.7)
        elif self.mix_type == 1:
            seg_img = np.reshape(
                np.array(self.colors, np.uint8)[np.reshape(pr, [-1])],
                [orininal_h, orininal_w, -1]
            )
            image = Image.fromarray(seg_img)
        else:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img)).astype('uint8')
            image = Image.fromarray(seg_img)

        return image

    def get_FPS(self, image, test_interval):
        image = cvtColor(image)
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(
            np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0
        )

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            self.net(images)

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                self.net(images)
        t2 = time.time()

        return (t2 - t1) / test_interval

    def convert_to_onnx(self, simplify, model_path):
        import onnx
        self.generate(onnx=True)

        im = torch.zeros(1, 3, *self.input_shape)
        torch.onnx.export(
            self.net,
            im,
            model_path,
            opset_version=12,
            training=torch.onnx.TrainingMode.EVAL,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["output"]
        )

        model_onnx = onnx.load(model_path)
        onnx.checker.check_model(model_onnx)

        if simplify:
            import onnxsim
            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check
            onnx.save(model_onnx, model_path)

    def get_miou_png(self, image):
        image = cvtColor(image)
        orininal_h, orininal_w = np.array(image).shape[:2]
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(
            np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0
        )

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr = pr[
                int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)
            ]
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)

        return Image.fromarray(pr.astype(np.uint8))
