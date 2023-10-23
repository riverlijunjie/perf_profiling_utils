#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import sys

import cv2
import numpy as np
from openvino.preprocess import PrePostProcessor, ResizeAlgorithm
from openvino.runtime import Core, Layout, Type

SOS_INDEX = 0
EOS_INDEX = 1
MAX_SEQ_LEN = 28

def _asarray_validated(a, check_finite=True,
                       sparse_ok=False, objects_ok=False, mask_ok=False,
                       as_inexact=False):
    if not sparse_ok:
        import scipy.sparse
        if scipy.sparse.issparse(a):
            msg = ('Sparse matrices are not supported by this function. '
                   'Perhaps one of the scipy.sparse.linalg functions '
                   'would work instead.')
            raise ValueError(msg)
    if not mask_ok:
        if np.ma.isMaskedArray(a):
            raise ValueError('masked arrays are not supported')
    toarray = np.asarray_chkfinite if check_finite else np.asarray
    a = toarray(a)
    if not objects_ok:
        if a.dtype is np.dtype('O'):
            raise ValueError('object arrays are not supported')
    if as_inexact:
        if not np.issubdtype(a.dtype, np.inexact):
            a = toarray(a, dtype=np.float_)
    return a

def softmax(x, axis=None):
    x = _asarray_validated(x, check_finite=False)
    x_max = np.amax(x, axis=axis, keepdims=True)
    exp_x_shifted = np.exp(x - x_max)
    return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)

def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)

    raw_output_message = True
    tr_threshold = 0.5
    trd_output_cur_hidden='hidden'
    alphabet='  abcdefghijklmnopqrstuvwxyz0123456789'
    trd_input_prev_symbol='prev_symbol'
    trd_input_prev_hidden='prev_hidden'
    trd_input_encoder_outputs='encoder_outputs'
    trd_output_symbols_distr='output'

    TEST_0001 = True #  True, False
    TEST_0005 = not TEST_0001
    # Text-spotting-0001 model
    if TEST_0001:
        # ww10 fp16
        MODEL_ROOT="/home/xiping/mydisk2_2T/local_accuary_checher_docker_example/no_docker/nfs_share/ww10_weekly_23.0.0-9926-63d282fd73c-API2.0"
        model_path_detector = f"{MODEL_ROOT}/text-spotting-0001-detector/onnx/onnx/FP16/1/dldt/text-spotting-0001-detector.xml"
        model_path_encoder = f"{MODEL_ROOT}/text-spotting-0001-recognizer-encoder/onnx/onnx/FP16/1/dldt/text-spotting-0001-recognizer-encoder.xml"
        model_path_decoder = f"{MODEL_ROOT}/text-spotting-0001-recognizer-decoder/onnx/onnx/FP16/1/dldt/text-spotting-0001-recognizer-decoder.xml"
        # # ww09 fp16
        # MODEL_ROOT="/home/xiping/mydisk2_2T/local_accuary_checher_docker_example/no_docker/nfs_share/ww09_weekly_23.0.0-9828-4fd38844a28-API2.0-FP16"
        # model_path_detector = f"{MODEL_ROOT}/text-spotting-0001-detector/onnx/onnx/FP16/1/dldt/text-spotting-0001-detector.xml"
        # model_path_encoder = f"{MODEL_ROOT}/text-spotting-0001-recognizer-encoder/onnx/onnx/FP16/1/dldt/text-spotting-0001-recognizer-encoder.xml"
        # model_path_decoder = f"{MODEL_ROOT}/text-spotting-0001-recognizer-decoder/onnx/onnx/FP16/1/dldt/text-spotting-0001-recognizer-decoder.xml"
        # # ww09 fp32
        # MODEL_ROOT="/home/xiping/mydisk2_2T/local_accuary_checher_docker_example/no_docker/nfs_share/ww09_weekly_23.0.0-9828-4fd38844a28-API2.0-FP32"
        # model_path_detector = f"{MODEL_ROOT}/text-spotting-0001-detector/onnx/onnx/FP32/1/dldt/text-spotting-0001-detector.xml"
        # model_path_encoder = f"{MODEL_ROOT}/text-spotting-0001-recognizer-encoder/onnx/onnx/FP32/1/dldt/text-spotting-0001-recognizer-encoder.xml"
        # model_path_decoder = f"{MODEL_ROOT}/text-spotting-0001-recognizer-decoder/onnx/onnx/FP32/1/dldt/text-spotting-0001-recognizer-decoder.xml"
    # Text-spotting-0005 model
    if TEST_0005:
        MODEL_ROOT="/home/xiping/mydisk2_2T/local_accuary_checher_docker_example/no_docker/nfs_share/ww09_weekly_23.0.0-9828-4fd38844a28-API2.0-FP32"
        model_path_detector=f"{MODEL_ROOT}/text-spotting-0005-detector/onnx/onnx/FP32/1/dldt/text-spotting-0005-detector.xml"
        model_path_encoder=f"{MODEL_ROOT}/text-spotting-0005-recognizer-encoder/onnx/onnx/FP32/1/dldt/text-spotting-0005-recognizer-encoder.xml"
        model_path_decoder=f"{MODEL_ROOT}/text-spotting-0005-recognizer-decoder/onnx/onnx/FP32/1/dldt/text-spotting-0005-recognizer-decoder.xml"
        MODEL_ROOT="/home/xiping/mydisk2_2T/local_accuary_checher_docker_example/no_docker/nfs_share/ww09_weekly_23.0.0-9828-4fd38844a28-API2.0-FP16"
        model_path_detector=f"{MODEL_ROOT}/text-spotting-0005-detector/onnx/onnx/FP16/1/dldt/text-spotting-0005-detector.xml"
        model_path_encoder=f"{MODEL_ROOT}/text-spotting-0005-recognizer-encoder/onnx/onnx/FP16/1/dldt/text-spotting-0005-recognizer-encoder.xml"
        model_path_decoder=f"{MODEL_ROOT}/text-spotting-0005-recognizer-decoder/onnx/onnx/FP16/1/dldt/text-spotting-0005-recognizer-decoder.xml"

    image_path = "523150613.jpg"
    image_path="/home/xiping/mydisk2_2T/local_accuary_checher_docker_example/no_docker/nfs_share_2/omz-validation-datasets/ICDAR15_DET/ch4_test_images/img_100.jpg"
    device_name = "CPU"
# --------------------------- Step 1. Initialize OpenVINO Runtime Core ------------------------------------------------
    log.info('Creating OpenVINO Runtime Core')
    core = Core()

# --------------------------- Step 2. Read a model --------------------------------------------------------------------
    log.info(f'Reading the model: {model_path_detector}')
    log.info(f'Reading the model: {model_path_encoder}')
    log.info(f'Reading the model: {model_path_decoder}')
    # (.xml and .bin files) or (.onnx file)
    model1 = core.read_model(model_path_detector)
    model_dec = core.read_model(model_path_decoder)
    model_enc = core.read_model(model_path_encoder)

# --------------------------- Step 3. Set up input --------------------------------------------------------------------
    # Read input image
    frame = cv2.imread(image_path)
    # Add N dimension
    # print("frame.shape=", frame.shape, frame.dtype)
    input_tensor = np.expand_dims(frame, 0)
    # print("input_tensor.shape=", input_tensor.shape)

# --------------------------- Step 4. Apply preprocessing -------------------------------------------------------------
    input_tensor_name = 'im_data' if TEST_0001 else 'image'
    try:
        n, c, h, w = model1.input(input_tensor_name).shape
        if n != 1:
            raise RuntimeError('Only batch 1 is supported by the demo application')
    except RuntimeError:
        raise RuntimeError('Demo supports only topologies with the following input tensor name: {}'.format(input_tensor_name))

    if TEST_0001:
        required_output_names = {'boxes','scores','classes','raw_masks','text_features'}
    else:
        required_output_names = {'boxes', 'labels', 'masks', 'text_features'}
    
    for output_tensor_name in required_output_names:
        try:
            model1.output(output_tensor_name)
        except RuntimeError:
            raise RuntimeError('Demo supports only topologies with the following output tensor names: {}'.format(
                ', '.join(required_output_names)))

# --------------------------- Step 5. Loading model to the device -----------------------------------------------------
    log.info('Loading the model to the plugin')
    compiled_model1 = core.compile_model(model1, device_name)
    compiled_model_dec = core.compile_model(model_dec, device_name)
    compiled_model_enc = core.compile_model(model_enc, device_name)

# --------------------------- Step 6. Create infer request and do inference synchronously -----------------------------
    log.info('Starting inference in synchronous mode')
    req1 = compiled_model1.create_infer_request()
    req_enc = compiled_model_enc.create_infer_request()
    text_enc_output_tensor = compiled_model_enc.outputs[0]
    req_dec = compiled_model_dec.create_infer_request()

    hidden_shape = model_dec.input("prev_hidden").shape
    text_dec_output_names = {"output", "hidden"}

# --------------------------- Step 7. Process output ------------------------------------------------------------------
    # cv2.imshow('frame', frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    scale_x = w / frame.shape[1]
    scale_y = h / frame.shape[0]
    input_image = cv2.resize(frame, (w, h))
    print("input_image=", input_image.shape)

    input_image_size = input_image.shape[:2]
    input_image = np.pad(input_image, ((0, h - input_image_size[0]),
                                        (0, w - input_image_size[1]),
                                        (0, 0)),
                            mode='constant', constant_values=0)
    
    # Change data layout from HWC to CHW.
    input_image = input_image.transpose((2, 0, 1))
    input_image = input_image.reshape((n, c, h, w)).astype(np.float32)

    MINUS_MEAN=False
    if MINUS_MEAN:
        img_hw_shape = [input_image.shape[2], input_image.shape[3]]
        print("img_hw_shape", img_hw_shape)
        mean_values = [123.675, 116.28, 103.53]
        b_mean = np.zeros(img_hw_shape)
        g_mean = np.zeros(img_hw_shape)
        r_mean = np.zeros(img_hw_shape)
        b_mean[:] = mean_values[2]
        g_mean[:] = mean_values[1]
        r_mean[:] = mean_values[0]
        input_image[0][0] = input_image[0][0] - b_mean
        input_image[0][1] = input_image[0][1] - g_mean
        input_image[0][2] = input_image[0][2] - r_mean
    
    req1.infer({input_tensor_name: input_image})
    outputs = {name: req1.get_tensor(name).data[:] for name in required_output_names}

    # Parse detection results of the current request
    # print("outputs=",outputs)
    if TEST_0001:
        boxes = outputs['boxes']
        scores = outputs['scores']
        classes = outputs['classes'].astype(np.uint32)
        raw_masks = outputs['raw_masks']
        text_features = outputs['text_features']
    else:
        boxes = outputs['boxes'][:, :4]
        scores = outputs['boxes'][:, 4]
        classes = outputs['labels'].astype(np.uint32)
        raw_masks = outputs['masks']
        text_features = outputs['text_features']

    # Filter out detections with low confidence.
    detections_filter = scores > 0.65
    scores = scores[detections_filter]
    classes = classes[detections_filter]
    boxes = boxes[detections_filter]
    raw_masks = raw_masks[detections_filter]
    text_features = text_features[detections_filter]

    log.info(f"Got: scores = {scores}")
    
    boxes[:, 0::2] /= scale_x
    boxes[:, 1::2] /= scale_y
    masks = []
    # for box, cls, raw_mask in zip(boxes, classes, raw_masks):
    #     mask = segm_postprocess(box, raw_mask, frame.shape[0], frame.shape[1])
    #     masks.append(mask)

    texts = []
    for feature in text_features:
        input_data = {'input': np.expand_dims(feature, axis=0)}
        feature = req_enc.infer(input_data)[text_enc_output_tensor]
        feature = np.reshape(feature, (feature.shape[0], feature.shape[1], -1))
        feature = np.transpose(feature, (0, 2, 1))

        hidden = np.zeros(hidden_shape)
        prev_symbol_index = np.ones((1,)) * SOS_INDEX

        text = ''
        text_confidence = 1.0
        for i in range(MAX_SEQ_LEN):
            req_dec.infer({
                trd_input_prev_symbol: np.reshape(prev_symbol_index, (1,)),
                trd_input_prev_hidden: hidden,
                trd_input_encoder_outputs: feature})
            decoder_output = {name: req_dec.get_tensor(name).data[:] for name in text_dec_output_names}
            symbols_distr = decoder_output[trd_output_symbols_distr]
            symbols_distr_softmaxed = softmax(symbols_distr, axis=1)[0]
            prev_symbol_index = int(np.argmax(symbols_distr, axis=1))
            text_confidence *= symbols_distr_softmaxed[prev_symbol_index]
            if prev_symbol_index == EOS_INDEX:
                break
            text += alphabet[prev_symbol_index]
            hidden = decoder_output[trd_output_cur_hidden]

        texts.append(text if text_confidence >= tr_threshold else '')

    # print("len(boxes)=", len(boxes))
    # print("scores=", scores)
    print("texts=", texts)
    # print("raw_output_message=", raw_output_message)
    if len(boxes) and raw_output_message:
        log.info('  -------------------------- Frame # --------------------------  ')
        log.info('  Class ID | Confidence |     XMIN |     YMIN |     XMAX |     YMAX ')
        for box, cls, score in zip(boxes, classes, scores):
            log.info('{:>10} | {:>10f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} '.format(cls, score, *box))

    return 0

if __name__ == '__main__':
    sys.exit(main())
