#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
from pathlib import Path

from psychopy.alerts import alert
from psychopy import logging
from psychopy.experiment.components import (
    BaseComponent, BaseDeviceComponent, Param, _translate, getInitVals
)
from psychopy.preferences import prefs
from psychopy.experiment.components.microphone import MicrophoneDeviceBackend
from psychopy.experiment.devices import DeviceBackend
from psychopy.tools import stringtools as st, systemtools as syst, audiotools as at


class CameraComponent(BaseDeviceComponent):
    """
    This component provides a way to use the webcam to record participants during an experiment.

    该组件用于在实验过程中使用摄像头记录被试的视频。

    **Note / 注意：**
    - For online experiments, the browser will notify participants to allow use of webcam
      before the start of the task.
    - 在线实验时，浏览器会在实验开始前请求访问摄像头的权限。

    When recording via webcam, specify the starting time relative to the start of the routine
    (see `start` below) and a stop time (= duration in seconds). A blank duration evaluates to
    recording for 0.000s.

    使用摄像头录制时，可以设置相对当前 Routine 的开始时间（“start”），以及停止时间或持续时间（秒）。
    若持续时间留空，则默认录制时长为 0.000 秒。

    The resulting video files are saved in .mp4 format if recorded locally and saved in .webm
    if recorded online. There will be one file per recording. The files appear in a new folder
    within the data directory in a folder called data_cam_recorded. The file names include the
    unix (epoch) time of the onset of the recording with milliseconds, e.g.::

        recording_cam_2022-06-16_14h32.42.064.mp4

    本地录制的视频文件为 .mp4，在线实验为 .webm。每次录制对应一个文件，
    存放在 data 目录下的 data_cam_recorded 文件夹中，文件名包含开始录制时的时间戳。

    This modified version additionally supports running an ONNX model (e.g. emotion-ferplus-8)
    on camera frames during the experiment (local PsychoPy only). It allows:

    这个增强版组件还支持在本地实验中对摄像头图像运行 ONNX 模型（例如 emotion-ferplus-8），并支持：

    - Optional dlib-based face detection / 可选 dlib 人脸检测
    - Configurable ONNX input shape / 可配置 ONNX 输入张量 shape
    - Regression or classification mode / 支持回归或分类两种任务模式
    - External JSON label file for classification / 分类任务可使用外部 JSON 标签文件
    - Flexible saving of model outputs (psydat + CSV/H5) / 模型输出可保存到 psydat，并可选保存到 CSV/H5
    - ONNX Runtime backend provider selection / 可配置 ONNX Runtime 后端 providers
    """

    categories = ['AI']
    targets = ["PsychoPy"]
    version = "2022.2.0"
    iconFile = Path(__file__).parent / 'webcam.png'
    tooltip = _translate(
        'AI Camera: Use webcam, optional ONNX model, face detection and logging. '
        '（AI 摄像头：支持摄像头录制，可选 ONNX 模型推理、人脸检测和结果记录。）'
    )
    beta = True
    deviceClasses = ["psychopy.hardware.camera.CameraDevice"]
    legacyParams = [
        # old device setup params, no longer needed as this is handled by DeviceManager
        "cameraLib",
        "device",
        "deviceManual",
        "frameRate",
        "frameRateManual",
        "mic",
        "micChannels",
        "micMaxRecSize",
        "micSampleRate",
        "resolution",
        "resolutionManual"
    ]

    def __init__(
            # Basic
            self, exp, parentName,
            name='cam',
            startType='time (s)', startVal='0', startEstim='',
            stopType='duration (s)', stopVal='', durationEstim='',
            # Device
            deviceLabel="",
            # audio
            micDeviceLabel="",
            # Data
            saveFile=True,
            saveStartStop=True, syncScreenRefresh=False,
            # Testing
            disabled=False,
            # === ONNX / Emotion / AI model support ===
            enableEmotion=False,
            emotionModelPath="",
            # legacy / kept for compatibility
            outputFileType="mp4",
            codec="h263",
            mic=None,
            channels='auto',
            sampleRate='DVD Audio (48kHz)',
            maxSize=24000,
            cameraLib="ffpyplayer",
            device="default",
            resolution="",
            frameRate="",
            deviceManual="",
            resolutionManual="",
            frameRateManual="",
    ):
        # Initialise superclass
        super(CameraComponent, self).__init__(
            exp, parentName,
            name=name,
            startType=startType, startVal=startVal, startEstim=startEstim,
            stopType=stopType, stopVal=stopVal, durationEstim=durationEstim,
            # Device
            deviceLabel=deviceLabel,
            # Data
            saveStartStop=saveStartStop, syncScreenRefresh=syncScreenRefresh,
            # Testing
            disabled=disabled,
        )
        # Mark as type
        self.type = 'Camera'
        # Store exp references
        self.exp = exp
        self.parentName = parentName

        # Add requirement imports to generated script
        self.exp.requireImport(importName="camera", importFrom="psychopy.hardware")
        self.exp.requireImport(importName="microphone", importFrom="psychopy.sound")

        # --- Device / order params ---
        self.order += [
            "deviceLabel",
            "enableEmotion",
            "emotionModelPath",
            "enableFace",
            "onnxInputShape",
            "inferMode",
            "labelFile",
            "onnxProviders",
            "saveInfer",
            "saveInferPath",
        ]

        # label to refer to mic device by
        self.params['micDeviceLabel'] = Param(
            micDeviceLabel, valType="device", inputType="device", categ="Device",
            allowedVals=[MicrophoneDeviceBackend],
            label=_translate("Microphone device / 麦克风设备"),
            hint=_translate(
                "The named device from Device Manager to use for this Component.\n"
                "在 Device Manager 中选择要用于本组件的麦克风设备。"
            )
        )

        # === Emotion / ONNX model related params ===
        self.params['enableEmotion'] = Param(
            enableEmotion, valType='bool', inputType="bool", categ="Data",
            label=_translate("Run ONNX model? / 是否运行 ONNX 模型？"),
            hint=_translate(
                "If enabled, run an ONNX model (e.g. emotion-ferplus-8) on camera "
                "frames and log the model output (local PsychoPy only).\n"
                "勾选后，会在本地实验中对摄像头帧运行 ONNX 模型（如 emotion-ferplus-8），"
                "并将模型输出写入数据文件。"
            )
        )

        self.params['emotionModelPath'] = Param(
            emotionModelPath, valType='str', inputType="file", categ="Data",
            label=_translate("ONNX model file / ONNX 模型文件"),
            hint=_translate(
                "Path to an ONNX model file, e.g. emotion-ferplus-8.onnx.\n"
                "ONNX 模型文件路径，例如 emotion-ferplus-8.onnx。"
            )
        )

        # === Face detection (dlib) ===
        self.params['enableFace'] = Param(
            False, valType='bool', inputType="bool", categ="Data",
            label=_translate("Use face detection (dlib)? / 使用 dlib 人脸检测？"),
            hint=_translate(
                "If enabled, use dlib frontal face detector before model inference.\n"
                "勾选后，将在模型推理前使用 dlib 的正面人脸检测器，只对人脸区域进行推理。"
            )
        )

        # === ONNX input shape ===
        self.params['onnxInputShape'] = Param(
            "1,1,64,64", valType='str', inputType="str", categ="Data",
            label=_translate("ONNX input shape / ONNX 输入张量形状"),
            hint=_translate(
                "Input tensor shape as N,C,H,W, e.g. 1,1,64,64 or 1,3,224,224.\n"
                "ONNX 模型输入张量的形状，格式为 N,C,H,W，例如 1,1,64,64 或 1,3,224,224。"
            )
        )

        # === Model output mode: classification or regression ===
        self.params['inferMode'] = Param(
            "classification", valType='code', inputType="choice", categ="Data",
            allowedVals=["classification", "regression"],
            label=_translate("Model output mode / 模型输出模式"),
            hint=_translate(
                "\"classification\": argmax + label mapping (need JSON file).\n"
                "\"regression\": use raw ONNX output (list/array).\n"
                "classification：使用 argmax 与标签文件做分类输出（需要 JSON 标签文件）；\n"
                "regression：直接使用 ONNX 原始输出（数组或列表）。"
            )
        )

        # === Classification label JSON ===
        self.params['labelFile'] = Param(
            "", valType='str', inputType="file", categ="Data",
            label=_translate("Class label JSON / 分类标签 JSON 文件"),
            hint=_translate(
                "JSON file mapping index (string) -> label, used for classification.\n"
                "用于分类模式的 JSON 标签文件，键为索引（字符串），值为类别名称。"
            )
        )

        # === Save inference outputs ===
        self.params['saveInfer'] = Param(
            True, valType='bool', inputType="bool", categ="Data",
            label=_translate("Save inference result to data file? / 将推理结果保存到数据文件？"),
            hint=_translate(
                "If enabled, predicted values will be stored in the PsychoPy data file (psydat/csv).\n"
                "勾选后，模型推理结果会写入 PsychoPy 的数据文件（psydat/csv）。"
            )
        )

        self.params['saveInferPath'] = Param(
            "", valType='str', inputType="file", categ="Data",
            label=_translate("Extra output file (CSV/H5) / 额外输出文件（CSV/H5）"),
            hint=_translate(
                "Optional extra file path. If given, inference results will also be appended to this CSV/H5 file.\n"
                "可选：额外的输出文件路径。如果填写，将把每帧推理结果追加写入该 CSV/H5 文件。"
            )
        )

        # === ONNX Runtime providers ===
        self.params['onnxProviders'] = Param(
            "CPUExecutionProvider", valType='str', inputType="str", categ="Data",
            label=_translate("ONNX Runtime providers / ONNX 运行后端 providers"),
            hint=_translate(
                "Comma-separated provider names, e.g. CUDAExecutionProvider,CPUExecutionProvider.\n"
                "ONNX Runtime 后端 providers，逗号分隔，例如：CUDAExecutionProvider,CPUExecutionProvider。"
            )
        )

        # --- Data params for video recording ---
        msg = _translate("Save webcam output to a file? / 是否将摄像头视频保存到文件？")
        self.params['saveFile'] = Param(
            saveFile, valType='bool', inputType="bool", categ="Data",
            hint=msg,
            label=_translate("Save file? / 保存视频文件？")
        )

    @staticmethod
    def setupMicNameInInits(inits):
        # substitute component name + "Microphone" for mic device name if blank
        if not inits['micDeviceLabel']:
            # if deviceName exists but is blank, use component name
            inits['micDeviceLabel'].val = inits['name'].val + "Microphone"
            inits['micDeviceLabel'].valType = 'str'
        # make a code version of mic device name
        inits['micDeviceLabelCode'] = copy.copy(inits['micDeviceLabel'])
        inits['micDeviceLabelCode'].valType = "code"

    def writeRoutineStartCode(self, buff):
        # nothing special at routine start
        pass

    def writeStartCode(self, buff):
        inits = getInitVals(self.params)
        # Use filename with a suffix to store recordings
        code = (
            "# make folder to store recordings from %(name)s\n"
            "# 为 %(name)s 创建录制文件夹\n"
            "%(name)sRecFolder = filename + '_%(name)s_recorded'\n"
            "if not os.path.isdir(%(name)sRecFolder):\n"
            "    os.mkdir(%(name)sRecFolder)\n"
        )
        buff.writeIndentedLines(code % inits)

    def writeInitCode(self, buff):
        inits = getInitVals(self.params, "PsychoPy")

        # if specified, get camera from device manager
        code = (
            "import os \n"
            "%(name)s = camera.Camera(\n"
            "    win=win,\n"
            "    device=%(deviceLabel)s,\n"
            "    mic=%(micDeviceLabel)s,\n"
            ")\n"
        )
        buff.writeIndentedLines(code % inits)

        if self.params['saveFile']:
            code = (
                "# connect camera save method to experiment handler so it's called when data saves\n"
                "# 将摄像头的保存方法连接到实验对象，在保存数据时一并保存视频\n"
                "thisExp.connectSaveMethod(%(name)s.save, os.path.join(%(name)sRecFolder, '_recovered.mp4'))\n"
            )
            buff.writeIndentedLines(code % inits)

        # === ONNX / emotion model init (only if enabled) ===
        if self.params['enableEmotion']:
            code = (
                "\n"
                "# --- setup ONNX model and helpers for %(name)s ---\n"
                "# 为 %(name)s 初始化 ONNX 模型和相关工具\n"
                "import numpy as np\n"
                "import onnxruntime as ort\n"
                "import json\n"
                "from PIL import Image\n"
                "import dlib\n"
                "\n"
                "# ONNX Runtime providers / 后端 providers\n"
                "%(name)s_providers = [p.strip() for p in %(onnxProviders)s.split(',') if p.strip()]\n"
                "if not %(name)s_providers:\n"
                "    %(name)s_providers = ['CPUExecutionProvider']\n"
                "%(name)s_session = ort.InferenceSession(%(emotionModelPath)s, providers=%(name)s_providers)\n"
                "%(name)s_input_name = %(name)s_session.get_inputs()[0].name\n"
                "\n"
                "# parse input shape N,C,H,W / 解析输入张量形状 N,C,H,W\n"
                "%(name)s_input_shape = [int(x) for x in %(onnxInputShape)s.split(',')]\n"
                "\n"
                "# face detection flag & detector / 人脸检测开关和检测器\n"
                "%(name)s_use_face = %(enableFace)s\n"
                "if %(name)s_use_face:\n"
                "    %(name)s_detector = dlib.get_frontal_face_detector()\n"
                "\n"
                "# inference mode: classification vs regression / 推理模式\n"
                "%(name)s_mode = %(inferMode)s \n"
                "# classification labels / 分类标签\n"
                "%(name)s_labels = None\n"
                "if %(name)s_mode == 'classification' and %(labelFile)s:\n"
                "    with open(%(labelFile)s, 'r', encoding='utf8') as f:\n"
                "        %(name)s_labels = json.load(f)\n"
                "\n"
                "# extra save path / 额外保存路径\n"
                "%(name)s_extra_save_path = %(saveInferPath)s\n"
            )
            buff.writeIndentedLines(code % inits)

    def writeInitCodeJS(self, buff):
        inits = getInitVals(self.params, target="PsychoJS")

        # Write code (no ONNX on JS side; webcam only)
        code = (
            "%(name)s = new hardware.Camera({\n"
            "    name:'%(name)s',\n"
            "    win: psychoJS.window,\n"
            "});\n"
            "// Get permission from participant to access their camera\n"
            "// 获取摄像头权限\n"
            "await %(name)s.authorize();\n"
            "// Switch on %(name)s\n"
            "// 打开摄像头\n"
            "await %(name)s.open();\n"
            "\n"
        )
        buff.writeIndentedLines(code % inits)

    def writeFrameCode(self, buff):
        # start webcam at component start
        indented = self.writeStartTestCode(buff)
        if indented:
            code = (
                "# start %(name)s recording\n"
                "# 开始 %(name)s 录制\n"
                "%(name)s.record()\n"
            )
            buff.writeIndentedLines(code % self.params)
        buff.setIndentLevel(-indented, relative=True)

        # update any params while active
        indented = self.writeActiveTestCode(buff)
        if indented:
            # 基本：轮询摄像头
            code = (
                "# get current frame data from camera\n"
                "# 从摄像头读取当前帧\n"
                "%(name)s.poll()\n"
            )
            buff.writeIndentedLines(code % self.params)

            # === ONNX model per-frame inference (if enabled) ===
            if self.params['enableEmotion']:
                code = (
                    "# --- ONNX model inference for %(name)s ---\n"
                    "# 对最新摄像头帧运行 ONNX 模型推理\n"
                    "frame_info = %(name)s.lastFrame  # (frame, pts, streamTime) or None\n"
                    "if frame_info is not None:\n"
                    "    frame = frame_info[0]\n"
                    "    # ffpyplayer frame -> numpy array (H, W, 3), uint8\n"
                    "    mv = frame.to_memoryview()[0].memview\n"
                    "    img_array = np.frombuffer(mv, dtype=np.uint8)\n"
                    "    w, h = frame.get_size()  # ffpyplayer: (width, height)\n"
                    "    img_array = img_array.reshape(h, w, 3)\n"
                    "\n"
                    "    # Optional dlib face detection / 可选 dlib 人脸检测\n"
                    "    if %(name)s_use_face:\n"
                    "        gray = img_array[:, :, 0]\n"
                    "        faces = %(name)s_detector(gray)\n"
                    "        if len(faces) > 0:\n"
                    "            f = faces[0]\n"
                    "            x, y, fw, fh = f.left(), f.top(), f.width(), f.height()\n"
                    "            # clip to bounds / 裁剪到图像边界\n"
                    "            x = max(0, x); y = max(0, y)\n"
                    "            fw = max(1, fw); fh = max(1, fh)\n"
                    "            img_array = img_array[y:y+fh, x:x+fw]\n"
                    "\n"
                    "    # Resize & format according to ONNX input shape / 按输入 shape 预处理\n"
                    "    N, C, H, W = %(name)s_input_shape\n"
                    "    img = Image.fromarray(img_array, mode='RGB').resize((W, H))\n"
                    "\n"
                    "    if C == 1:\n"
                    "        img = img.convert('L')\n"
                    "        input_data = np.asarray(img, dtype=np.float32)[None, None, :, :]\n"
                    "    else:\n"
                    "        arr = np.asarray(img, dtype=np.float32)\n"
                    "        # HWC -> CHW\n"
                    "        input_data = arr.transpose(2, 0, 1)[None, :, :, :]\n"
                    "\n"
                    "    # Run ONNX session / 运行 ONNX 推理\n"
                    "    outputs = %(name)s_session.run(None, {%(name)s_input_name: input_data})\n"
                    "    out = outputs[0]\n"
                    "\n"
                    "    # Post-process according to mode / 根据模式做后处理\n"
                    "    if %(name)s_mode == 'classification' and %(name)s_labels is not None:\n"
                    "        # assume first dim is batch / 假定第 1 维为 batch\n"
                    "        vec = out[0]\n"
                    "        idx = int(np.argmax(vec))\n"
                    "        # labels may be list or dict / 标签可以是 list 或 dict\n"
                    "        label = None\n"
                    "        if isinstance(%(name)s_labels, dict):\n"
                    "            # try string key first\n"
                    "            label = %(name)s_labels.get(str(idx), %(name)s_labels.get(idx, idx))\n"
                    "        elif isinstance(%(name)s_labels, (list, tuple)) and 0 <= idx < len(%(name)s_labels):\n"
                    "            label = %(name)s_labels[idx]\n"
                    "        else:\n"
                    "            label = idx\n"
                    "        pred_value = label\n"
                    "    else:\n"
                    "        # regression or no labels / 回归或无标签\n"
                    "        pred_value = out.tolist()\n"
                    "\n"
                    "    # Save to PsychoPy data file / 写入 PsychoPy 数据\n"
                    "    if %(saveInfer)s:\n"
                    "        thisExp.addData('%(name)s_infer', pred_value)\n"
                    "\n"
                    "    # Optional extra file saving / 可选额外文件保存\n"
                    "    if %(name)s_extra_save_path:\n"
                    "        try:\n"
                    "            import os\n"
                    "            # simple CSV-style append / 简单按行追加写入\n"
                    "            with open(%(name)s_extra_save_path, 'a', encoding='utf8') as f:\n"
                    "                f.write(str(pred_value) + '\\n')\n"
                    "        except Exception as e:\n"
                    "            # log but do not crash / 记录错误但不中止实验\n"
                    "            logging.error('Failed to save ONNX inference result: ' + str(e))\n"
                )
                buff.writeIndentedLines(code % self.params)

        buff.setIndentLevel(-indented, relative=True)

        # stop webcam at component stop
        indented = self.writeStopTestCode(buff)
        if indented:
            code = (
                "# stop %(name)s recording\n"
                "# 停止 %(name)s 录制\n"
                "%(name)s.stop()\n"
            )
            buff.writeIndentedLines(code % self.params)
        buff.setIndentLevel(-indented, relative=True)

    def writeFrameCodeJS(self, buff):
        # Start webcam at component start
        indent = self.writeStartTestCodeJS(buff)
        if indent:
            code = (
                "await %(name)s.record();\n"
            )
            buff.writeIndentedLines(code % self.params)
            buff.setIndentLevel(-indent, relative=True)
            code = (
                "};\n"
            )
            buff.writeIndentedLines(code)

        # Stop webcam at component stop
        indent = self.writeStopTestCodeJS(buff)
        if indent:
            code = (
                "await %(name)s.stop();\n"
            )
            buff.writeIndentedLines(code % self.params)
            buff.setIndentLevel(-indent, relative=True)
            code = (
                "};\n"
            )
            buff.writeIndentedLines(code)

    def writeRoutineEndCode(self, buff):
        code = (
            "# Make sure %(name)s has stopped recording\n"
            "# 确保 %(name)s 已经停止录制\n"
            "if %(name)s.status == STARTED:\n"
            "    %(name)s.stop()\n"
        )
        buff.writeIndentedLines(code % self.params)
        if self.params['saveFile']:
            # 注意这里用 %%s 来避免和上面的 %(name)s 冲突
            code = (
                "# Save %(name)s recording\n"
                "# 保存 %(name)s 的录制文件\n"
                "%(name)sFilename = os.path.join(\n"
                "    %(name)sRecFolder,\n"
                "    'recording_%(name)s_%%s.mp4' %% data.utils.getDateStr()\n"
                ")\n"
                "%(name)s.save(%(name)sFilename)\n"
                "thisExp.currentLoop.addData('%(name)s.clip', %(name)sFilename)\n"
            )
            buff.writeIndentedLines(code % self.params)

    def writeRoutineEndCodeJS(self, buff):
        code = (
            "// Ensure that %(name)s is stopped\n"
            "// 确保摄像头录制已停止\n"
            "if (%(name)s.status === PsychoJS.Status.STARTED) {\n"
            "    await %(name)s.stop();\n"
            "}\n"
        )
        buff.writeIndentedLines(code % self.params)
        if self.params['saveFile']:
            code = (
                "// Save %(name)s recording\n"
                "// 保存摄像头录制文件\n"
                "let %(name)sFilename = `recording_%(name)s_${util.MonotonicClock.getDateStr()}`;\n"
                "await %(name)s.save({\n"
                "    tag: %(name)sFilename,\n"
                "    waitForCompletion: true,\n"
                "    showDialog: true,\n"
                "    dialogMsg: \"Please wait a few moments while the video is uploading to the server...\"\n"
                "});\n"
                "psychoJS.experiment.addData('%(name)s.clip', %(name)sFilename);\n"
            )
            buff.writeIndentedLines(code % self.params)

    def writeExperimentEndCode(self, buff):
        code = (
            "# Switch off %(name)s\n"
            "# 关闭摄像头设备\n"
            "%(name)s.close()\n"
        )
        buff.writeIndentedLines(code % self.params)

    def writeExperimentEndCodeJS(self, buff):
        code = (
            "// Switch off %(name)s\n"
            "// 关闭摄像头设备\n"
            "%(name)s.close();\n"
        )
        buff.writeIndentedLines(code % self.params)


class CameraDeviceBackend(DeviceBackend):
    # name of this backend to display in Device Manager
    backendLabel = "Camera"
    # class of the device which this backend corresponds to
    deviceClass = "psychopy.hardware.camera.CameraDevice"
    # icon to show in device manager
    icon = "light/webcam.png"

    def writeDeviceCode(self, buff):
        # write base setup
        self.writeBaseDeviceCode(buff, close=False)
        # add params
        code = (
            "    frameRate=%(frameRate)s,\n"
            "    frameSize=%(frameSize)s\n"
            ")"
        )
        buff.writeIndentedLines(code % self.params)

    def getParams(self):
        from psychopy.hardware.camera import CameraDevice

        # get supported resolutions and framerates
        resolutions = set()
        frameRates = set()
        for profile in CameraDevice.getAvailableDevices(best=False):
            if profile['deviceName'] == self.profile['deviceName']:
                resolutions.add(profile['frameSize'])
                frameRates.add(profile['frameRate'])

        order = [
            'frameSize',
            'frameRate',
        ]
        params = {}

        self.params['frameSize'] = Param(
            "", valType='list', inputType="choice",
            allowedVals=[""] + list(sorted(resolutions)),
            allowedLabels=["Default / 默认"] + list(sorted(resolutions)),
            hint=_translate(
                "Resolution (w x h) to record to, leave blank to use device default.\n"
                "录制时使用的分辨率（宽 x 高），留空则使用设备默认分辨率。"
            ),
            label=_translate("Resolution / 分辨率")
        )
        params['frameRate'] = Param(
            None, valType='int', inputType="choice",
            allowedVals=[""] + list(frameRates),
            allowedLabels=["Default / 默认"] + list(frameRates),
            hint=_translate(
                "Frame rate (frames per second) to record at, leave blank to use device default.\n"
                "录制时使用的帧率（每秒帧数），留空则使用设备默认帧率。"
            ),
            label=_translate("Frame rate / 帧率")
        )

        return params, order


# register backend with Component
CameraComponent.registerBackend(CameraDeviceBackend)


if __name__ == "__main__":
    pass
