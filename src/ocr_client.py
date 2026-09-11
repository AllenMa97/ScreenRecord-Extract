"""
OCR客户端

支持三种模式:
  1. API模式: 调用PaddleOCR AI Studio异步任务API(需配置api_key=TOKEN)
  2. 本地模式: 调用本机paddleocr(需安装paddlepaddle+paddleocr)
  3. Mock模式: 生成模拟OCR结果(用于测试, 无需API)

AI Studio任务API流程:
  1. POST {api_endpoint} 上传图片创建任务, 返回jobId
  2. GET {api_endpoint}/{jobId} 轮询任务状态(pending/running/done/failed)
  3. done后下载resultUrl.jsonUrl对应的jsonl并解析文本

使用前请在config.yaml中配置:
"""

import time
from typing import List

import numpy as np

from src.data_types import TextLine
from src.utils.image_utils import normalize_for_ocr
from src.utils.behavior_logger import BehaviorLogger


class OcrClient:
    """OCR识别客户端"""

    def __init__(self, config: dict, logger: BehaviorLogger):
        self._mode: str = config.get("mode", "api")
        self._api_endpoint: str = config.get("api_endpoint", "")
        self._api_key: str = config.get("api_key", "")
        self._model: str = config.get("model", "PP-OCRv6")
        self._poll_interval: float = config.get("poll_interval", 5)
        self._lang: str = config.get("lang", "ch")
        self._timeout: int = config.get("timeout", 30)
        # 429限流/5xx/网络错误 → 指数退避重试
        self._max_retries: int = 3
        self._retry_base_delay: float = 1.0
        self._logger = logger
        self._local_ocr = None  # 延迟初始化

        # 检查配置
        if self._mode == "api" and (not self._api_key or not self._api_endpoint):
            self._logger.log("ocr_client", "warning",
                             "API模式但未配置api_key/api_endpoint, 将使用mock模式 / "
                             "API mode but no api_key/api_endpoint, falling back to mock")
            self._mode = "mock"

    def _ensure_local_ocr(self) -> None:
        """延迟初始化本地PaddleOCR"""
        if self._local_ocr is None:
            try:
                from paddleocr import PaddleOCR
                self._local_ocr = PaddleOCR(use_angle_cls=True, lang=self._lang,
                                            show_log=False)
                self._logger.log("ocr_client", "local_init",
                                 "本地PaddleOCR初始化成功 / Local PaddleOCR initialized")
            except ImportError:
                self._logger.log("ocr_client", "warning",
                                 "paddleocr包未安装, 使用mock模式 / "
                                 "paddleocr not installed, using mock")
                self._mode = "mock"

    def _submit_job(self, img_bytes: bytes, frame_id: int) -> str:
        """提交OCR任务, 返回jobId"""
        import json
        import requests

        headers = {"Authorization": f"bearer {self._api_key}"}
        data = {
            "model": self._model,
            "optionalPayload": json.dumps({
                "useDocOrientationClassify": False,
                "useDocUnwarping": False,
                "useTextlineOrientation": False,
            }),
        }
        files = {"file": (f"frame_{frame_id}.jpg", img_bytes, "image/jpeg")}
        resp = requests.post(self._api_endpoint, headers=headers, data=data,
                             files=files, timeout=self._timeout)
        resp.raise_for_status()
        return resp.json()["data"]["jobId"]

    def _encode_jpg(self, roi_frame: np.ndarray) -> bytes:
        """编码ROI帧为JPEG字节"""
        import cv2
        _, encoded = cv2.imencode(".jpg", roi_frame,
                                  [cv2.IMWRITE_JPEG_QUALITY, 90])
        return encoded.tobytes()

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """是否为可重试错误(限流/服务端/网络)"""
        import requests

        if isinstance(exc, requests.HTTPError):
            status = exc.response.status_code if exc.response is not None else 0
            return status in (429, 500, 502, 503, 504)
        return isinstance(exc, (requests.ConnectionError, requests.Timeout))

    def _with_retry(self, fn, frame_id: int, what: str):
        """指数退避重试包装"""
        last_exc = None
        delay = self._retry_base_delay
        for attempt in range(self._max_retries + 1):
            try:
                return fn()
            except Exception as e:
                last_exc = e
                if not self._is_retryable(e):
                    raise
                if attempt >= self._max_retries:
                    break
                self._logger.log("ocr_client", "retry",
                                 f"{what}重试 {attempt + 1}/{self._max_retries} "
                                 f"(帧{frame_id}): {str(e)[:80]}")
                time.sleep(delay)
                delay *= 2
        raise last_exc

    def submit(self, roi_frame: np.ndarray, frame_id: int = 0,
               timestamp: float = 0.0) -> str:
        """
        提交OCR任务, 立即返回jobId(不等待结果)
        用于并行流水线: 提交后由轮询线程统一收集结果。
        非API模式返回None
        """
        if self._mode != "api":
            return None
        prepared = normalize_for_ocr(roi_frame)
        img_bytes = self._encode_jpg(prepared)
        return self._with_retry(
            lambda: self._submit_job(img_bytes, frame_id),
            frame_id, "OCR提交"
        )

    def poll_once(self, job_id: str, frame_id: int = 0,
                  timestamp: float = 0.0):
        """
        单次轮询任务状态
        pending/running → None(未完成, 下次再查)
        done → 文本行列表
        failed → 抛RuntimeError
        非API模式返回None
        """
        import requests

        if self._mode != "api":
            return None

        def _poll():
            headers = {"Authorization": f"bearer {self._api_key}"}
            resp = requests.get(f"{self._api_endpoint}/{job_id}",
                                headers=headers, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json().get("data", {})
            state = data.get("state", "")
            if state in ("pending", "running"):
                return None
            if state == "done":
                jsonl_url = data["resultUrl"]["jsonUrl"]
                jsonl_resp = requests.get(jsonl_url, timeout=self._timeout)
                jsonl_resp.raise_for_status()
                return self._parse_job_result(jsonl_resp.text, frame_id,
                                              timestamp)
            if state == "failed":
                raise RuntimeError(data.get("errorMsg",
                                            "OCR任务失败 / OCR job failed"))
            raise RuntimeError(f"未知任务状态 / Unknown job state: {state}")

        return self._with_retry(_poll, frame_id, "OCR轮询")

    def recognize(self, roi_frame: np.ndarray, frame_id: int = 0,
                  timestamp: float = 0.0) -> List[TextLine]:
        """
        识别ROI帧中的文本

        ROI = Region of Interest (感兴趣区域): 剔除状态栏/导航栏后真正有内容的屏幕区域。

        参数:
            roi_frame: ROI裁剪帧(原始分辨率)
            frame_id: 帧ID
            timestamp: 时间戳
        返回:
            文本行列表
        """
        prepared = normalize_for_ocr(roi_frame)

        if self._mode == "mock":
            return self._mock_recognize(prepared, frame_id, timestamp)
        elif self._mode == "local":
            self._ensure_local_ocr()
            if self._local_ocr is not None:
                return self._local_recognize(prepared, frame_id, timestamp)
            return self._mock_recognize(prepared, frame_id, timestamp)
        else:
            return self._api_recognize(prepared, frame_id, timestamp)

    def _api_recognize(self, frame: np.ndarray, frame_id: int,
                       timestamp: float) -> List[TextLine]:
        """通过AI Studio任务API识别(同步)"""
        try:
            job_id = self.submit(frame, frame_id, timestamp)
            if job_id is None:
                return []
            while True:
                lines = self.poll_once(job_id, frame_id, timestamp)
                if lines is not None:
                    break
                time.sleep(self._poll_interval)
        except Exception as e:
            self._logger.log("ocr_client", "api_error",
                             f"API调用失败 / API call failed: {e}",
                             frame_id=frame_id)
            return []

        self._logger.log("ocr_client", "recognize",
                         f"API识别: {len(lines)}行 / {len(lines)} lines",
                         frame_id=frame_id,
                         extra={"line_count": len(lines)})
        return lines

    def _local_recognize(self, frame: np.ndarray, frame_id: int,
                         timestamp: float) -> List[TextLine]:
        """通过本地PaddleOCR识别"""
        try:
            result = self._local_ocr.ocr(frame, cls=True)
        except Exception as e:
            self._logger.log("ocr_client", "local_error",
                             f"本地OCR失败 / Local OCR failed: {e}",
                             frame_id=frame_id)
            return []

        lines = []
        if result and result[0]:
            for item in result[0]:
                box_points = item[0]
                text = item[1][0]
                conf = float(item[1][1])

                xs = [p[0] for p in box_points]
                ys = [p[1] for p in box_points]
                x = int(min(xs))
                y = int(min(ys))
                w = int(max(xs) - min(xs))
                h = int(max(ys) - min(ys))

                if w > 0 and h > 0 and text.strip():
                    lines.append(TextLine(
                        text=text.strip(), bbox=(x, y, w, h),
                        confidence=conf, frame_id=frame_id, timestamp=timestamp
                    ))

        self._logger.log("ocr_client", "recognize",
                         f"本地识别: {len(lines)}行 / {len(lines)} lines",
                         frame_id=frame_id, extra={"line_count": len(lines)})
        return lines

    def _mock_recognize(self, frame: np.ndarray, frame_id: int,
                        timestamp: float) -> List[TextLine]:
        """
        生成模拟OCR结果(用于测试)

        基于图像的行投影生成模拟文本行, 模拟真实OCR的bbox分布。
        """
        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        h, w = gray.shape[:2]

        # 用行投影检测文本行位置
        row_sum = gray.sum(axis=1)
        threshold = row_sum.mean() * 0.9

        lines: List[TextLine] = []
        in_line = False
        line_start = 0
        margin = int(w * 0.05)

        for y in range(h):
            is_text_row = row_sum[y] < threshold
            if is_text_row and not in_line:
                in_line = True
                line_start = y
            elif not is_text_row and in_line:
                in_line = False
                line_h = y - line_start
                if line_h > 8:
                    mock_text = f"mock_line_{frame_id}_{line_start}"
                    lines.append(TextLine(
                        text=mock_text,
                        bbox=(margin, line_start, w - 2 * margin, line_h),
                        confidence=0.95, frame_id=frame_id, timestamp=timestamp
                    ))

        self._logger.log("ocr_client", "mock_recognize",
                         f"Mock识别: {len(lines)}行 / {len(lines)} lines",
                         frame_id=frame_id, extra={"line_count": len(lines)})
        return lines

    def _parse_job_result(self, jsonl_text: str, frame_id: int,
                          timestamp: float) -> List[TextLine]:
        """解析任务结果jsonl"""
        import json

        lines: List[TextLine] = []
        for raw_line in jsonl_text.splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            result = obj.get("result", obj) if isinstance(obj, dict) else {}
            if not isinstance(result, dict):
                continue
            ocr_results = result.get("ocrResults", [])
            if not isinstance(ocr_results, list):
                ocr_results = [ocr_results]
            for res in ocr_results:
                if not isinstance(res, dict):
                    continue
                for text, box, conf in zip(*self._extract_res(res)):
                    text = text.strip()
                    bbox = self._normalize_box(box)
                    if not text or bbox is None:
                        continue
                    lines.append(TextLine(
                        text=text, bbox=bbox, confidence=conf,
                        frame_id=frame_id, timestamp=timestamp
                    ))
        return lines

    def _extract_res(self, res: dict):
        """从单个ocrResults项提取(文本, bbox, 置信度)列表"""
        texts, boxes, confs = [], [], []
        pruned = res.get("prunedResult")
        if isinstance(pruned, dict):
            rec_texts = pruned.get("rec_texts") or pruned.get("recTexts") or []
            rec_boxes = pruned.get("rec_boxes") or pruned.get("recBoxes") or []
            rec_scores = pruned.get("rec_scores") or pruned.get("recScores") or []
            for i, text in enumerate(rec_texts):
                box = rec_boxes[i] if i < len(rec_boxes) else None
                conf = rec_scores[i] if i < len(rec_scores) else 1.0
                texts.append(text)
                boxes.append(box)
                confs.append(float(conf))
        else:
            text = res.get("text") or res.get("words") or ""
            if text:
                box = (res.get("rec_boxes") or res.get("recBoxes") or
                       res.get("points") or res.get("bbox") or
                       res.get("location") or res.get("text_box_position"))
                conf = res.get("confidence", res.get("score", 1.0))
                texts.append(text)
                boxes.append(box)
                confs.append(float(conf))
        return texts, boxes, confs

    def _normalize_box(self, box):
        """
        归一化bbox为(x, y, w, h)
        兼容三种格式:
          [x, y, w, h]                     (rec_boxes)
          [x1, y1, x2, y2, x3, y3, x4, y4] (扁平四点
          [[x, y], [x, y], [x, y], [x, y]] (四点坐标
        """
        if not box:
            return None
        if isinstance(box, dict):
            if {"left", "top", "width", "height"}.issubset(box):
                try:
                    x, y = int(box["left"]), int(box["top"])
                    w, h = int(box["width"]), int(box["height"])
                    return (x, y, w, h) if w > 0 and h > 0 else None
                except (TypeError, ValueError):
                    return None
            return None
        if not isinstance(box, (list, tuple)):
            return None
        # [x, y, w, h] 扁平坐标(左, 上, 宽, 高)
        if len(box) == 4 and all(isinstance(p, (int, float)) for p in box):
            x, y, w, h = (int(p) for p in box)
            return (x, y, w, h) if w > 0 and h > 0 else None
        # 扁平四点坐标
        if len(box) == 8 and all(isinstance(p, (int, float)) for p in box):
            xs = box[0::2]
            ys = box[1::2]
            x, y = int(min(xs)), int(min(ys))
            w, h = int(max(xs) - x), int(max(ys) - y)
            return (x, y, w, h) if w > 0 and h > 0 else None
        # 四点坐标列表
        pts = []
        for p in box:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                try:
                    pts.append((float(p[0]), float(p[1])))
                except (TypeError, ValueError):
                    pass
        if len(pts) < 2:
            return None
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x, y = int(min(xs)), int(min(ys))
        w, h = int(max(xs) - x), int(max(ys) - y)
        return (x, y, w, h) if w > 0 and h > 0 else None

    @property
    def mode(self) -> str:
        return self._mode
