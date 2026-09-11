"""
语义去重器

双模态联动的核心: LLM作为视觉和文本两个模态都拿不准时的"仲裁者"

两大职责:
  1. 灰色地带语义去重 (方案A的LLM仲裁)
     - 当视觉去重输出GRAY_ZONE时, 当前帧已送OCR
     - LLM对比当前帧OCR文本与历史文本, 做语义级判定:
       * "今天天气真好" vs "今天天气真不错" → 语义重复 → 丢弃
       * "我明天去找你" vs "我后天去找你" → 语义不同 → 保留
     - 编辑距离做不到: 字符相似度高但语义一个重复一个不同

  2. OCR错误兜底纠错
     - 利用跨帧上下文修正单帧OCR错误
     - 结合多帧观测的置信度投票
     - LLM根据上下文推断最可能的正确文本

设计原则:
  - LLM不是每帧都调, 只在灰色地带调用(约20-30%帧)
  - LLM不可用时降级为纯文本相似度判定
  - 所有LLM调用记录在行为日志中
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from src.data_types import TextLine
from src.utils.text_utils import text_similarity, normalize_text
from src.utils.behavior_logger import BehaviorLogger


class SemanticVerdict(Enum):
    """LLM 语义判定结果"""
    DUPLICATE = "duplicate"             # 语义重复, 丢弃
    NEW_CONTENT = "new_content"         # 语义新内容, 保留
    CORRECTED = "corrected"             # OCR 纠错后保留
    UNCERTAIN = "uncertain"             # 不确定, 保守保留


@dataclass
class SemanticResult:
    """语义判定结果"""
    verdict: SemanticVerdict
    corrected_text: Optional[str] = None    # LLM 纠错后的文本
    confidence: float = 0.0                 # 判定置信度
    reason: str = ""                        # 判定原因
    llm_called: bool = False                # 是否调用了 LLM
    text_verdict_for_calibration: str = ""  # 给方案B的反馈


class SemanticDedup:
    """语义去重器(LLM 仲裁 + OCR 纠错)"""

    def __init__(self, config: dict, logger: BehaviorLogger):
        self._enabled: bool = config.get("enabled", False)
        self._endpoint: str = config.get("api_endpoint", "")
        self._api_key: str = config.get("api_key", "")
        self._model: str = config.get("model", "deepseek-chat")
        self._timeout: int = config.get("timeout", 60)
        self._logger = logger

        # 纯文本判定降级阈值
        self._text_dup_threshold: float = config.get("text_dup_threshold", 0.92)
        self._text_new_threshold: float = config.get("text_new_threshold", 0.60)
        self._ocr_correction_confidence: float = config.get("ocr_correction_confidence", 0.5)
        self._thinking: bool = config.get("thinking", False)

        # 历史文本缓存(用于LLM上下文)
        self._recent_texts: List[Tuple[int, str, float]] = []  # (frame_id, text, confidence)
        self._max_history: int = 20

        # 统计
        self._stats = {
            "total_calls": 0,
            "llm_calls": 0,
            "llm_duplicates": 0,
            "llm_new": 0,
            "llm_corrected": 0,
            "fallback_used": 0,
        }

    @property
    def enabled(self) -> bool:
        return self._enabled and bool(self._api_key)

    def arbitrate(self, frame_id: int, timestamp: float,
                  current_lines: List[TextLine],
                  prev_lines: Optional[List[TextLine]]) -> Tuple[SemanticResult, List[TextLine]]:
        """
        对灰色地带帧做语义判定。

        参数:
            frame_id: 帧ID
            timestamp: 时间戳
            current_lines: 当前帧OCR文本行
            prev_lines: 上一帧OCR文本行
        返回:
            - SemanticResult: 语义判定结果
            - corrected_lines: OCR纠错后的文本行(可能被修正)
        """
        self._stats["total_calls"] += 1

        if not current_lines:
            return SemanticResult(
                verdict=SemanticVerdict.UNCERTAIN,
                reason="无OCR文本, 无法判定 / No OCR text",
                text_verdict_for_calibration=""
            ), current_lines

        # 尝试OCR纠错
        corrected_lines = self._correct_ocr_errors(frame_id, timestamp, current_lines)

        if prev_lines is None or len(prev_lines) == 0:
            # 无对比对象 → 新内容
            self._add_to_history(frame_id, corrected_lines)
            return SemanticResult(
                verdict=SemanticVerdict.NEW_CONTENT,
                reason="首帧或无对比对象 / First frame or no comparison",
                text_verdict_for_calibration="text_new"
            ), corrected_lines

        # 提取文本用于比对
        curr_texts = [normalize_text(l.text) for l in corrected_lines if l.text.strip()]
        prev_texts = [normalize_text(l.text) for l in prev_lines if l.text.strip()]

        if not curr_texts:
            return SemanticResult(
                verdict=SemanticVerdict.UNCERTAIN,
                reason="当前帧无有效文本 / No valid text in current frame",
                text_verdict_for_calibration=""
            ), corrected_lines

        if self.enabled:
            return self._llm_arbitrate(frame_id, timestamp, curr_texts, prev_texts,
                                       corrected_lines)
        else:
            return self._text_fallback_arbitrate(curr_texts, prev_texts, corrected_lines)

    # ==========================================================================
    # LLM语义判定
    # ==========================================================================

    def _llm_arbitrate(self, frame_id: int, timestamp: float,
                       curr_texts: List[str], prev_texts: List[str],
                       corrected_lines: List[TextLine]) -> Tuple[SemanticResult, List[TextLine]]:
        """调用LLM做语义判定"""
        self._stats["llm_calls"] += 1

        prompt = self._build_arbitration_prompt(curr_texts, prev_texts)

        try:
            response = self._call_llm_api(prompt)
            result = self._parse_arbitration_response(response)

            self._logger.log("semantic_dedup", "llm_arbitrate",
                             f"帧{frame_id}: LLM判定={result.verdict.value}, "
                             f"reason={result.reason}",
                             frame_id=frame_id,
                             extra={"verdict": result.verdict.value,
                                    "confidence": result.confidence,
                                    "reason": result.reason})

            # 更新统计
            if result.verdict == SemanticVerdict.DUPLICATE:
                self._stats["llm_duplicates"] += 1
            elif result.verdict == SemanticVerdict.NEW_CONTENT:
                self._stats["llm_new"] += 1
            elif result.verdict == SemanticVerdict.CORRECTED:
                self._stats["llm_corrected"] += 1

            # 如果LLM提供了纠错文本, 应用纠错
            if result.corrected_text and result.verdict == SemanticVerdict.CORRECTED:
                corrected_lines = self._apply_correction(corrected_lines, result.corrected_text)

            self._add_to_history(frame_id, corrected_lines)
            return result, corrected_lines

        except Exception as e:
            self._logger.log("semantic_dedup", "llm_error",
                             f"LLM调用失败, 降级为文本判定: {e}",
                             frame_id=frame_id)
            self._stats["fallback_used"] += 1
            return self._text_fallback_arbitrate(curr_texts, prev_texts, corrected_lines)

    def _build_arbitration_prompt(self, curr_texts: List[str],
                                  prev_texts: List[str]) -> str:
        """构建LLM仲裁提示"""
        curr_block = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(curr_texts))
        prev_block = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(prev_texts))

        prompt = f"""你是一个手机录屏OCR去重助手。请判断当前帧的文本与上一帧的文本是否语义重复。

判断标准:
- "今天天气真好" vs "今天天气真不错" → 语义重复(表述不同但含义相同)
- "我明天去找你" vs "我后天去找你" → 语义不同(关键信息不同)
- 完全相同的文本 → 语义重复
- 如果存在明显OCR错误(如乱码、缺字), 请给出纠正后的文本

上一帧OCR文本:
{prev_block}

当前帧OCR文本:
{curr_block}

请以JSON格式输出:
{{
  "verdict": "duplicate" 或 "new_content" 或 "corrected",
  "corrected_text": "纠错后的文本(仅corrected时填写, 否则为空)",
  "confidence": 0.0到1.0,
  "reason": "简要说明判定原因"
}}"""
        return prompt

    def _call_llm_api(self, prompt: str) -> dict:
        """调用LLM API"""
        import requests

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": "你是一个专业的OCR语义判定助手。只输出JSON。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1024,
        }
        if not self._thinking:
            payload["thinking"] = {"type": "disabled"}
        resp = requests.post(self._endpoint, json=payload, headers=headers,
                             timeout=self._timeout)
        resp.raise_for_status()
        return resp.json()

    def _parse_arbitration_response(self, response: dict) -> SemanticResult:
        """解析LLM仲裁响应"""
        content = response["choices"][0]["message"]["content"]

        # 从markdown代码块中提取JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        data = json.loads(content.strip())

        verdict_str = data.get("verdict", "uncertain")
        verdict_map = {
            "duplicate": SemanticVerdict.DUPLICATE,
            "new_content": SemanticVerdict.NEW_CONTENT,
            "corrected": SemanticVerdict.CORRECTED,
        }
        verdict = verdict_map.get(verdict_str, SemanticVerdict.UNCERTAIN)

        # 给方案B的校准反馈
        if verdict == SemanticVerdict.DUPLICATE:
            calib_feedback = "text_dup"
        elif verdict == SemanticVerdict.NEW_CONTENT:
            calib_feedback = "text_new"
        else:
            calib_feedback = ""

        return SemanticResult(
            verdict=verdict,
            corrected_text=data.get("corrected_text", "") or None,
            confidence=float(data.get("confidence", 0.5)),
            reason=data.get("reason", ""),
            llm_called=True,
            text_verdict_for_calibration=calib_feedback
        )

    # ==========================================================================
    # OCR错误兜底纠错
    # ==========================================================================

    def _correct_ocr_errors(self, frame_id: int, timestamp: float,
                            lines: List[TextLine]) -> List[TextLine]:
        """
        OCR错误兜底纠错。

        策略:
          1. 低置信度行(< ocr_correction_confidence)标记为需纠错
          2. 在历史文本中查找最相似的行, 如果相似度>0.8, 用历史文本替换
          3. 这是纯CPU纠错, 不调LLM(LLM纠错在arbitrate中按需触发)
        """
        corrected = []
        for line in lines:
            if line.confidence < self._ocr_correction_confidence and self._recent_texts:
                # 在历史中找最相似
                best_sim = 0.0
                best_text = None
                for _, hist_text, _ in self._recent_texts:
                    sim = text_similarity(line.text, hist_text)
                    if sim > best_sim:
                        best_sim = sim
                        best_text = hist_text

                if best_sim > 0.8 and best_text:
                    # 用历史文本替换低置信度文本
                    corrected_line = TextLine(
                        text=best_text,
                        bbox=line.bbox,
                        confidence=max(line.confidence, best_sim),
                        frame_id=frame_id,
                        timestamp=timestamp
                    )
                    self._logger.log("semantic_dedup", "ocr_corrected",
                                     f"OCR纠错: '{line.text}' → '{best_text}' "
                                     f"(sim={best_sim:.2f})",
                                     frame_id=frame_id,
                                     extra={"original": line.text,
                                            "corrected": best_text,
                                            "similarity": round(best_sim, 2)})
                    corrected.append(corrected_line)
                else:
                    corrected.append(line)
            else:
                corrected.append(line)
        return corrected

    def _apply_correction(self, lines: List[TextLine],
                          corrected_text: str) -> List[TextLine]:
        """应用LLM纠错文本"""
        if not lines:
            return lines
        # 简单策略: 将纠错文本应用到第一个文本行
        result = list(lines)
        result[0] = TextLine(
            text=corrected_text,
            bbox=result[0].bbox,
            confidence=0.95,
            frame_id=result[0].frame_id,
            timestamp=result[0].timestamp
        )
        self._logger.log("semantic_dedup", "llm_correction_applied",
                         f"LLM纠错应用: → '{corrected_text}'")
        return result

    # ==========================================================================
    # 纯文本降级判定
    # ==========================================================================

    def _text_fallback_arbitrate(self, curr_texts: List[str],
                                 prev_texts: List[str],
                                 corrected_lines: List[TextLine]
                                 ) -> Tuple[SemanticResult, List[TextLine]]:
        """
        纯文本相似度降级判定(LLM不可用时使用)

        利用编辑距离做判定, 但阈值比message_bubble_tracker更宽松(因为是灰色地带)。
        """
        # 计算所有文本行的最大相似度
        max_sim = 0.0
        for curr in curr_texts:
            for prev in prev_texts:
                sim = text_similarity(curr, prev)
                if sim > max_sim:
                    max_sim = sim

        # 整体相似度: 当前帧与上一帧的行级重叠率
        matched = 0
        for curr in curr_texts:
            for prev in prev_texts:
                if text_similarity(curr, prev) > 0.85:
                    matched += 1
                    break
        overlap_ratio = matched / max(len(curr_texts), len(prev_texts)) if curr_texts else 0.0

        # 判定
        if overlap_ratio > self._text_dup_threshold:
            verdict = SemanticVerdict.DUPLICATE
            calib = "text_dup"
            reason = f"文本降级判定: 重叠率={overlap_ratio:.2f} > {self._text_dup_threshold}"
        elif overlap_ratio < self._text_new_threshold:
            verdict = SemanticVerdict.NEW_CONTENT
            calib = "text_new"
            reason = f"文本降级判定: 重叠率={overlap_ratio:.2f} < {self._text_new_threshold}"
        else:
            # 不确定时保守保留
            verdict = SemanticVerdict.NEW_CONTENT
            calib = "text_new"
            reason = f"文本降级判定: 重叠率={overlap_ratio:.2f}, 不确定, 保守保留"

        self._stats["fallback_used"] += 1
        self._add_to_history(corrected_lines[0].frame_id if corrected_lines else 0,
                             corrected_lines)

        return SemanticResult(
            verdict=verdict,
            confidence=overlap_ratio,
            reason=reason,
            llm_called=False,
            text_verdict_for_calibration=calib
        ), corrected_lines

    # ==========================================================================
    # 历史管理
    # ==========================================================================

    def _add_to_history(self, frame_id: int, lines: List[TextLine]) -> None:
        """添加到历史缓存"""
        for line in lines:
            if line.text.strip():
                self._recent_texts.append((frame_id, line.text, line.confidence))
        # 保留最近N条
        if len(self._recent_texts) > self._max_history:
            self._recent_texts = self._recent_texts[-self._max_history:]

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    def reset_stats(self) -> None:
        """重置统计"""
        self._stats = {
            "total_calls": 0,
            "llm_calls": 0,
            "llm_duplicates": 0,
            "llm_new": 0,
            "llm_corrected": 0,
            "fallback_used": 0,
        }
