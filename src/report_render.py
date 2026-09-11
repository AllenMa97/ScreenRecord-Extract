"""
人类可读报告渲染器

将提取结果(extracted_content_<视频名>.json 的结构)渲染为 Markdown 和 HTML,
供 main.py(批处理)与 scripts/convert_report.py(独立 CLI)共用。
"""

import html as html_mod
import os


def fmt_time(seconds: float) -> str:
    """秒数格式化为 mm:ss"""
    total = max(0, int(round(seconds)))
    minutes, sec = divmod(total, 60)
    return f"{minutes:02d}:{sec:02d}"


def _sv(seconds: float) -> int:
    """秒数 → 整数(语音气泡行用 +Ns 形式)"""
    return max(0, int(round(seconds)))


def to_markdown(data: dict) -> str:
    """生成 Markdown 报告"""
    lines = []
    lines.append("# 录屏内容提取报告")
    lines.append("")
    lines.append(f"> {data.get('summary', '')} · "
                 f"LLM增强: {'是' if data.get('llm_enhanced') else '否'}")
    lines.append("")

    # 统计概览
    stats = data.get("statistics", {})
    lines.append("## 统计概览")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("|------|----|")
    stat_rows = [
        ("文本行数(跟踪)", stats.get("total_tracked_lines", 0)),
        ("新增文本行", stats.get("total_new_lines", 0)),
        ("段落数", len(data.get("sections", []))),
        ("内容缺口", stats.get("total_gaps", 0)),
        ("LLM增强", "是" if stats.get("llm_enhanced") else "否"),
    ]
    for label, value in stat_rows:
        lines.append(f"| {label} | {value} |")
    lines.append("")

    # 内容段落
    sections = data.get("sections", [])
    if sections:
        lines.append("## 内容段落")
        lines.append("")
        for idx, section in enumerate(sections, start=1):
            timestamp = section.get("timestamp", 0)
            lines.append(f"### 段落 {idx} · {fmt_time(timestamp)}")
            lines.append("")
            stamps = section.get("timestamps", [])
            if stamps:
                lines.append(f"> 时间: {' | '.join(stamps)}")
                lines.append("")
            content = section.get("content", "").strip()
            if content:
                lines.append("```")
                lines.append(content)
                lines.append("```")
                lines.append("")
            note = section.get("note", "").strip()
            if note:
                lines.append(f"> 备注: {note}")
                lines.append("")
    else:
        lines.append("## 内容段落")
        lines.append("")
        lines.append("无")
        lines.append("")

    # 内容缺口
    gaps = data.get("gaps", [])
    lines.append("## 内容缺口")
    lines.append("")
    if gaps:
        for gap in gaps:
            timestamp = gap.get("timestamp", 0) if isinstance(gap, dict) else 0
            detail = gap.get("detail", str(gap)) if isinstance(gap, dict) else str(gap)
            lines.append(f"- **{fmt_time(timestamp)}**: {detail}")
    else:
        lines.append("无")
    lines.append("")

    # 音频转写(独立于视觉配对, 保证音频内容始终可见)
    audio = data.get("audio") or {}
    segments = audio.get("segments") or []
    lines.append("## 音频转写")
    lines.append("")
    if segments:
        has_text = False
        for seg in segments:
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            has_text = True
            start = seg.get("start", 0)
            end = seg.get("end", start)
            tail = f"- **{fmt_time(start)}–{fmt_time(end)}**: {text}"
            lines.append(tail)
        if not has_text:
            lines.append("有音轨但未采集到人声(静音 / 语音未播放)")
    else:
        reason = audio.get("reason", "")
        lines.append(f"无{('(' + reason + ')') if reason else ''}")
    lines.append("")

    # 视觉语音气泡: 若无对应音频被采集, 仍保留下气泡时间与方向, 避免信息丢失
    lines.append("## 语音消息(气泡)")
    lines.append("")
    voice_events = data.get("voice_events") or []
    audio_has_text = bool(
        audio.get("has_audio")
        and any((s.get("text") or "").strip() for s in (audio.get("segments") or []))
    )
    if voice_events:
        for ev in voice_events:
            t = ev.get("timestamp", 0)
            side = f"[{ev.get('side')}] " if ev.get("side") else ""
            text = (ev.get("text") or "").strip()
            if ev.get("matched") and text:
                marker = f"：{text}"
            elif text:
                marker = f"：{text}(未配对)"
            elif audio_has_text:
                marker = "：(有语音段未配对)"
            else:
                marker = "：(未采集到对应声音)"
            lines.append(f"- **+{_sv(t)}s** {side}{marker}")
    else:
        lines.append("无(未检测到语音气泡)")
    lines.append("")

    # 全文合并
    if sections:
        lines.append("## 全文合并")
        lines.append("")
        merged = "\n".join(
            s.get("content", "").strip() for s in sections if s.get("content", "").strip()
        )
        lines.append("```")
        lines.append(merged)
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def to_html(data: dict) -> str:
    """生成 HTML 报告"""
    stats = data.get("statistics", {})
    sections = data.get("sections", [])
    gaps = data.get("gaps", [])

    stat_rows = "".join(
        f"<tr><td>{label}</td><td>{value}</td></tr>"
        for label, value in [
            ("文本行数(跟踪)", stats.get("total_tracked_lines", 0)),
            ("新增文本行", stats.get("total_new_lines", 0)),
            ("段落数", len(sections)),
            ("内容缺口", stats.get("total_gaps", 0)),
            ("LLM增强", "是" if stats.get("llm_enhanced") else "否"),
        ]
    )

    section_html = ""
    for idx, section in enumerate(sections, start=1):
        timestamp = section.get("timestamp", 0)
        content = html_mod.escape(section.get("content", "").strip())
        note = html_mod.escape(section.get("note", "").strip())
        stamps = section.get("timestamps", [])
        stamps_html = ""
        if stamps:
            chips = "".join(
                f'<span class="stamp">{html_mod.escape(s)}</span>' for s in stamps
            )
            stamps_html = f'<div class="stamps">{chips}</div>'
        note_html = f'<div class="note">{note}</div>' if note else ""
        section_html += (
            f'<div class="section">'
            f'<div class="section-header">'
            f'<span class="section-id">段落 {idx}</span>'
            f'<span class="badge">{fmt_time(timestamp)}</span>'
            f'</div>'
            f'{stamps_html}'
            f'<pre class="content">{content}</pre>'
            f'{note_html}'
            f'</div>'
        )
    if not sections:
        section_html = '<p class="empty">无</p>'

    if gaps:
        gap_items = "".join(
            f"<li><strong>{fmt_time(g.get('timestamp', 0))}</strong>: "
            f"{html_mod.escape(g.get('detail', str(g)))}</li>"
            for g in gaps
        )
        gap_html = f"<ul>{gap_items}</ul>"
    else:
        gap_html = '<p class="empty">无</p>'

    merged_html = ""
    if sections:
        merged = "\n".join(
            s.get("content", "").strip() for s in sections if s.get("content", "").strip()
        )
        merged_html = (
            '<h2>全文合并</h2>'
            f'<pre class="content">{html_mod.escape(merged)}</pre>'
        )

    # 音频转写区块(独立于视觉配对, 保证音频内容始终可见)
    audio = data.get("audio") or {}
    segments = audio.get("segments") or []
    if segments:
        rows = []
        for seg in segments:
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            start = seg.get("start", 0)
            end = seg.get("end", start)
            rows.append(
                '<div class="audio-item">'
                f'<span class="audio-time">{fmt_time(start)}–{fmt_time(end)}</span>'
                f'<span class="audio-text">{html_mod.escape(text)}</span>'
                "</div>"
            )
        if rows:
            audio_html = (f'<h2>音频转写</h2><div class="audio-list">'
                          f'<p class="empty">共 {len(rows)} 段(有文本)</p>'
                          f'{"".join(rows)}</div>')
        else:
            audio_html = ('<h2>音频转写</h2>'
                          '<p class="empty">有音轨但未采集到人声(静音 / 语音未播放)</p>')
    else:
        reason = html_mod.escape(audio.get("reason", ""))
        audio_html = (f'<h2>音频转写</h2>'
                      f'<p class="empty">无{(("(" + reason + ")") if reason else "")}</p>')

    # 视觉语音气泡: 若无对应音频被采集, 仍保留下气泡时间与方向, 避免信息丢失
    voice_events = data.get("voice_events") or []
    audio_has_text = bool(
        audio.get("has_audio")
        and any((s.get("text") or "").strip() for s in (audio.get("segments") or []))
    )
    if voice_events:
        v_items = []
        for ev in voice_events:
            t = ev.get("timestamp", 0)
            side = html_mod.escape((ev.get("side") or "") or "")
            text = (ev.get("text") or "").strip()
            if ev.get("matched") and text:
                marker = html_mod.escape(text)
            elif text:
                marker = html_mod.escape(text) + "(未配对)"
            elif audio_has_text:
                marker = "(有语音段未配对)"
            else:
                marker = "(未采集到对应声音)"
            v_items.append(
                '<div class="audio-item">'
                f'<span class="audio-time">+{_sv(t)}s'
                f'{(" [" + side + "]") if side else ""}</span>'
                f'<span class="audio-text">{marker}</span>'
                "</div>"
            )
        voice_html = ('<h2>语音消息(气泡)</h2>'
                      f'<div class="audio-list">{"".join(v_items)}</div>')
    else:
        voice_html = '<h2>语音消息(气泡)</h2><p class="empty">无(未检测到语音气泡)</p>'

    page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>录屏内容提取报告</title>
<style>
  body {{
    font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
    margin: 0; padding: 0;
    background: #f5f6f8; color: #24292f;
  }}
  .container {{ max-width: 860px; margin: 0 auto; padding: 32px 20px; }}
  h1 {{ font-size: 26px; margin: 0 0 8px; }}
  .subtitle {{ color: #57606a; margin-bottom: 28px; }}
  h2 {{ font-size: 20px; margin: 36px 0 12px; padding-bottom: 6px;
       border-bottom: 1px solid #d0d7de; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff;
         border: 1px solid #d0d7de; border-radius: 8px; overflow: hidden; }}
  th, td {{ padding: 8px 14px; text-align: left; font-size: 14px; }}
  th {{ background: #f6f8fa; }}
  .section {{ background: #fff; border: 1px solid #d0d7de;
            border-radius: 8px; padding: 16px 18px; margin-bottom: 14px; }}
  .section-header {{ display: flex; justify-content: space-between;
                    align-items: center; margin-bottom: 10px; }}
  .section-id {{ font-weight: 600; }}
  .badge {{ background: #0969da; color: #fff; border-radius: 12px;
           padding: 2px 10px; font-size: 13px; }}
  pre.content {{ white-space: pre-wrap; word-break: break-word;
               margin: 0; font-size: 14px; line-height: 1.6;
               font-family: "Microsoft YaHei", sans-serif; }}
  .stamps {{ margin-bottom: 8px; }}
  .stamp {{ display: inline-block; background: #e6f4ea; color: #1a7f37;
           border-radius: 10px; padding: 1px 8px; margin-right: 6px;
           font-size: 12px; }}
  .note {{ margin-top: 10px; padding: 8px 12px; background: #fff8c5;
          border-radius: 6px; font-size: 13px; color: #7d4e00; }}
  .empty {{ color: #57606a; }}
  .audio-list {{ background: #fff; border: 1px solid #d0d7de;
               border-radius: 8px; padding: 12px 16px; }}
  .audio-item {{ display: flex; gap: 12px; padding: 6px 0;
               border-bottom: 1px dashed #eaeef2; }}
  .audio-item:last-child {{ border-bottom: none; }}
  .audio-time {{ color: #0969da; font-weight: 600; white-space: nowrap;
               min-width: 96px; }}
  .audio-text {{ word-break: break-word; }}
  .audio-list .empty {{ margin: 0 0 8px; font-size: 13px; }}
</style>
</head>
<body>
<div class="container">
  <h1>录屏内容提取报告</h1>
  <div class="subtitle">{html_mod.escape(data.get("summary", ""))} · LLM增强: {'是' if data.get("llm_enhanced") else '否'}</div>

  <h2>统计概览</h2>
  <table><tbody>{stat_rows}</tbody></table>

  <h2>内容段落</h2>
  {section_html}

  <h2>内容缺口</h2>
  {gap_html}

  {audio_html}

  {voice_html}

  {merged_html}
</div>
</body>
</html>
"""
    return page


def save_reports(data: dict, output_dir: str, stem: str) -> tuple[str, str]:
    """按视频名生成并保存 md/html 报告。

    参数:
        data: 提取结果字典。
        output_dir: 输出目录。
        stem: 报告文件名(不含扩展名, 通常为 extracted_content_<视频名>)。

    返回:
        (md_path, html_path): 生成的 md 与 html 文件路径
    """
    os.makedirs(output_dir, exist_ok=True)
    md_path = os.path.join(output_dir, f"{stem}.md")
    html_path = os.path.join(output_dir, f"{stem}.html")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(to_markdown(data))
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(to_html(data))
    return md_path, html_path
