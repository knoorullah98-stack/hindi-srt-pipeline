import os
import time
import subprocess
import requests
from faster_whisper import WhisperModel

GH_TOKEN = os.environ["GH_TOKEN"]
GH_OWNER = os.environ["GH_OWNER"]
GH_REPO = os.environ["GH_REPO"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]

JOB_ID = os.environ["JOB_ID"]
RELEASE_TAG = os.environ["RELEASE_TAG"]
ASSET_ID = os.environ["ASSET_ID"]
TRANSCRIPT_ASSET_ID = os.environ.get("TRANSCRIPT_ASSET_ID", "")
MODEL = os.environ["MODEL"]
LANGUAGE = os.environ["LANGUAGE"]
MODE = os.environ["MODE"]
ORIG_FILENAME = os.environ["ORIG_FILENAME"]
TG_CHAT_ID = os.environ["TG_CHAT_ID"]
TG_STATUS_MSG_ID = os.environ["TG_STATUS_MSG_ID"]

GH_API = "https://api.github.com"
HEADERS = {"Authorization": f"Bearer {GH_TOKEN}", "Accept": "application/vnd.github+json"}
LANG_CODE = {"english": "en", "hindi": "hi"}.get(LANGUAGE, None)


def tg_edit_status(text):
    try:
        res = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/editMessageText",
            json={"chat_id": TG_CHAT_ID, "message_id": TG_STATUS_MSG_ID, "text": text},
            timeout=30,
        )
        if not res.ok:
            print(f"Telegram status update failed: {res.status_code} {res.text[:300]}")
    except requests.RequestException as e:
        print(f"Telegram status update exception: {e}")


def tg_send_document(path):
    with open(path, "rb") as f:
        res = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument",
            data={"chat_id": TG_CHAT_ID},
            files={"document": (os.path.basename(path), f)},
            timeout=120,
        )
    if not res.ok:
        raise RuntimeError(f"Telegram sendDocument failed: {res.status_code} {res.text[:500]}")
    return res


def download_asset(asset_id, out_path):
    res = requests.get(
        f"{GH_API}/repos/{GH_OWNER}/{GH_REPO}/releases/assets/{asset_id}",
        headers={**HEADERS, "Accept": "application/octet-stream"},
        stream=True,
        timeout=60,
    )
    res.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in res.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def get_audio_duration(audio_path):
    """Return audio duration in seconds. Uses ffprobe available on GitHub runners."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        duration = float(result.stdout.strip())
        return duration if duration > 0 else None
    except Exception as e:
        print(f"Could not read audio duration with ffprobe: {e}")
        return None


def format_duration(seconds):
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def format_srt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for i, (start, end, text) in enumerate(segments, 1):
            f.write(
                f"{i}\n{format_srt_time(start)} --> {format_srt_time(end)}\n"
                f"{text.strip()}\n\n"
            )


class ProgressTracker:
    """Telegram progress/ETA updater for a single transcription job."""

    def __init__(self, duration):
        self.duration = duration
        self.started = time.monotonic()
        self.last_update = 0.0
        self.last_pct = -1

    def update(self, processed_seconds, prefix):
        if not self.duration or processed_seconds is None:
            return

        pct = min(99, max(0, int((processed_seconds / self.duration) * 100)))
        elapsed = time.monotonic() - self.started

        # Avoid noisy Telegram edits. Update roughly every 15s or every 5%.
        if elapsed < 8:
            return
        if pct < 5:
            return
        if (pct - self.last_pct) < 5 and (time.monotonic() - self.last_update) < 15:
            return

        remaining = None
        if pct > 0:
            remaining = elapsed * (100 - pct) / pct

        eta_text = f"~{format_duration(remaining)}" if remaining is not None else "calculating…"
        text = (
            f"{prefix}\n"
            f"📊 Progress: {pct}%\n"
            f"⏱ Elapsed: {format_duration(elapsed)}\n"
            f"🕒 ETA: {eta_text}"
        )
        tg_edit_status(text)
        self.last_update = time.monotonic()
        self.last_pct = pct


def normal_asr(audio_path, duration):
    tg_edit_status(
        f"🧠 Transcribing ({MODEL})…\n"
        f"🎧 Duration: {format_duration(duration) if duration else 'unknown'}\n"
        f"📊 Progress: 0%\n"
        f"🕒 ETA: calculating…"
    )

    model = WhisperModel(MODEL, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, language=LANG_CODE)

    tracker = ProgressTracker(duration)
    result = []

    for s in segments:
        result.append((s.start, s.end, s.text))
        tracker.update(
            s.end,
            f"🧠 Transcribing ({MODEL})…"
        )

    return result


def forced_align(audio_path, transcript_path, duration):
    tg_edit_status(
        f"🧠 Forced-align chal raha hai ({MODEL})…\n"
        f"🎧 Duration: {format_duration(duration) if duration else 'unknown'}\n"
        f"📊 Progress: 0%\n"
        f"🕒 ETA: calculating…"
    )

    model = WhisperModel(MODEL, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(
        audio_path,
        language=LANG_CODE,
        word_timestamps=True,
    )

    tracker = ProgressTracker(duration)
    asr_words = []

    for seg in segments:
        for w in seg.words:
            asr_words.append((w.start, w.end))
        tracker.update(seg.end, f"🧠 Forced-align chal raha hai ({MODEL})…")

    with open(transcript_path, "r", encoding="utf-8") as f:
        script_words = f.read().split()

    if not asr_words or not script_words:
        raise RuntimeError("ASR ya script me words nahi mile — alignment nahi ho saka.")

    n_asr = len(asr_words)
    n_script = len(script_words)
    aligned = []

    for i, word in enumerate(script_words):
        asr_idx = min(int(i * n_asr / n_script), n_asr - 1)
        aligned.append(
            (asr_words[asr_idx][0], asr_words[asr_idx][1], word)
        )

    # ~8 words ke groups me subtitle lines banao
    grouped = []
    chunk = []

    for start, end, word in aligned:
        chunk.append((start, end, word))
        if len(chunk) == 8:
            grouped.append(
                (chunk[0][0], chunk[-1][1],
                 " ".join(w for _, _, w in chunk))
            )
            chunk = []

    if chunk:
        grouped.append(
            (chunk[0][0], chunk[-1][1],
             " ".join(w for _, _, w in chunk))
        )

    return grouped


def cleanup_release():
    res = requests.get(
        f"{GH_API}/repos/{GH_OWNER}/{GH_REPO}/releases/tags/{RELEASE_TAG}",
        headers=HEADERS,
        timeout=30,
    )
    if res.ok:
        release_id = res.json()["id"]
        requests.delete(
            f"{GH_API}/repos/{GH_OWNER}/{GH_REPO}/releases/{release_id}",
            headers=HEADERS,
            timeout=30,
        )

    requests.delete(
        f"{GH_API}/repos/{GH_OWNER}/{GH_REPO}/git/refs/tags/{RELEASE_TAG}",
        headers=HEADERS,
        timeout=30,
    )


def main():
    audio_path = "input_audio"
    download_asset(ASSET_ID, audio_path)

    duration = get_audio_duration(audio_path)
    if duration:
        print(f"Audio duration: {format_duration(duration)}")
    else:
        print("Audio duration: unknown; ETA will not be shown.")

    if MODE == "transcript" and TRANSCRIPT_ASSET_ID:
        transcript_path = "transcript.txt"
        download_asset(TRANSCRIPT_ASSET_ID, transcript_path)
        segments = forced_align(audio_path, transcript_path, duration)
    else:
        segments = normal_asr(audio_path, duration)

    srt_name = os.path.splitext(ORIG_FILENAME)[0] + ".srt"
    write_srt(segments, srt_name)

    tg_edit_status("📤 SRT Telegram pe bhej raha hoon…")
    tg_send_document(srt_name)

    tg_edit_status(f"✅ Done — job `{JOB_ID}` ka SRT upar bhej diya.")
    cleanup_release()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        tg_edit_status(f"❌ Transcription job fail: {e}")
        raise


def main():
    audio_path = "input_audio"
    download_asset(ASSET_ID, audio_path)

    if MODE == "transcript" and TRANSCRIPT_ASSET_ID:
        transcript_path = "transcript.txt"
        download_asset(TRANSCRIPT_ASSET_ID, transcript_path)
        segments = forced_align(audio_path, transcript_path)
    else:
        segments = normal_asr(audio_path)

    srt_name = os.path.splitext(ORIG_FILENAME)[0] + ".srt"
    write_srt(segments, srt_name)

    tg_edit_status("📤 SRT Telegram pe bhej raha hoon…")
    with open(srt_name, "rb") as f:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument",
            data={"chat_id": TG_CHAT_ID},
            files={"document": (srt_name, f)},
        )

    tg_edit_status(f"✅ Done — job `{JOB_ID}` ka SRT upar bhej diya.")
    cleanup_release()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        tg_edit_status(f"❌ Transcription job fail: {e}")
        raise
