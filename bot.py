"""
Usage (local):
  export BOT_TOKEN="..."
  python bot.py
"""

import asyncio
import contextlib
import dataclasses
import html
import logging
import os
import re
import tempfile
import time
import traceback
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

import requests
import yt_dlp
from telegram import InlineQueryResultCachedVideo, InputMediaPhoto, InputMediaVideo, Update
from telegram.constants import ParseMode
from telegram.ext import Application, ContextTypes, InlineQueryHandler, MessageHandler, filters

# -------------------------
# Configuration
# -------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
MAX_CONCURRENT_DOWNLOADS = 1
MAX_BOT_FILE_BYTES = int(os.getenv("MAX_BOT_FILE_BYTES", str(50 * 1024 * 1024)))
DOWNLOAD_RETRY_ATTEMPTS = int(os.getenv("DOWNLOAD_RETRY_ATTEMPTS", "3"))
RETRY_PAUSE_SECONDS = float(os.getenv("RETRY_PAUSE_SECONDS", "3"))
REQUEST_PAUSE_SECONDS = float(os.getenv("REQUEST_PAUSE_SECONDS", "1"))
HEARTBEAT_INTERVAL_SECONDS = int(os.getenv("HEARTBEAT_INTERVAL_SECONDS", "300"))
CHROME_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
COOKIES_DIR = Path("cookies")
TIKTOK_COOKIES = COOKIES_DIR / "tiktok.txt"
INSTAGRAM_COOKIES = COOKIES_DIR / "ig.txt"
YOUTUBE_COOKIES = COOKIES_DIR / "youtube.txt"
REDGIFS_COOKIES = COOKIES_DIR / "redgifs.txt"
URL_EXPAND_TIMEOUT_SECONDS = float(os.getenv("URL_EXPAND_TIMEOUT_SECONDS", "12"))
SEEN_URL_TTL_SECONDS = int(os.getenv("SEEN_URL_TTL_SECONDS", "300"))
GLOBAL_DOWNLOAD_COOLDOWN_SECONDS = float(os.getenv("GLOBAL_DOWNLOAD_COOLDOWN_SECONDS", "5"))

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("video-bot")
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# -------------------------
# URL Detection
# -------------------------
URL_REGEX = re.compile(r"(https?://[^\s]+)", re.IGNORECASE)
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "gif", "bmp"}


@dataclasses.dataclass
class MediaItem:
    type: str
    meta: dict
    preferred_ext: str
    estimated_filesize: int | None
    index: int = 1


class MediaDownloadError(RuntimeError):
    pass


class LoginRequiredError(MediaDownloadError):
    pass


def extract_links(text: str) -> list[str]:
    raw_links = URL_REGEX.findall(text or "")
    return [link.rstrip(".,;:!?)]}>") for link in raw_links]


def detect_platform(url: str) -> str | None:
    parsed = urlparse(url)
    host = parsed.netloc.lower().split(":")[0]
    path = parsed.path.lower()

    if host in {"youtube.com", "www.youtube.com", "m.youtube.com"} and (
        path.startswith("/watch") or path.startswith("/shorts/")
    ):
        return "youtube"
    if host == "youtu.be":
        return "youtube"

    if host in {"instagram.com", "www.instagram.com", "instagr.am", "www.instagr.am"} and (
        path.startswith("/reel/") or path.startswith("/p/")
    ):
        return "instagram"

    if host in {"tiktok.com", "www.tiktok.com", "vm.tiktok.com", "vt.tiktok.com"}:
        return "tiktok"

    if host in {"twitch.tv", "www.twitch.tv", "m.twitch.tv", "clips.twitch.tv"}:
        return "twitch"

    if host in {"redgifs.com", "www.redgifs.com"}:
        return "redgifs"

    return None


def is_supported_download_url(url: str, platform: str) -> bool:
    parsed = urlparse(url)
    host = parsed.netloc.lower().split(":")[0]
    path = parsed.path.lower()
    if platform == "youtube":
        return path.startswith("/shorts/")
    if platform == "instagram":
        return path.startswith("/reel/") or path.startswith("/p/")
    if platform == "twitch":
        if host == "clips.twitch.tv":
            return True
        return "/clip/" in path
    if platform == "redgifs":
        return path.startswith("/watch/") or path.startswith("/ifr/")
    return platform == "tiktok"


def get_cookiefile_for_platform(platform: str) -> Path | None:
    if platform == "tiktok":
        return TIKTOK_COOKIES
    if platform == "instagram":
        return INSTAGRAM_COOKIES
    if platform == "youtube":
        return YOUTUBE_COOKIES
    if platform == "redgifs":
        return REDGIFS_COOKIES
    return None


def _expand_url_sync(url: str) -> str:
    request_headers = {
        "User-Agent": CHROME_USER_AGENT,
        "Referer": "https://www.youtube.com/",
    }

    request = Request(url, headers=request_headers, method="HEAD")
    try:
        with contextlib.closing(urlopen(request, timeout=URL_EXPAND_TIMEOUT_SECONDS)) as response:
            return response.geturl()
    except Exception as head_err:
        logger.info("HEAD expand failed for %s: %s", url, head_err)

    request = Request(url, headers=request_headers, method="GET")
    with contextlib.closing(urlopen(request, timeout=URL_EXPAND_TIMEOUT_SECONDS)) as response:
        return response.geturl()


async def expand_url(url: str) -> str:
    try:
        final_url = await asyncio.to_thread(_expand_url_sync, url)
        if final_url != url:
            logger.info("Expanded URL: original=%s final=%s", url, final_url)
        else:
            logger.info("Resolved URL without redirects: %s", final_url)
        return final_url
    except Exception as expand_err:
        logger.warning("Could not expand URL %s: %s", url, expand_err)
        return url


async def find_download_target(text: str) -> tuple[str | None, str | None]:
    for raw_url in extract_links(text):
        final_url = await expand_url(raw_url)
        platform = detect_platform(final_url) or detect_platform(raw_url)
        if not platform:
            continue

        if not is_supported_download_url(final_url, platform):
            if platform == "youtube":
                logger.info("Skipping non-Shorts YouTube URL: original=%s final=%s", raw_url, final_url)
            if platform == "twitch":
                logger.info("Skipping non-clip Twitch URL: original=%s final=%s", raw_url, final_url)
            continue

        logger.info("Using URL: original=%s final=%s platform=%s", raw_url, final_url, platform)
        return final_url, platform

    return None, None


# -------------------------
# Downloader
# -------------------------
def _normalize_url_for_extraction(url: str, platform: str) -> list[str]:
    parsed = urlparse(url)
    candidates: list[str] = []
    if platform == "instagram":
        clean_query = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if k != "img_index"]
        cleaned = parsed._replace(query=urlencode(clean_query))
        candidates.append(urlunparse(cleaned))
    elif platform == "tiktok" and "/photo/" in parsed.path:
        candidates.append(urlunparse(parsed._replace(path=parsed.path.replace("/photo/", "/video/", 1))))
        candidates.append(url)
    else:
        candidates.append(url)
    return list(dict.fromkeys(candidates))


def _base_ydl_opts(platform: str, temp_dir: Path | None = None, prefer_photo: bool = False) -> dict:
    ydl_opts = {
        "noplaylist": False,
        "retries": 2,
        "extractor_retries": 2,
        "fragment_retries": 2,
        "sleep_interval_requests": REQUEST_PAUSE_SECONDS,
        "user_agent": CHROME_USER_AGENT,
        "http_headers": {
            "User-Agent": CHROME_USER_AGENT,
            "Referer": "https://www.youtube.com/",
        },
        "quiet": True,
        "no_warnings": True,
    }
    if temp_dir is not None:
        ydl_opts["outtmpl"] = str(temp_dir / "%(id)s.%(ext)s")
    if prefer_photo:
        ydl_opts["format"] = "best"
    else:
        ydl_opts["format"] = "bestvideo[height<=720]+bestaudio/best[height<=720]/best"
        ydl_opts["merge_output_format"] = "mp4"
        ydl_opts["max_filesize"] = MAX_BOT_FILE_BYTES
    if platform == "redgifs":
        ydl_opts["format"] = "best[ext=mp4]/best"
        ydl_opts["sleep_interval_requests"] = 10
        ydl_opts["http_headers"] = {
            "User-Agent": CHROME_USER_AGENT,
            "Referer": "https://www.redgifs.com/",
        }

    default_cookiefile = get_cookiefile_for_platform(platform)
    if default_cookiefile and default_cookiefile.exists():
        ydl_opts["cookiefile"] = str(default_cookiefile)
    elif default_cookiefile:
        logger.warning("Cookies file is missing for %s: %s", platform, default_cookiefile)
    return ydl_opts


def _infer_media_type(meta: dict) -> str:
    ext = (meta.get("ext") or "").lower()
    url = (meta.get("url") or "").lower()
    if meta.get("vcodec") and meta.get("vcodec") != "none":
        return "video"
    if meta.get("duration") not in (None, 0):
        return "video"
    if ext in IMAGE_EXTENSIONS or any(url.endswith(f".{image_ext}") for image_ext in IMAGE_EXTENSIONS):
        return "photo"
    if meta.get("thumbnails") and not meta.get("formats"):
        return "photo"
    return "video"


def _build_media_item(meta: dict | str, index: int) -> MediaItem:
    if not isinstance(meta, dict):
        meta = {"url": str(meta), "id": f"item_{index}"}
    media_type = _infer_media_type(meta)
    preferred_ext = (meta.get("ext") or ("jpg" if media_type == "photo" else "mp4")).lower()
    estimated_filesize = meta.get("filesize") or meta.get("filesize_approx")
    return MediaItem(
        type=media_type,
        meta=meta,
        preferred_ext=preferred_ext,
        estimated_filesize=estimated_filesize,
        index=index,
    )


async def extract_media_items(url: str, platform: str) -> list[MediaItem]:
    ydl_opts = _base_ydl_opts(platform, temp_dir=None, prefer_photo=True)
    ydl_opts["skip_download"] = True

    def _run_extract(target_url: str) -> dict:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(target_url, download=False)

    candidates = _normalize_url_for_extraction(url, platform)
    last_error: Exception | None = None
    info: dict | None = None
    for candidate in candidates:
        try:
            info = await asyncio.to_thread(_run_extract, candidate)
            break
        except yt_dlp.utils.UnsupportedError as err:
            last_error = err
        except Exception as err:
            last_error = err
            if candidate != candidates[-1]:
                logger.warning("extract_info failed, trying next candidate: url=%s err=%s", candidate, err)

    if info is None:
        raise MediaDownloadError(f"Failed to extract media info: {url}") from last_error

    entries = info.get("entries") if isinstance(info, dict) else None
    if entries:
        items = [_build_media_item(entry, idx + 1) for idx, entry in enumerate(entries) if entry]
    else:
        items = [_build_media_item(info, 1)]

    if not items:
        raise MediaDownloadError("No media entries found in extracted info")
    return items


def _pick_downloaded_file(temp_dir: Path, result_info: dict, ydl: yt_dlp.YoutubeDL) -> Path | None:
    requested_downloads = result_info.get("requested_downloads") or []
    for download_info in requested_downloads:
        filepath = download_info.get("filepath")
        if filepath and Path(filepath).exists():
            return Path(filepath)
    filename = result_info.get("_filename")
    if filename and Path(filename).exists():
        return Path(filename)
    prepared = ydl.prepare_filename(result_info)
    if prepared and Path(prepared).exists():
        return Path(prepared)
    files = sorted(temp_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _resolve_image_url(meta: dict) -> str | None:
    candidates = [
        meta.get("url"),
        meta.get("thumbnail"),
        meta.get("display_url"),
    ]
    thumbnails = meta.get("thumbnails") or []
    if thumbnails and isinstance(thumbnails, list):
        first_thumb = thumbnails[0] or {}
        if isinstance(first_thumb, dict):
            candidates.append(first_thumb.get("url"))
    formats = meta.get("formats") or []
    if formats and isinstance(formats, list):
        first_format = formats[0] or {}
        if isinstance(first_format, dict):
            candidates.append(first_format.get("url"))

    for candidate in candidates:
        if not candidate:
            continue
        clean = str(candidate).replace("\\u0026", "&")
        clean = html.unescape(clean)
        if clean.startswith("http://") or clean.startswith("https://"):
            return clean
    return None


def _http_extension_from_response(url: str, content_type: str | None, preferred_ext: str) -> str:
    if content_type:
        lowered = content_type.lower()
        if "jpeg" in lowered:
            return "jpg"
        if "png" in lowered:
            return "png"
        if "webp" in lowered:
            return "webp"
        if "gif" in lowered:
            return "gif"
    suffix = Path(urlparse(url).path).suffix.lower().lstrip(".")
    if suffix in IMAGE_EXTENSIONS:
        return suffix
    return preferred_ext if preferred_ext in IMAGE_EXTENSIONS else "jpg"


async def _direct_http_download_photo(item: MediaItem, temp_dir: Path) -> Path:
    img_url = _resolve_image_url(item.meta)
    if not img_url:
        raise MediaDownloadError("Photo URL not found in media metadata")

    def _run_http() -> Path:
        headers = {"User-Agent": CHROME_USER_AGENT}
        last_err: Exception | None = None
        for attempt in range(1, 4):
            try:
                with requests.get(img_url, headers=headers, timeout=20, stream=True) as response:
                    status_code = response.status_code
                    if status_code in (401, 403):
                        raise LoginRequiredError("login required")
                    response.raise_for_status()
                    ext = _http_extension_from_response(
                        img_url, response.headers.get("Content-Type"), item.preferred_ext
                    )
                    filename = f"{item.meta.get('id') or f'photo_{int(time.time() * 1000)}'}.{ext}"
                    destination = temp_dir / filename
                    with destination.open("wb") as output:
                        for chunk in response.iter_content(chunk_size=64 * 1024):
                            if not chunk:
                                continue
                            output.write(chunk)
                            if output.tell() > MAX_BOT_FILE_BYTES:
                                raise MediaDownloadError("media is too large for Telegram")
                    return destination
            except Exception as err:
                last_err = err
                if attempt >= 3:
                    break
                time.sleep(RETRY_PAUSE_SECONDS)
        raise MediaDownloadError(f"Direct photo download failed: {img_url}") from last_err

    return await asyncio.to_thread(_run_http)


async def download_media_item(item: MediaItem, temp_dir: Path, platform: str) -> Path:
    async def _download_with_ytdlp(prefer_photo: bool) -> Path:
        ydl_opts = _base_ydl_opts(platform, temp_dir=temp_dir, prefer_photo=prefer_photo)

        def _run() -> Path:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                try:
                    result_info = ydl.process_ie_result(item.meta, download=True)
                except Exception:
                    source_url = (
                        item.meta.get("webpage_url")
                        or item.meta.get("original_url")
                        or item.meta.get("url")
                    )
                    if not source_url:
                        raise
                    result_info = ydl.extract_info(source_url, download=True)
                if isinstance(result_info, list):
                    result_info = result_info[0] if result_info else {}
                if not isinstance(result_info, dict):
                    raise MediaDownloadError("yt-dlp returned unexpected result")
                file_path = _pick_downloaded_file(temp_dir, result_info, ydl)
                if not file_path or not file_path.exists():
                    raise FileNotFoundError("Downloaded file not found after yt-dlp run")
                return file_path

        return await asyncio.to_thread(_run)

    if item.estimated_filesize and item.estimated_filesize > MAX_BOT_FILE_BYTES:
        raise MediaDownloadError("media is too large for Telegram")

    if item.type == "video":
        file_path = await _download_with_ytdlp(prefer_photo=False)
        item.meta["_download_method"] = "yt-dlp"
        return file_path

    try:
        file_path = await _download_with_ytdlp(prefer_photo=True)
        item.meta["_download_method"] = "yt-dlp"
        return file_path
    except Exception as err:
        error_text = str(err).lower()
        if "no video formats" in error_text:
            logger.warning("No video formats; trigger photo-fallback")
        if not any(
            phrase in error_text
            for phrase in ("unsupported", "no video formats", "requested format is not available")
        ):
            raise
    file_path = await _direct_http_download_photo(item, temp_dir)
    item.meta["_download_method"] = "direct-http"
    return file_path


async def download_media_items(url: str, platform: str, temp_dir: Path) -> list[tuple[Path, str]]:
    items = await extract_media_items(url, platform)
    downloaded: list[tuple[Path, str]] = []
    for idx, item in enumerate(items, start=1):
        file_path = await download_media_item(item, temp_dir, platform)
        if not file_path.exists():
            raise FileNotFoundError("Downloaded file not found.")
        size_bytes = file_path.stat().st_size
        if size_bytes > MAX_BOT_FILE_BYTES:
            raise MediaDownloadError("media is too large for Telegram")
        method = item.meta.get("_download_method", "yt-dlp")
        logger.info(
            "Media item downloaded: download_method=%s size_bytes=%s index_in_carousel=%s",
            method,
            size_bytes,
            idx,
        )
        downloaded.append((file_path, item.type))
    return downloaded


async def download_video(url: str, platform: str, temp_dir: Path) -> Path:
    media_files = await download_media_items(url, platform, temp_dir)
    if len(media_files) == 1 and media_files[0][1] == "video":
        return media_files[0][0]
    first_video = next((path for path, media_type in media_files if media_type == "video"), None)
    if first_video:
        return first_video
    raise MediaDownloadError("No single video file available")


# -------------------------
# Bot State
# -------------------------
download_semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
inline_cache: dict[str, str] = {}
seen_urls: dict[str, float] = {}
processing_urls: set[str] = set()
url_dedupe_lock = asyncio.Lock()
download_queue_lock = asyncio.Lock()
next_download_allowed_at = 0.0


def _prune_seen_urls(now: float) -> None:
    expired_urls = [u for u, expires_at in seen_urls.items() if expires_at <= now]
    for url in expired_urls:
        seen_urls.pop(url, None)


async def try_acquire_url(url: str) -> bool:
    now = time.monotonic()
    async with url_dedupe_lock:
        _prune_seen_urls(now)
        if url in processing_urls:
            return False
        expires_at = seen_urls.get(url)
        if expires_at and expires_at > now:
            return False
        processing_urls.add(url)
        return True


async def release_url(url: str, mark_seen: bool = True) -> None:
    now = time.monotonic()
    async with url_dedupe_lock:
        processing_urls.discard(url)
        if mark_seen:
            seen_urls[url] = now + SEEN_URL_TTL_SECONDS
        _prune_seen_urls(now)


@contextlib.asynccontextmanager
async def global_download_slot():
    global next_download_allowed_at
    async with download_queue_lock:
        now = time.monotonic()
        wait_seconds = max(0.0, next_download_allowed_at - now)
        if wait_seconds > 0:
            logger.info("Waiting for global download cooldown: %.2fs", wait_seconds)
            await asyncio.sleep(wait_seconds)
        try:
            yield
        finally:
            next_download_allowed_at = time.monotonic() + GLOBAL_DOWNLOAD_COOLDOWN_SECONDS


async def safe_edit_status(status_msg, text: str) -> None:
    if not status_msg:
        return
    try:
        await status_msg.edit_text(text)
    except Exception as edit_err:
        logger.info("Could not edit status message: %s", edit_err)


def resolve_sender_name(user) -> str:
    if not user:
        return "unknown"
    if user.username:
        return f"@{user.username}"
    return user.full_name or "unknown"


def build_video_caption(source_url: str, sender_name: str, multiple_links: bool = False) -> str:
    safe_url = html.escape(source_url, quote=True)
    safe_sender_name = html.escape(sender_name)
    caption = f'<a href="{safe_url}">Post</a> sent by <b>{safe_sender_name}</b>'
    if multiple_links:
        caption += "\n\nOnly one link per message is supported."
    return caption


def _normalize_error_reason(raw_reason: str) -> str:
    reason = (raw_reason or "").strip()
    if not reason:
        return "unknown error"
    reason = reason.replace("\n", " ").replace("\r", " ")
    reason = re.sub(r"\s+", " ", reason).strip()
    return reason[:180]


def get_download_failure_reason(err: Exception, platform: str) -> str:
    error_chain = [str(err or "")]
    if getattr(err, "__cause__", None):
        error_chain.append(str(err.__cause__))
    full_error_text = " | ".join(error_chain)
    error_text = full_error_text.lower()

    login_required = any(
        phrase in error_text
        for phrase in (
            "login required",
            "sign in",
            "age-restricted",
            "age restricted",
            "confirm you're not a bot",
            "confirm you are not a bot",
        )
    )
    if login_required:
        cookiefile = get_cookiefile_for_platform(platform)
        if cookiefile and not cookiefile.exists():
            return f"login required: cookies file is missing ({cookiefile})"
        return "login required: cookies are missing or expired"

    if "downloaded file not found" in error_text:
        return "download failed: file not found after yt-dlp run"
    if "max-filesize" in error_text or "larger than max" in error_text:
        return "media is too large for Telegram"
    if "too large for telegram" in error_text:
        return "media is too large for Telegram"
    if "no media entries found" in error_text:
        return "no downloadable media found"

    return _normalize_error_reason(error_chain[-1] or error_chain[0])


def build_download_error_message(source_url: str, reason: str) -> str:
    safe_url = html.escape(source_url, quote=True)
    safe_reason = html.escape(_normalize_error_reason(reason))
    return f'Couldn\'t download <a href="{safe_url}">media</a>: <i>{safe_reason}</i>'


async def find_download_target_from_first_link(text: str) -> tuple[str | None, str | None, bool]:
    links = extract_links(text)
    if not links:
        return None, None, False

    raw_url = links[0]
    has_multiple_links = len(links) > 1
    final_url = await expand_url(raw_url)
    platform = detect_platform(final_url) or detect_platform(raw_url)
    if not platform:
        return None, None, has_multiple_links

    if not is_supported_download_url(final_url, platform):
        if platform == "youtube":
            logger.info("Skipping non-Shorts YouTube URL: original=%s final=%s", raw_url, final_url)
        if platform == "twitch":
            logger.info("Skipping non-clip Twitch URL: original=%s final=%s", raw_url, final_url)
        return None, None, has_multiple_links

    logger.info("Using first URL from message: original=%s final=%s platform=%s", raw_url, final_url, platform)
    return final_url, platform, has_multiple_links


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if not message or not message.text:
        return

    chat = update.effective_chat
    if not chat:
        return

    url, platform, has_multiple_links = await find_download_target_from_first_link(message.text)
    if not url or not platform:
        return

    acquired = await try_acquire_url(url)
    if not acquired:
        try:
            await message.delete()
        except Exception as delete_err:
            logger.info("Could not delete duplicate message: %s", delete_err)
        logger.info("Skipping duplicate URL: chat_id=%s url=%s", chat.id, url)
        return

    sender_name = resolve_sender_name(update.effective_user)
    video_caption = build_video_caption(url, sender_name, multiple_links=has_multiple_links)
    status_msg = None

    try:
        async with download_semaphore:
            async with global_download_slot():
                # Scenario 1:
                # 1) delete original message with link
                try:
                    await message.delete()
                except Exception as delete_err:
                    logger.info("Could not delete message: %s", delete_err)

                # 2) send short status immediately after queue wait
                status_msg = await chat.send_message("Processing...")
                logger.info("Download started: chat_id=%s url=%s", chat.id, url)

                with tempfile.TemporaryDirectory(prefix="video-bot-") as temp_dir_name:
                    temp_dir = Path(temp_dir_name)

                    try:
                        media_files = await download_media_items(url, platform, temp_dir)
                        if not media_files:
                            raise MediaDownloadError("No downloadable media found")

                        if len(media_files) == 1:
                            file_path, media_type = media_files[0]
                            logger.info("Sending single media: chat_id=%s url=%s type=%s", chat.id, url, media_type)
                            with file_path.open("rb") as f:
                                if media_type == "photo":
                                    await chat.send_photo(photo=f, caption=video_caption, parse_mode=ParseMode.HTML)
                                else:
                                    await chat.send_video(
                                        video=f,
                                        supports_streaming=True,
                                        caption=video_caption,
                                        parse_mode=ParseMode.HTML,
                                    )
                        else:
                            logger.info(
                                "Sending media group: chat_id=%s url=%s count=%s",
                                chat.id,
                                url,
                                len(media_files),
                            )
                            for chunk_start in range(0, len(media_files), 10):
                                chunk = media_files[chunk_start : chunk_start + 10]
                                opened_files = [file_path.open("rb") for file_path, _ in chunk]
                                try:
                                    media_group = []
                                    for idx, ((_, media_type), opened_file) in enumerate(zip(chunk, opened_files)):
                                        caption = video_caption if chunk_start == 0 and idx == 0 else None
                                        parse_mode = ParseMode.HTML if caption else None
                                        if media_type == "photo":
                                            media_group.append(
                                                InputMediaPhoto(media=opened_file, caption=caption, parse_mode=parse_mode)
                                            )
                                        else:
                                            media_group.append(
                                                InputMediaVideo(
                                                    media=opened_file,
                                                    caption=caption,
                                                    parse_mode=parse_mode,
                                                    supports_streaming=True,
                                                )
                                            )
                                    await chat.send_media_group(media=media_group)
                                finally:
                                    for opened_file in opened_files:
                                        opened_file.close()

                        try:
                            await status_msg.delete()
                            status_msg = None
                        except Exception as delete_err:
                            logger.info("Could not delete status message: %s", delete_err)
                    except Exception as e:
                        logger.error("Error handling URL %s: %s", url, e)
                        logger.error(traceback.format_exc())
                        reason = get_download_failure_reason(e, platform)
                        await safe_edit_status(status_msg, f"Failed: {reason}")
                        await chat.send_message(
                            build_download_error_message(url, reason),
                            parse_mode=ParseMode.HTML,
                            disable_web_page_preview=True,
                        )
    finally:
        await release_url(url)


async def handle_inline_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    inline_query = update.inline_query
    if not inline_query:
        return

    query = (inline_query.query or "").strip()
    if not query:
        return

    url, platform = await find_download_target(query)
    if not url or not platform:
        return

    user_id = inline_query.from_user.id
    sender_name = resolve_sender_name(inline_query.from_user)
    video_caption = build_video_caption(url, sender_name)

    # Scenario 2:
    # 1) user enters @bot + url
    # 2) bot downloads, sends to private chat, then returns cached video in inline
    cached_file_id = inline_cache.get(url)
    if cached_file_id:
        results = [
            InlineQueryResultCachedVideo(
                id=f"cached-{abs(hash(url))}",
                video_file_id=cached_file_id,
                title="Video",
            )
        ]
        await inline_query.answer(results=results, cache_time=60, is_personal=True)
        return

    async with download_semaphore:
        async with global_download_slot():
            with tempfile.TemporaryDirectory(prefix="video-bot-") as temp_dir_name:
                temp_dir = Path(temp_dir_name)
                try:
                    logger.info("Inline download started: user_id=%s url=%s", user_id, url)
                    media_files = await download_media_items(url, platform, temp_dir)
                    if not media_files:
                        raise MediaDownloadError("No downloadable media found")

                    if len(media_files) == 1 and media_files[0][1] == "video":
                        file_path, _ = media_files[0]
                        with file_path.open("rb") as f:
                            sent = await context.bot.send_video(
                                chat_id=user_id,
                                video=f,
                                supports_streaming=True,
                                caption=video_caption,
                                parse_mode=ParseMode.HTML,
                            )
                        file_id = sent.video.file_id if sent and sent.video else None
                        if not file_id:
                            raise RuntimeError("Failed to get file_id for inline result.")
                        inline_cache[url] = file_id
                        results = [
                            InlineQueryResultCachedVideo(
                                id=f"cached-{abs(hash(url))}",
                                video_file_id=file_id,
                                title="Video",
                            )
                        ]
                        await inline_query.answer(results=results, cache_time=60, is_personal=True)
                        return

                    if len(media_files) == 1:
                        file_path, media_type = media_files[0]
                        with file_path.open("rb") as f:
                            if media_type == "photo":
                                await context.bot.send_photo(
                                    chat_id=user_id,
                                    photo=f,
                                    caption=video_caption,
                                    parse_mode=ParseMode.HTML,
                                )
                            else:
                                await context.bot.send_video(
                                    chat_id=user_id,
                                    video=f,
                                    supports_streaming=True,
                                    caption=video_caption,
                                    parse_mode=ParseMode.HTML,
                                )
                    else:
                        for chunk_start in range(0, len(media_files), 10):
                            chunk = media_files[chunk_start : chunk_start + 10]
                            opened_files = [file_path.open("rb") for file_path, _ in chunk]
                            try:
                                media_group = []
                                for idx, ((_, media_type), opened_file) in enumerate(zip(chunk, opened_files)):
                                    caption = video_caption if chunk_start == 0 and idx == 0 else None
                                    parse_mode = ParseMode.HTML if caption else None
                                    if media_type == "photo":
                                        media_group.append(
                                            InputMediaPhoto(media=opened_file, caption=caption, parse_mode=parse_mode)
                                        )
                                    else:
                                        media_group.append(
                                            InputMediaVideo(
                                                media=opened_file,
                                                caption=caption,
                                                parse_mode=parse_mode,
                                                supports_streaming=True,
                                            )
                                        )
                                await context.bot.send_media_group(chat_id=user_id, media=media_group)
                            finally:
                                for opened_file in opened_files:
                                    opened_file.close()

                    await inline_query.answer(results=[], cache_time=1, is_personal=True)
                except Exception as e:
                    logger.error("Error handling inline URL %s: %s", url, e)
                    logger.error(traceback.format_exc())
                    await inline_query.answer(results=[], cache_time=1, is_personal=True)
                    try:
                        await context.bot.send_message(
                            chat_id=user_id,
                            text="Error while downloading media. Try again later.",
                        )
                    except Exception:
                        logger.error("Failed to notify user in private chat: user_id=%s", user_id)


async def log_heartbeat(_: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info("Bot is listening...")


# -------------------------
# Main
# -------------------------
def main() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN is required.")

    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .connect_timeout(60)
        .read_timeout(300)
        .write_timeout(300)
        .pool_timeout(60)
        .build()
    )

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(InlineQueryHandler(handle_inline_query))

    app.job_queue.run_repeating(log_heartbeat, interval=HEARTBEAT_INTERVAL_SECONDS, first=0)
    logger.info("Bot started. Polling and waiting for updates...")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
