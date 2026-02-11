"""
Usage (local):
  export BOT_TOKEN="..."
  python bot.py
"""

import asyncio
import contextlib
import html
import logging
import os
import re
import tempfile
import traceback
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import yt_dlp
from telegram import InlineQueryResultCachedVideo, Update
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
URL_EXPAND_TIMEOUT_SECONDS = float(os.getenv("URL_EXPAND_TIMEOUT_SECONDS", "12"))

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
    return platform == "tiktok"


def get_cookiefile_for_platform(platform: str) -> Path | None:
    if platform == "tiktok":
        return TIKTOK_COOKIES
    if platform == "instagram":
        return INSTAGRAM_COOKIES
    if platform == "youtube":
        return YOUTUBE_COOKIES
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
async def download_video(url: str, platform: str, temp_dir: Path) -> Path:
    """Run yt-dlp in a thread and return downloaded file path."""
    ydl_opts = {
        "outtmpl": str(temp_dir / "%(id)s.%(ext)s"),
        "format": "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        "noplaylist": True,
        "merge_output_format": "mp4",
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

    default_cookiefile = get_cookiefile_for_platform(platform)
    if default_cookiefile and default_cookiefile.exists():
        ydl_opts["cookiefile"] = str(default_cookiefile)
    elif default_cookiefile:
        logger.warning("Cookies file is missing for %s: %s", platform, default_cookiefile)

    def _run() -> str:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if "entries" in info and info["entries"]:
                info = info["entries"][0]
            return ydl.prepare_filename(info)

    last_error = None
    for attempt in range(1, DOWNLOAD_RETRY_ATTEMPTS + 1):
        try:
            filename = await asyncio.to_thread(_run)
            return Path(filename)
        except Exception as err:
            last_error = err
            error_text = str(err).lower()

            if "redirect" in error_text:
                logger.warning("Redirect-related yt-dlp error. final_url=%s error=%s", url, err)

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
            if login_required and default_cookiefile and "cookiefile" not in ydl_opts:
                if default_cookiefile.exists():
                    ydl_opts["cookiefile"] = str(default_cookiefile)
                    logger.warning(
                        "Retrying with cookies after login-required error: platform=%s cookies=%s",
                        platform,
                        default_cookiefile,
                    )
                    continue
                logger.warning(
                    "Login-required error but cookies are missing: platform=%s expected=%s",
                    platform,
                    default_cookiefile,
                )

            if attempt >= DOWNLOAD_RETRY_ATTEMPTS:
                break
            logger.warning(
                "Download attempt failed (attempt=%s/%s, platform=%s, url=%s): %s",
                attempt,
                DOWNLOAD_RETRY_ATTEMPTS,
                platform,
                url,
                err,
            )
            await asyncio.sleep(RETRY_PAUSE_SECONDS)

    raise RuntimeError(f"Failed to download after retries: {url}") from last_error


# -------------------------
# Bot State
# -------------------------
download_semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
inline_cache: dict[str, str] = {}


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


def build_video_caption(source_url: str, sender_name: str) -> str:
    safe_url = html.escape(source_url, quote=True)
    safe_sender_name = html.escape(sender_name)
    return f'<a href="{safe_url}">Video</a> sent by <b>{safe_sender_name}</b>'


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if not message or not message.text:
        return

    chat = update.effective_chat
    if not chat:
        return

    url, platform = await find_download_target(message.text)
    if not url or not platform:
        return
    sender_name = resolve_sender_name(update.effective_user)
    video_caption = build_video_caption(url, sender_name)

    # Scenario 1:
    # 1) delete original message with link
    try:
        await message.delete()
    except Exception as delete_err:
        logger.info("Could not delete message: %s", delete_err)

    # 2) send status immediately
    status_msg = await chat.send_message("Downloading video...")
    logger.info("Download started: chat_id=%s url=%s", chat.id, url)

    async with download_semaphore:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="video-bot-")
        temp_dir = Path(temp_dir_obj.name)

        try:
            file_path = await download_video(url, platform, temp_dir)
            if not file_path.exists():
                raise FileNotFoundError("Downloaded file not found.")

            file_size = file_path.stat().st_size
            logger.info("Download finished: chat_id=%s url=%s size_bytes=%s", chat.id, url, file_size)

            # 3) after download edit status
            await safe_edit_status(status_msg, "Video downloaded. Sending...")

            if file_size <= MAX_BOT_FILE_BYTES:
                # 4) success: send video then remove status
                logger.info("Sending video: chat_id=%s url=%s", chat.id, url)
                with file_path.open("rb") as f:
                    await chat.send_video(
                        video=f,
                        supports_streaming=True,
                        caption=video_caption,
                        parse_mode=ParseMode.HTML,
                    )
                try:
                    await status_msg.delete()
                    status_msg = None
                except Exception as delete_err:
                    logger.info("Could not delete status message: %s", delete_err)
            else:
                # 5) error/limit case: keep status and edit
                await safe_edit_status(
                    status_msg,
                    "Video is too large for Telegram. Download it from the original link.",
                )

        except Exception as e:
            logger.error("Error handling URL %s: %s", url, e)
            logger.error(traceback.format_exc())
            # 5) error: edit status
            await safe_edit_status(status_msg, "Download error. Try again later.")
        finally:
            try:
                temp_dir_obj.cleanup()
            except Exception:
                logger.error("Failed to clean temp dir: %s", temp_dir)


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
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="video-bot-")
        temp_dir = Path(temp_dir_obj.name)
        try:
            logger.info("Inline download started: user_id=%s url=%s", user_id, url)
            file_path = await download_video(url, platform, temp_dir)
            if not file_path.exists():
                raise FileNotFoundError("Downloaded file not found.")

            file_size = file_path.stat().st_size
            logger.info(
                "Inline download finished: user_id=%s url=%s size_bytes=%s",
                user_id,
                url,
                file_size,
            )
            if file_size > MAX_BOT_FILE_BYTES:
                await context.bot.send_message(
                    chat_id=user_id,
                    text="Video is too large for Telegram. Download it from the original link.",
                )
                await inline_query.answer(results=[], cache_time=1, is_personal=True)
                return

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
        except Exception as e:
            logger.error("Error handling inline URL %s: %s", url, e)
            logger.error(traceback.format_exc())
            await inline_query.answer(results=[], cache_time=1, is_personal=True)
            try:
                await context.bot.send_message(
                    chat_id=user_id,
                    text="Error while downloading video. Try again later.",
                )
            except Exception:
                logger.error("Failed to notify user in private chat: user_id=%s", user_id)
        finally:
            try:
                temp_dir_obj.cleanup()
            except Exception:
                logger.error("Failed to clean temp dir: %s", temp_dir)


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
