"""
Usage (local):
  export BOT_TOKEN="..."
  python bot.py
"""

import asyncio
import logging
import os
import re
import tempfile
import traceback
from pathlib import Path

import yt_dlp
from telegram import InlineQueryResultCachedVideo, Update
from telegram.ext import Application, ContextTypes, InlineQueryHandler, MessageHandler, filters

# -------------------------
# Configuration
# -------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
MAX_CONCURRENT_DOWNLOADS = int(os.getenv("MAX_CONCURRENT_DOWNLOADS", "2"))
MAX_BOT_FILE_BYTES = int(os.getenv("MAX_BOT_FILE_BYTES", str(50 * 1024 * 1024)))

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("video-bot")

# -------------------------
# URL Detection
# -------------------------
URL_REGEX = re.compile(r"(https?://[^\s]+)", re.IGNORECASE)

SUPPORTED_PATTERNS = [
    re.compile(r"https?://(www\.)?tiktok\.com/.*", re.IGNORECASE),
    re.compile(r"https?://(www\.)?instagram\.com/reels?/.*", re.IGNORECASE),
    re.compile(r"https?://(www\.)?youtube\.com/shorts/.*", re.IGNORECASE),
    re.compile(r"https?://(www\.)?youtu\.be/.*", re.IGNORECASE),
]


def extract_supported_links(text: str) -> list[str]:
    links = URL_REGEX.findall(text or "")
    return [link for link in links if any(p.match(link) for p in SUPPORTED_PATTERNS)]


# -------------------------
# Downloader
# -------------------------
async def download_video(url: str, temp_dir: Path) -> Path:
    """Run yt-dlp in a thread and return downloaded file path."""
    ydl_opts = {
        "outtmpl": str(temp_dir / "%(id)s.%(ext)s"),
        "format": "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        "noplaylist": True,
        "merge_output_format": "mp4",
        "retries": 2,
        "quiet": True,
        "no_warnings": True,
    }

    def _run() -> str:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if "entries" in info and info["entries"]:
                info = info["entries"][0]
            return ydl.prepare_filename(info)

    filename = await asyncio.to_thread(_run)
    return Path(filename)


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


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if not message or not message.text:
        return

    chat = update.effective_chat
    if not chat:
        return

    links = extract_supported_links(message.text)
    if not links:
        return

    url = links[0]

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
            file_path = await download_video(url, temp_dir)
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
                        caption="Here you go!",
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

    links = extract_supported_links(query)
    if not links:
        return

    user_id = inline_query.from_user.id
    url = links[0]

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
            file_path = await download_video(url, temp_dir)
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
                    caption="Here you go!",
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

    logger.info("Bot started. Polling...")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
