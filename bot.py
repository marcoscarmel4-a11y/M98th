import os
import asyncio
import logging

from aiohttp import web
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from openai import AsyncOpenAI


# ============================================================
# CONFIG
# ============================================================

BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PORT = int(os.getenv("PORT", "8080"))

# Use a current OpenAI model available to your API account.
AI_MODEL = os.getenv("AI_MODEL", "gpt-5.6-luna")


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# REQUIRED ENVIRONMENT VARIABLES
# ============================================================

if not BOT_TOKEN:
    raise RuntimeError(
        "BOT_TOKEN is missing. "
        "Add BOT_TOKEN to Railway Variables."
    )

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is missing. "
        "Add OPENAI_API_KEY to Railway Variables."
    )


# ============================================================
# CLIENTS
# ============================================================

bot = Bot(token=BOT_TOKEN)

dp = Dispatcher()

ai = AsyncOpenAI(
    api_key=OPENAI_API_KEY
)


# ============================================================
# MAIN MENU
# ============================================================

def main_menu():

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🤖 Ask AI",
                    callback_data="ask_ai"
                )
            ],
            [
                InlineKeyboardButton(
                    text="✍️ Writing",
                    callback_data="writing"
                ),
                InlineKeyboardButton(
                    text="📝 Summarize",
                    callback_data="summarize"
                )
            ],
            [
                InlineKeyboardButton(
                    text="🌍 Translate",
                    callback_data="translate"
                ),
                InlineKeyboardButton(
                    text="💡 Ideas",
                    callback_data="ideas"
                )
            ],
            [
                InlineKeyboardButton(
                    text="ℹ️ About",
                    callback_data="about"
                ),
                InlineKeyboardButton(
                    text="🔐 Privacy",
                    callback_data="privacy"
                )
            ]
        ]
    )


# ============================================================
# TOOLS MENU
# ============================================================

def tools_menu():

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🤖 Ask AI",
                    callback_data="ask_ai"
                )
            ],
            [
                InlineKeyboardButton(
                    text="✍️ Writing",
                    callback_data="writing"
                ),
                InlineKeyboardButton(
                    text="📝 Summarize",
                    callback_data="summarize"
                )
            ],
            [
                InlineKeyboardButton(
                    text="🌍 Translate",
                    callback_data="translate"
                ),
                InlineKeyboardButton(
                    text="💡 Ideas",
                    callback_data="ideas"
                )
            ],
            [
                InlineKeyboardButton(
                    text="🏠 Main Menu",
                    callback_data="main"
                )
            ]
        ]
    )


# ============================================================
# /START
# ============================================================

@dp.message(Command("start"))
async def start_command(message: types.Message):

    text = (
        "👋 Welcome to AI Assistant.\n\n"
        "Ask questions, improve your writing, "
        "summarize text, translate content, or "
        "generate ideas.\n\n"
        "Choose a tool below or simply send me a message."
    )

    await message.answer(
        text,
        reply_markup=main_menu()
    )


# ============================================================
# /HELP
# ============================================================

@dp.message(Command("help"))
async def help_command(message: types.Message):

    text = (
        "❓ Help\n\n"
        "/start — Open the main menu\n"
        "/help — Show help\n"
        "/about — About this bot\n"
        "/privacy — Privacy information\n\n"
        "You can also simply send a message "
        "and ask the AI assistant a question."
    )

    await message.answer(text)


# ============================================================
# /ABOUT
# ============================================================

@dp.message(Command("about"))
async def about_command(message: types.Message):

    text = (
        "ℹ️ About AI Assistant\n\n"
        "AI Assistant is an independent third-party "
        "Telegram bot that provides AI-powered help "
        "for writing, summarization, translation, "
        "brainstorming and general questions.\n\n"
        "The bot is not affiliated with Telegram."
    )

    await message.answer(
        text,
        reply_markup=main_menu()
    )


# ============================================================
# /PRIVACY
# ============================================================

@dp.message(Command("privacy"))
async def privacy_command(message: types.Message):

    text = (
        "🔐 Privacy\n\n"
        "Please do not send passwords, authentication "
        "codes, payment card details, private keys or "
        "other sensitive credentials.\n\n"
        "Messages sent to this bot may be processed by "
        "an external AI service to generate a response.\n\n"
        "Use the bot only for information you are "
        "comfortable sharing with the service."
    )

    await message.answer(text)


# ============================================================
# MAIN MENU BUTTON
# ============================================================

@dp.callback_query(lambda c: c.data == "main")
async def main_button(callback: types.CallbackQuery):

    await callback.message.edit_text(
        "🏠 Main Menu\n\n"
        "Choose a tool or send me a message.",
        reply_markup=main_menu()
    )

    await callback.answer()


# ============================================================
# ASK AI
# ============================================================

@dp.callback_query(lambda c: c.data == "ask_ai")
async def ask_ai_button(callback: types.CallbackQuery):

    text = (
        "🤖 Ask AI\n\n"
        "Send me any question or task.\n\n"
        "Examples:\n"
        "• Explain a difficult topic\n"
        "• Write an email\n"
        "• Generate ideas\n"
        "• Help me plan a project"
    )

    await callback.message.edit_text(
        text,
        reply_markup=tools_menu()
    )

    await callback.answer()


# ============================================================
# WRITING
# ============================================================

@dp.callback_query(lambda c: c.data == "writing")
async def writing_button(callback: types.CallbackQuery):

    text = (
        "✍️ Writing Assistant\n\n"
        "Send me your text and tell me what you want "
        "to improve.\n\n"
        "For example:\n"
        "Make this email more professional."
    )

    await callback.message.edit_text(
        text,
        reply_markup=tools_menu()
    )

    await callback.answer()


# ============================================================
# SUMMARIZE
# ============================================================

@dp.callback_query(lambda c: c.data == "summarize")
async def summarize_button(callback: types.CallbackQuery):

    text = (
        "📝 Summarize\n\n"
        "Send me an article, paragraph or other text "
        "and ask me to summarize it."
    )

    await callback.message.edit_text(
        text,
        reply_markup=tools_menu()
    )

    await callback.answer()


# ============================================================
# TRANSLATE
# ============================================================

@dp.callback_query(lambda c: c.data == "translate")
async def translate_button(callback: types.CallbackQuery):

    text = (
        "🌍 Translation\n\n"
        "Send the text and specify the target language.\n\n"
        "Example:\n"
        "Translate this to French: Hello, how are you?"
    )

    await callback.message.edit_text(
        text,
        reply_markup=tools_menu()
    )

    await callback.answer()


# ============================================================
# IDEAS
# ============================================================

@dp.callback_query(lambda c: c.data == "ideas")
async def ideas_button(callback: types.CallbackQuery):

    text = (
        "💡 Ideas\n\n"
        "Tell me what you are working on and "
        "I can help you brainstorm ideas."
    )

    await callback.message.edit_text(
        text,
        reply_markup=tools_menu()
    )

    await callback.answer()


# ============================================================
# ABOUT BUTTON
# ============================================================

@dp.callback_query(lambda c: c.data == "about")
async def about_button(callback: types.CallbackQuery):

    text = (
        "ℹ️ About AI Assistant\n\n"
        "An independent AI assistant for questions, "
        "writing, summaries, translations and ideas.\n\n"
        "This bot is not affiliated with Telegram."
    )

    await callback.message.edit_text(
        text,
        reply_markup=main_menu()
    )

    await callback.answer()


# ============================================================
# PRIVACY BUTTON
# ============================================================

@dp.callback_query(lambda c: c.data == "privacy")
async def privacy_button(callback: types.CallbackQuery):

    text = (
        "🔐 Privacy\n\n"
        "Do not send passwords, OTP codes, payment "
        "details or private keys.\n\n"
        "Messages may be processed by an external AI "
        "service to provide responses."
    )

    await callback.message.edit_text(
        text,
        reply_markup=main_menu()
    )

    await callback.answer()


# ============================================================
# AI RESPONSE
# ============================================================

async def ask_ai(user_text: str) -> str:

    response = await ai.responses.create(
        model=AI_MODEL,
        instructions=(
            "You are a helpful general-purpose AI assistant "
            "inside a Telegram bot. "
            "Be accurate, concise and clear. "
            "Do not claim to be Telegram or affiliated with Telegram. "
            "Do not ask users for passwords, OTPs or private credentials."
        ),
        input=user_text
    )

    return response.output_text


# ============================================================
# NORMAL TEXT MESSAGES
# ============================================================

@dp.message()
async def text_message(message: types.Message):

    if not message.text:
        await message.answer(
            "Please send a text message."
        )
        return

    user_text = message.text.strip()

    if not user_text:
        return

    # Telegram message processing indicator
    await bot.send_chat_action(
        chat_id=message.chat.id,
        action="typing"
    )

    try:

        answer = await ask_ai(user_text)

        if not answer:
            answer = (
                "I couldn't generate a response. "
                "Please try again."
            )

        # Telegram has message-size limits, so split long answers.
        max_length = 4000

        for i in range(0, len(answer), max_length):

            await message.answer(
                answer[i:i + max_length]
            )

    except Exception as error:

        logger.exception(
            "AI request failed: %s",
            error
        )

        await message.answer(
            "Sorry, I couldn't process that request "
            "right now. Please try again."
        )


# ============================================================
# RAILWAY HEALTH SERVER
# ============================================================

async def health(request):

    return web.Response(
        text="AI Assistant Bot is running",
        status=200
    )


async def start_web_server():

    app = web.Application()

    app.router.add_get("/", health)
    app.router.add_get("/health", health)

    runner = web.AppRunner(app)

    await runner.setup()

    site = web.TCPSite(
        runner,
        host="0.0.0.0",
        port=PORT
    )

    await site.start()

    logger.info(
        "Railway health server running on port %s",
        PORT
    )


# ============================================================
# MAIN
# ============================================================

async def main():

    logger.info("Starting AI Assistant Bot...")

    await start_web_server()

    # Ensure polling can be used.
    await bot.delete_webhook(
        drop_pending_updates=True
    )

    logger.info(
        "Starting Telegram polling..."
    )

    await dp.start_polling(bot)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:
        asyncio.run(main())

    except KeyboardInterrupt:

        logger.info("Bot stopped.")
