import subprocess
import sys
import os
from dotenv import load_dotenv

# ---------------- Imports ----------------
import gradio as gr
import google.generativeai as genai
from PIL import Image
import PyPDF2
from typing import Tuple

# ---------------- Chatbot Class ----------------
class GeminiChatbot:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.text_model = None
        self.vision_model = None
        self.chat_history = []
        self.pdf_text = ""

        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.text_model = genai.GenerativeModel('gemini-2.5-flash')
                self.vision_model = genai.GenerativeModel('gemini-2.5-flash')
                print("✅ API key loaded from .env and models ready.")
            except Exception as e:
                print(f"❌ Failed to configure API key: {e}")

    # ---------- API Setup ----------
    def setup_api_key(self, api_key: str) -> str:
        try:
            if not api_key.strip():
                return "❌ Please enter a valid API key"

            genai.configure(api_key=api_key)
            self.api_key = api_key
            self.text_model = genai.GenerativeModel('gemini-2.5-flash')
            self.vision_model = genai.GenerativeModel('gemini-2.5-flash')
            response = self.text_model.generate_content("Hello")
            return "✅ API key configured successfully! Models are ready."
        except Exception as e:
            return f"❌ Error setting up API key: {str(e)}"

    # ---------- Text Generation ----------
    def text_generation(self, prompt: str, temperature: float = 0.7) -> str:
        try:
            if not self.text_model:
                return "❌ Please configure your API key first"
            if not prompt.strip():
                return "❌ Please enter a prompt"

            gen_config = genai.types.GenerationConfig(temperature=temperature, max_output_tokens=2048)
            response = self.text_model.generate_content(prompt, generation_config=gen_config)
            return f"{response.text}"
        except Exception as e:
            return f"❌ Error generating text: {str(e)}"

    # ---------- Translation ----------
    def translate_text(self, text: str, target_language: str) -> str:
        try:
            if not self.text_model:
                return "❌ Please configure your API key first"
            if not text.strip():
                return "❌ Please enter text to translate"

            prompt = f"Translate the following text to {target_language}. Only provide the translation:\n\n{text}"
            response = self.text_model.generate_content(prompt)
            return f"{response.text}"
        except Exception as e:
            return f"❌ Error translating text: {str(e)}"

    # ---------- Image Analysis ----------
    def analyze_image(self, image, prompt: str = "Describe this image in detail") -> str:
        try:
            if not self.vision_model:
                return "❌ Please configure your API key first"
            if image is None:
                return "❌ Please upload an image"
            if isinstance(image, str):
                image = Image.open(image)
            elif hasattr(image, 'convert'):
                pass
            else:
                return "❌ Invalid image format"

            response = self.vision_model.generate_content([prompt, image])
            return f"{response.text}"
        except Exception as e:
            return f"❌ Error analyzing image: {str(e)}"

    # ---------- Conversational Chat ----------
    def chat_with_history(self, message: str, history: list) -> Tuple[str, list]:
        try:
            if not self.text_model:
                return "❌ Please configure your API key first", history
            if not message.strip():
                return "❌ Please enter a message", history

            history.append({"role":"user", "content":message})
            context = ""
            for msg in history[:-1]:
                role = msg["role"].capitalize()
                context += f"{role}: {msg['content']}\n\n"
            full_prompt = f"{context}User: {message}\nAssistant:"

            response = self.text_model.generate_content(full_prompt)
            bot_response = response.text
            history.append({"role":"assistant", "content":bot_response})
            return "", history
        except Exception as e:
            history.append({"role":"assistant", "content":f"❌ Error: {str(e)}"} )
            return "", history

    # ---------- PDF Functions ----------
    def extract_pdf_text(self, pdf_file) -> str:
        try:
            if not pdf_file:
                return "❌ Please upload a PDF file"

            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            self.pdf_text = text.strip()
            return "✅ PDF loaded and text extracted successfully!"
        except Exception as e:
            return f"❌ Error extracting PDF: {str(e)}"

    def ask_pdf_question(self, question: str) -> str:
        try:
            if not self.pdf_text:
                return "❌ Please upload a PDF first"
            if not question.strip():
                return "❌ Please enter a question"

            prompt = f"Answer the question based on the following PDF content:\n{self.pdf_text}\n\nQuestion: {question}\nAnswer:"
            response = self.text_model.generate_content(prompt)
            return f"{response.text}"
        except Exception as e:
            return f"❌ Error answering PDF question: {str(e)}"

# ---------------- Initialize Chatbot ----------------
chatbot = GeminiChatbot()

# ---------------- Language Options ----------------
LANGUAGES = ["Spanish", "French", "German", "Italian", "Portuguese",
             "Russian", "Chinese (Simplified)", "Chinese (Traditional)",
             "Japanese", "Korean"]

# ---------------- Gradio Interface ----------------
def create_advanced_interface():
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="purple")) as demo:
        # Global font styling
        gr.HTML("""
        <style>
        * {
            font-family: 'Times New Roman', serif !important;
            font-size: 22px !important;
        }
        </style>
        """)

        # Centered Title Only
        gr.HTML("""
        <div style="text-align:center; font-size:28px; font-weight:bold; margin-top:20px; margin-bottom:20px;">
        🤖 STUDYMATE: An AI-Powered Chatbot for Students
        </div>
        """)

        # API Key
        with gr.Row():
            api_key_input = gr.Textbox(label="🔑 Gemini API Key", type="password", scale=3)
            setup_btn = gr.Button("🚀 Setup API Key", variant="primary", scale=1)
        api_status = gr.Textbox(label="API Status", value="⏳ Please configure your API key", interactive=False)
        setup_btn.click(fn=chatbot.setup_api_key, inputs=[api_key_input], outputs=[api_status])

        gr.Markdown("---")

        # Format output
        def format_output(text: str) -> str:
            return f"<div>{text.replace(chr(10), '<br>')}</div>"

        with gr.Tabs():
            # Text Generation
            with gr.TabItem("💬 Text Generation"):
                text_prompt = gr.Textbox(label="Enter your prompt", lines=3)
                temperature = gr.Slider(0.0, 1.0, value=0.7, step=0.1, label="Temperature")
                text_output = gr.Markdown()
                generate_btn = gr.Button("✨ Generate")
                generate_btn.click(fn=lambda prompt, temp: format_output(chatbot.text_generation(prompt, temp)),
                                   inputs=[text_prompt, temperature], outputs=[text_output])

            # Translation
            with gr.TabItem("🌍 Language Translation"):
                translate_input = gr.Textbox(label="Text to Translate", lines=4)
                target_lang = gr.Dropdown(LANGUAGES, label="Target Language", value="Spanish")
                translate_output = gr.Markdown()
                translate_btn = gr.Button("🔄 Translate")
                translate_btn.click(fn=lambda text, lang: format_output(chatbot.translate_text(text, lang)),
                                    inputs=[translate_input, target_lang], outputs=[translate_output])

            # Image Analysis
            with gr.TabItem("🖼️ Image Analysis"):
                image_input = gr.Image(label="Upload Image", type="pil")
                image_prompt = gr.Textbox(label="Question about the image", value="Describe this image in detail")
                image_output = gr.Markdown()
                analyze_btn = gr.Button("🔍 Analyze Image")
                analyze_btn.click(fn=lambda img, prompt: format_output(chatbot.analyze_image(img, prompt)),
                                  inputs=[image_input, image_prompt], outputs=[image_output])

            # Chat
            with gr.TabItem("💭 Conversational Chat"):
                chatbot_interface = gr.Chatbot(label="Chat History", type="messages", height=400)
                chat_input = gr.Textbox(label="Your message")
                chat_btn = gr.Button("📤 Send")
                clear_btn = gr.Button("🗑️ Clear History")
                def clear_chat(): return [], ""
                chat_btn.click(fn=chatbot.chat_with_history, inputs=[chat_input, chatbot_interface], outputs=[chat_input, chatbot_interface])
                chat_input.submit(fn=chatbot.chat_with_history, inputs=[chat_input, chatbot_interface], outputs=[chat_input, chatbot_interface])
                clear_btn.click(fn=clear_chat, outputs=[chatbot_interface, chat_input])

            # PDF Q&A
            with gr.TabItem("📄 PDF Q&A"):
                pdf_input = gr.File(label="Upload PDF", type="filepath", file_types=[".pdf"])
                load_pdf_btn = gr.Button("📥 Load PDF")
                pdf_status = gr.Textbox(label="PDF Status", value="No PDF uploaded yet", interactive=False)
                load_pdf_btn.click(fn=chatbot.extract_pdf_text, inputs=[pdf_input], outputs=[pdf_status])

                pdf_question = gr.Textbox(label="Ask a question about PDF", lines=2)
                pdf_answer = gr.Markdown()
                ask_pdf_btn = gr.Button("❓ Ask")
                ask_pdf_btn.click(fn=lambda q: format_output(chatbot.ask_pdf_question(q)), inputs=[pdf_question], outputs=[pdf_answer])

    demo.launch()

# ---------------- Launch ----------------
if __name__ == "__main__":
    create_advanced_interface()
