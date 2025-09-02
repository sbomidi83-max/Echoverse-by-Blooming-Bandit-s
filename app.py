# app.py
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import uuid
import sys
import logging

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("EchoVerse")

# --- IBM Granite Model Integration ---
class GraniteEngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self.load_model()

    def load_model(self):
        try:
            model_name = "ibm-granite/granite-3b-code-instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            )
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            self.is_loaded = True
            logger.info("IBM Granite model loaded")
        except Exception as e:
            logger.error(f"Error loading Granite: {e}")
            self.model = None
            self.tokenizer = None
            self.is_loaded = False

    def rewrite_text(self, text, tone):
        if not self.is_loaded or not self.model or not self.tokenizer:
            prefixes = {
                "neutral": "[NEUTRAL] ",
                "suspenseful": "[SUSPENSEFUL] ",
                "inspiring": "[INSPIRING] "
            }
            return prefixes.get(tone, "") + text

        try:
            base_prompts = {
                "neutral": "Rewrite clearly: ",
                "suspenseful": "Make suspenseful: ",
                "inspiring": "Make inspiring: "
            }
            prompt = base_prompts.get(tone, "Rewrite: ") + text[:500]

            inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
            if torch.cuda.is_available():
                inputs = inputs.cuda()

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=min(200, len(text)//2),
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if result.startswith(prompt):
                result = result[len(prompt):].strip()

            if not result or len(result) < 10:
                result = text

            return result

        except Exception as e:
            logger.error(f"Granite error: {e}")
            return "[ERROR] " + text

# --- Updated Audio Generation using Coqui TTS ---
def generate_human_audio(text):
    if not text:
        return None

    try:
        text_to_speak = text[:300] # Limit text for Coqui TTS

        logger.info("Initializing Coqui TTS...")
        
        # Import Coqui TTS inside the function to handle potential import errors gracefully
        from TTS.api import TTS
        
        # List available models (optional, for debugging/info)
        # logger.info("Available Coqui TTS models:")
        # for model_name in TTS().list_models():
        #     logger.info(f" - {model_name}")

        # Initialize a Coqui TTS model
        # Using a lightweight, English multi-speaker model for compatibility and speed.
        # You can experiment with different models from the list above.
        model_name = "tts_models/en/ljspeech/tacotron2-DDC"
        logger.info(f"Loading Coqui TTS model: {model_name}")
        tts = TTS(model_name=model_name, progress_bar=False, gpu=False) # Set gpu=True if CUDA is available and you want to use it

        # Generate audio
        logger.info("Generating speech with Coqui TTS...")
        filename = f"coqui_audio_{uuid.uuid4().hex}.wav" # Coqui often outputs WAV
        tts.tts_to_file(text=text_to_speak, file_path=filename)
        
        logger.info(f"Coqui TTS audio saved as {filename}")
        return filename

    except ImportError as e:
        logger.error(f"Coqui TTS library not found or could not be imported: {e}")
    except Exception as e:
        logger.error(f"Error in Coqui TTS generation: {e}")

    # --- Fallback to gTTS if Coqui fails ---
    logger.warning("Falling back to gTTS.")
    try:
        # Redefine text_to_speak for clarity in fallback
        fallback_text = (text[:300] if text else "Error generating audio.")
        from gtts import gTTS
        tts = gTTS(text=fallback_text, lang='en', slow=False)
        filename = f"fallback_audio_{uuid.uuid4().hex}.mp3"
        tts.save(filename)
        logger.info(f"Fallback audio saved as {filename}")
        return filename
    except Exception as fallback_e:
        logger.error(f"Fallback gTTS also failed: {fallback_e}")
        error_filename = f"error_{uuid.uuid4().hex}.txt"
        with open(error_filename, 'w') as f:
            f.write("Audio generation failed.")
        return error_filename


# --- Gradio Interface ---
logger.info("Initializing Granite Engine...")
granite_engine = GraniteEngine()

with gr.Blocks(title="EchoVerse") as demo:
    gr.Markdown("# 🌊 EchoVerse - AI Audiobook")
    gr.Markdown("by Blooming Bandits")

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Input Text", lines=6, placeholder="Enter text...")
            tone = gr.Radio(
                choices=["neutral", "suspenseful", "inspiring"],
                value="neutral",
                label="Tone"
            )
            process_btn = gr.Button("Process Text", variant="primary")

        with gr.Column():
            output_text = gr.Textbox(label="Processed Text", lines=6)
            audio_btn = gr.Button("🔊 Generate Audiobook")
            audio_output = gr.Audio(label="Audiobook", type="filepath")

    process_btn.click(
        fn=lambda text, tone: granite_engine.rewrite_text(text, tone),
        inputs=[input_text, tone],
        outputs=output_text
    )

    audio_btn.click(
        fn=generate_human_audio,
        inputs=input_text,
        outputs=audio_output
    )

# --- Launch the app ---
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
