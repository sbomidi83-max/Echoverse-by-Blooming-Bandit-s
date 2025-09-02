
EchoVerse - AI Audiobook Creator

Transform text into expressive audiobooks using state-of-the-art AI. This project integrates IBM Granite for intelligent text rewriting and leverages open-source TTS engines for high-quality audio generation. Designed for ease of deployment in both local and cloud environments.

Developed by Blooming Bandits.

Features

* Advanced Text Rewriting: Employs the IBM Granite 3B language model to dynamically rewrite input text according to user-selected tones (Neutral, Suspenseful, Inspiring).
* High-Quality Text-to-Speech (TTS): Generates natural-sounding audio using the open-source Coqui TTS library, providing superior quality compared to basic TTS solutions.
* Intuitive Web Interface: Built with Gradio, offering a user-friendly web UI for seamless interaction.
* Robust Fallback Mechanism: Incorporates gTTS as a reliable backup TTS engine to ensure functionality even if the primary TTS method encounters issues.
* Flexible Deployment: Easily runnable locally or deployable to platforms such as Hugging Face Spaces.

Installation and Setup

Follow these steps to set up and run EchoVerse.

Prerequisites

* Python 3.8 or higher
* pip (Python package installer)

Steps

1. Clone or Download the Repository
   Obtain the project files (app.py, requirements.txt).

2. Create a Virtual Environment (recommended)
   Isolate project dependencies to avoid conflicts.

   python -m venv echoverse-env

   # Activate the environment:

   # On Windows:

   echoverse-env\Scripts\activate

   # On macOS/Linux:

   source echoverse-env/bin/activate

3. Install Dependencies
   Use pip to install the required Python packages listed in requirements.txt.

   pip install -r requirements.txt

   Note for Hugging Face Spaces: This step is handled automatically by the platform upon pushing your requirements.txt file.

4. Configure Environment Variables (optional)
   If future features require API keys or specific settings, these can be managed via a .env file or the deployment platform's secrets management system (e.g., Hugging Face Space Settings).

5. Run the Application
   Execute the main application script.

   python app.py

   The Gradio interface will typically be accessible at [http://localhost:7860](http://localhost:7860) in your web browser.
   For Hugging Face Spaces: This command is executed automatically based on the app.py file.

How It Works

1. User Input: The user provides text via the Gradio UI.
2. Text Processing (IBM Granite):

   * The application loads the IBM Granite 3B model (if not already loaded).
   * Based on the user's selected tone, it crafts a specific prompt and utilizes the model to rewrite the text.
3. Audio Generation (Coqui TTS):

   * The original input text is passed to the Coqui TTS engine.
   * Coqui TTS generates a speech waveform and saves it as a .wav file.
   * If Coqui TTS encounters an error, the system automatically falls back to using gTTS to generate an .mp3 file.
4. User Output: The rewritten text and the generated audio file are displayed in the Gradio UI for the user.

Project Structure

* app.py: The main application file containing the Gradio interface and core logic for text processing and audio generation.
* requirements.txt: Lists all the Python libraries required to run the application.

Acknowledgements

* IBM Granite: For the powerful language model used for text rewriting.
* Coqui TTS: For the open-source text-to-speech engine providing more natural voices.
* Gradio: For the simple and effective way to create machine learning web demos.
* Hugging Face: For providing a platform to easily deploy and share machine learning demos (Hugging Face Spaces).

Thanks for using EchoVerse!
