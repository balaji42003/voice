from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from langdetect import detect
import os
import tempfile
import whisper
from TTS.api import TTS
from transformers import AutoModel, AutoProcessor, pipeline
import torch
import shutil
import google.generativeai as genai
from googletrans import Translator
from gtts import gTTS
import speech_recognition as sr
from pydub import AudioSegment

app = Flask(__name__)
CORS(app)

# FFmpeg path fix for TTS (Render compatible)
ffmpeg_path = shutil.which("ffmpeg")
if ffmpeg_path:
    os.environ["COQUI_TTS_FFMPEG"] = ffmpeg_path

# Configure Gemini
genai.configure(api_key="AIzaSyDL8euQekfkLNJ5E2fPGukDd-0H9mMstrc")  # Replace with your actual key in Render secrets
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

translator = Translator()
whisper_model = whisper.load_model("base")
coqui_tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=False)

SYSTEM_PROMPT = """
You are a smart agricultural assistant for Indian farmers.
Your role is to help with the following farming topics only:
- Crop management
- Pest control
- Weather impact
- Market prices
You must respond strictly in the **same language as the user's input**, without switching, mixing, or translating to any other language — under any circumstance.
If the user asks in English, reply entirely in English.  
If the user asks in Hindi, reply entirely in Hindi.  
If the user asks in Telugu, reply entirely in Telugu.  
Do not mix languages in any way.
Strictly dont use special characters in script only the language and no numbering.
Do NOT use bullet points, numbering, or special characters.
Do NOT add introductions or conclusions.
Just give a clean, plain, practical answer in paragraph form
The **text of your response must be fully in that same language**, with no foreign or untranslated words.
Use local Indian farming terms, examples, and measurements. Keep the content focused only on the user's query — no extra information, no introductions, no general tips unless asked.
"""

def limit_tts_text(text, max_chars=500):
    if len(text) <= max_chars:
        return text
    sentences = text.split('. ')
    result = ''
    for sentence in sentences:
        if len(result) + len(sentence) + 2 > max_chars:
            break
        result += sentence + '. '
    return result.strip()

@app.route('/process_text', methods=['POST'])
def process_text():
    user_text = request.json.get('text')
    print("Received text:", user_text)

    if not user_text or not user_text.strip():
        return jsonify({"error": "Empty input"}), 400

    detected_lang = detect(user_text)
    print("Detected language:", detected_lang)

    custom_prompt = SYSTEM_PROMPT
    if detected_lang != 'en':
        custom_prompt += f"\n\nPlease respond in {detected_lang}."
    custom_prompt += (
        "\n\nWrite the answer as 4 short paragraphs, each a few sentences. "
        "Do NOT use bullet points, numbering, or special characters. "
        "Do NOT add introductions or conclusions. "
        "Just give a clean, plain, practical answer in paragraph form."
    )

    try:
        response_text = gemini_model.generate_content(
            custom_prompt + "\n\nFarmer's question: " + user_text
        ).text
    except Exception as e:
        print(f"Error generating Gemini response: {e}")
        return jsonify({"error": "Failed to generate response"}), 500

    # If Gemini responds in English but input was not English, translate back
    if detected_lang != 'en':
        resp_lang = detect(response_text)
        if resp_lang != detected_lang:
            response_text = translator.translate(response_text, dest=detected_lang).text

    # Generate TTS in a temp file and stream it
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio_path = temp_audio.name
        if detected_lang in ["te", "ta", "kn", "hi"]:
            tts = gTTS(text=response_text, lang=detected_lang)
            tts.save(temp_audio_path)
        elif detected_lang == "en":
            speaker = coqui_tts.speakers[0]  # Use first speaker for safety
            temp_audio_path = temp_audio_path.replace(".mp3", ".wav")
            coqui_tts.tts_to_file(
                text=limit_tts_text(response_text),
                file_path=temp_audio_path,
                speaker=speaker,
                language='en'
            )
        else:
            temp_audio_path = None

        # Stream audio file and delete after sending
        if temp_audio_path and os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 1000:
            return send_file(temp_audio_path, mimetype="audio/mpeg" if temp_audio_path.endswith(".mp3") else "audio/wav", as_attachment=True)
        else:
            return jsonify({
                "response": response_text,
                "audio": None,
                "detected_language": detected_lang
            })
    finally:
        if 'temp_audio_path' in locals() and temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file"}), 400

    file = request.files['audio']
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_input:
        input_path = temp_input.name
    file.save(input_path)
    wav_path = input_path.replace(".mp3", ".wav")

    try:
        audio = AudioSegment.from_file(input_path)
        audio.export(wav_path, format="wav")

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            try:
                user_text = recognizer.recognize_google(audio_data)
                print("🗣️ Recognized Text:", user_text)
            except sr.UnknownValueError:
                return jsonify({"error": "Speech not recognized."}), 400
            except sr.RequestError as e:
                return jsonify({"error": f"API unavailable: {e}"}), 500

        detected_lang = detect(user_text)
        print("🌐 Detected Language:", detected_lang)

        requested_lang = None
        if "in telugu" in user_text.lower():
            requested_lang = "te"
        elif "in hindi" in user_text.lower():
            requested_lang = "hi"
        elif "in tamil" in user_text.lower():
            requested_lang = "ta"
        elif "in kannada" in user_text.lower():
            requested_lang = "kn"

        if requested_lang:
            prompt = (
                f"Reply ONLY in {requested_lang} language, using only {requested_lang} script, "
                f"to this farming question. "
                "Write the answer as 4 short paragraphs, each a few sentences. "
                "Do NOT use bullet points, numbering, or special characters. "
                "Do NOT add introductions or conclusions. "
                "Just give a clean, plain, practical answer in paragraph form.\n"
                f"{user_text}"
            )
            tts_lang = requested_lang
        else:
            prompt = (
                f"Reply ONLY in {detected_lang} language, using only {detected_lang} script, "
                f"to this farming question. "
                "Write the answer as 4 short paragraphs, each a few sentences. "
                "Do NOT use bullet points, numbering, or special characters. "
                "Do NOT add introductions or conclusions. "
                "Just give a clean, plain, practical answer in paragraph form.\n"
                f"{user_text}"
            )
            tts_lang = detected_lang

        response = gemini_model.generate_content(prompt)
        reply_text = response.text.strip()
        print("🤖 Gemini Reply:", reply_text)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio_path = temp_audio.name
        tts = gTTS(text=reply_text, lang=tts_lang)
        tts.save(temp_audio_path)

        # Stream audio file and delete after sending
        if os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 1000:
            return send_file(temp_audio_path, mimetype="audio/mpeg", as_attachment=True)
        else:
            return jsonify({
                "recognized": user_text,
                "response": reply_text,
                "audio": None,
                "detected_language": detected_lang
            })
    except Exception as e:
        print(f"Error processing audio: {e}", flush=True)
        import traceback; traceback.print_exc()
        return jsonify({"error": "Audio processing failed"}), 500
    finally:
        for path in [input_path, wav_path]:
            if os.path.exists(path):
                os.remove(path)
        if 'temp_audio_path' in locals() and temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))  # Render sets PORT env var
    app.run(host="0.0.0.0", port=port)