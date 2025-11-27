from gtts import gTTS
import pyttsx3
from .detector import DetectionResult
from dataclasses import dataclass
import io

@dataclass
class TranslatedResult:
    text: str
    audio: Optional[bytes] = None  # For playback

class Translator:
    """
    Translates detected signs to text and speech.
    Supports English/Hindi for ISL.
    """
    def __init__(self, use_speech: bool = False, lang: str = "en"):
        self.use_speech = use_speech
        self.lang = lang
        if use_speech:
            self.tts_engine = pyttsx3.init()  # Offline
            # Or use gTTS for online: self.tts = gTTS(text, lang=lang)

    def translate(self, result: DetectionResult) -> TranslatedResult:
        # Simple: text is already translated in detector; add speech
        audio = None
        if self.use_speech and result.text:
            tts = gTTS(result.text, lang=self.lang)
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio = audio_buffer.getvalue()
            # For immediate play: self.tts_engine.say(result.text); self.tts_engine.runAndWait()
        
        return TranslatedResult(text=result.text, audio=audio)