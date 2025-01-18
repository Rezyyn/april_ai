import os
import logging
import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from nemo.collections.tts.models import FastPitchModel
from nemo.collections.tts.models import HifiGanModel

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory to store audio outputs
AUDIO_OUTPUT_DIR = "./audio_outputs"
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="April_TTS_NeMo",
    description="Text-to-Speech (TTS) module using NVIDIA NeMo for April's voice.",
    version="1.0.0"
)

class TTSRequest(BaseModel):
    text: str  # Text to convert to speech
    output_filename: str = "april_tts.wav"  # Filename for the generated audio file

class TTSResponse(BaseModel):
    audio_path: str  # Path to the generated audio file

# Initialize NeMo models
logger.info("Loading NeMo FastPitch and HiFi-GAN models...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    # Load FastPitch model (text to spectrogram)
    fastpitch_model = FastPitchModel.from_pretrained("tts_en_fastpitch").to(device)

    # Load HiFi-GAN model (spectrogram to waveform)
    hifigan_model = HifiGanModel.from_pretrained("tts_hifigan").to(device)

    logger.info("Models loaded successfully.")
except Exception as e:
    logger.critical(f"Error loading models: {e}")
    raise RuntimeError(f"Failed to load NeMo models: {e}")

# Generate speech
def generate_tts_nemo(text: str, output_path: str):
    """
    Generate TTS audio using FastPitch and HiFi-GAN.
    """
    try:
        logger.info(f"Generating spectrogram for text: {text}")
        with torch.no_grad():
            # Convert text to tokens
            parsed = fastpitch_model.parse(text)

            # Generate mel-spectrogram
            spectrogram = fastpitch_model.generate_spectrogram(tokens=parsed)

            logger.info("Spectrogram generated. Converting to audio waveform...")
            # Convert mel-spectrogram to audio waveform
            audio = hifigan_model.convert_spectrogram_to_audio(spec=spectrogram)

        # Save audio to file
        sf.write(output_path, audio.cpu().numpy(), samplerate=22050)
        logger.info(f"Audio saved at: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error generating TTS audio: {e}")
        raise RuntimeError(f"Error generating TTS: {e}")

@app.post("/tts", response_model=TTSResponse)
async def tts_endpoint(req: TTSRequest):
    """
    Endpoint to generate TTS audio for the given text.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    # Output file path
    output_path = os.path.join(AUDIO_OUTPUT_DIR, req.output_filename)

    try:
        # Generate audio
        audio_path = generate_tts_nemo(req.text, output_path)
        return TTSResponse(audio_path=audio_path)
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8090)
