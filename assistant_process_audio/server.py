import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Disable symlinks in Hugging Face Hub
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"


import logging
import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
import asyncio
import uuid
from starlette.applications import Starlette
from starlette.websockets import WebSocket
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline, Model
from pyannote.core import Annotation, Segment
import uvicorn
from huggingface_hub import login
from speechbrain.utils.fetching import LocalStrategy
from .database import get_users_voice_recognition
import io
import json
import tempfile

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("whisper_server")

# Initialize the device
device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)
logger.info(f"Using device: {device}")

# Initialize the Whisper model
model_size = os.getenv("WHISPER_MODEL", "medium.en")
model = WhisperModel(model_size, device=device_str)
logger.info(f"Loaded Faster Whisper model: {model_size} on {device}")

# Initialize the pyannote pipelines
# Remember to set your Hugging Face authentication token
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=HUGGINGFACE_TOKEN,
)
diarization_pipeline.to(device)
embedding_model = Model.from_pretrained(
    "pyannote/embedding",
    use_auth_token=HUGGINGFACE_TOKEN,
    huggingface_cache_dir=None
)
embedding_model.eval()
embedding_model.to(device)

vad_pipeline = Pipeline.from_pretrained(
    "pyannote/voice-activity-detection",
    use_auth_token=HUGGINGFACE_TOKEN
)
vad_pipeline.to(device)
logger.info("Loaded pyannote VAD pipeline")

logger.info("Loaded pyannote speaker diarization and embedding models")

# Load reference audio embeddings for known speakers
reference_embeddings = {}
# known_speakers = {
#     "speaker1": "voice_samples/keeva.mp3",
#     "speaker2": "voice_samples/sam.mp3"
# }
def extract_embedding(signal, sample_rate):
    # Ensure signal is a tensor on the correct device
    signal = signal.to(device)
    # The model expects inputs of shape (batch_size, num_channels, num_samples)
    signal = signal.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        embedding = embedding_model(signal)  # Shape: (batch_size, embedding_dim)
    embedding = embedding.squeeze(0)  # Remove batch dimension
    return embedding

# Load voice recognition data from the database
known_users = get_users_voice_recognition()

# Load reference embeddings for known speakers
for user in known_users:
    # user.voice_recognition is now a bytes object
    raw_bytes = user.voice_recognition
    # Wrap raw bytes in a BytesIO stream and let torchaudio load it.
    mp3_stream = io.BytesIO(raw_bytes)
    # Let torchaudio auto-detect the format
    signal, sample_rate = torchaudio.load(mp3_stream)  

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        signal = resampler(signal)
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    embedding = extract_embedding(signal, sample_rate)
    reference_embeddings[user.nick_name] = embedding
    logger.info(f"Loaded reference embedding for {user.nick_name}")

def transcribe(audio_file_path):
    # Transcribe the audio file
    segments, info = model.transcribe(audio_file_path, beam_size=5, language="en", vad_filter=True)
    transcription = []
    for segment in segments:
        transcription.append(segment)
    logger.info("Transcription completed")
    return transcription

def diarize(audio_file_path):
    # Process the audio file with the pipeline
    diarization = diarization_pipeline(audio_file_path)
    logger.info("Diarization completed")
    return diarization

def recognize_speakers(diarization_result, audio_file_path):
    identified_speakers = {}
    signal_full, sample_rate = torchaudio.load(audio_file_path)
    for speaker_label in diarization_result.labels():
        # Get all segments for this speaker
        speaker_segments = diarization_result.label_timeline(speaker_label)
        combined_signal = []
        for segment in speaker_segments:
            start_frame = int(segment.start * sample_rate)
            end_frame = int(segment.end * sample_rate)
            signal_segment = signal_full[:, start_frame:end_frame]
            combined_signal.append(signal_segment)
        if not combined_signal:
            identified_speakers[speaker_label] = "Unknown"
            continue
        # Concatenate signals along time dimension
        combined_signal = torch.cat(combined_signal, dim=1)

        # Resample and convert to mono if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            combined_signal = resampler(combined_signal)
            sample_rate = 16000
        if combined_signal.shape[0] > 1:
            combined_signal = torch.mean(combined_signal, dim=0, keepdim=True)
        
        # Ensure signal length is at least the minimum required (7 samples);
        # If too short, pad with zeros to reach length 7.
        min_length = 7
        if combined_signal.shape[1] < min_length:
            pad_length = min_length - combined_signal.shape[1]
            logger.info(f"Speaker {speaker_label} segment too short; padding with {pad_length} zeros")
            combined_signal = torch.nn.functional.pad(combined_signal, (0, pad_length))
        
        # Extract embedding and compare with reference embeddings
        embedding = extract_embedding(combined_signal, sample_rate)
        scores = {}
        for ref_speaker, ref_embedding in reference_embeddings.items():
            score = torch.nn.functional.cosine_similarity(embedding, ref_embedding.to(device), dim=0)
            scores[ref_speaker] = score.item()
        if scores:
            identified_speaker = max(scores, key=scores.get)
        else:
            identified_speaker = "Unknown"
        identified_speakers[speaker_label] = identified_speaker
        logger.info(f"Speaker {speaker_label} identified as {identified_speaker}")
    return identified_speakers

def associate_speakers(transcription, diarization_result, identified_speakers):
    # Create a list to hold the final transcription
    final_transcription = []
    for segment in transcription:
        start_time = segment.start
        end_time = segment.end
        text = segment.text
        # Find overlapping diarization segments
        overlapping_speakers = diarization_result.crop(Segment(start_time, end_time))
        if overlapping_speakers:
            # Get the most frequent speaker in the overlapping segments
            speaker_label = overlapping_speakers.labels()[0]
            identified_speaker = identified_speakers.get(speaker_label, "Unknown")
        else:
            identified_speaker = "Unknown"
        final_transcription.append({identified_speaker: text})
    return final_transcription

async def transcribe_audio(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted")

    # unique_id = str(uuid.uuid4())
    # audio_file_path = f"{unique_id}_received_audio.wav"

    try:
        while True:
            data = await websocket.receive_bytes()
            if not data:
                break
            logger.info("Received audio data")
            
            # Save the received audio data to a file
            audio_file_path = "received_audio.wav"

            # After receiving all data, write to file
            with open(audio_file_path, "wb") as f:
                f.write(data)
            logger.info(f"Audio data saved to {audio_file_path}")

                        # Perform voice activity detection
            vad_result = await asyncio.to_thread(vad_pipeline, audio_file_path)
            logger.info("VAD completed")

            # Check if any speech is detected
            speech_segments = vad_result.get_timeline().support()
            if not speech_segments:
                logger.info("No speech detected in audio")
                await websocket.send_text(json.dumps({}))
                continue  # Skip processing

            # Perform transcription and diarization concurrently
            transcription_task = asyncio.to_thread(transcribe, audio_file_path)
            diarization_task = asyncio.to_thread(diarize, audio_file_path)

            # Wait for both tasks to complete
            transcription, diarization_result = await asyncio.gather(
                transcription_task, diarization_task
            )

            # Perform speaker recognition
            identified_speakers = recognize_speakers(diarization_result, audio_file_path)

            # Associate speakers with transcription
            final_transcription = associate_speakers(transcription, diarization_result, identified_speakers)

            # Send the final transcription back to the client
            # final_transcription_text = "\n".join(final_transcription)
            await websocket.send_text(json.dumps(final_transcription))

    except Exception as e:
        logger.error(f"Error during WebSocket communication: {e}")
        await websocket.send_text({"error": str(e)})
    finally:
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)
            logger.info(f"Removed temporary file {audio_file_path}")
        if not websocket.client_state.closed:
            await websocket.close()
        logger.info("WebSocket connection closed")

app = Starlette()

# Add the WebSocket route
app.add_websocket_route("/ws", transcribe_audio)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
