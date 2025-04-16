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
        
        # Check if the segment is too short for embedding
        # The model's kernel size is 7, so we need at least that many samples
        # Adding a small safety margin
        min_length = 10  # Just slightly more than the kernel size of 7
        if combined_signal.shape[1] < min_length:
            # If the segment is extremely short, skip it or mark as unknown
            if combined_signal.shape[1] < 3:  # Too short to be meaningful
                logger.info(f"Speaker {speaker_label} segment too short to process: {combined_signal.shape[1]} samples")
                identified_speakers[speaker_label] = "Unknown"
                continue
                
            # Pad with zeros to reach minimum length
            pad_length = min_length - combined_signal.shape[1]
            logger.info(f"Speaker {speaker_label} segment short; padding with {pad_length} zeros")
            # Pad on both sides for better results
            pad_left = pad_length // 2
            pad_right = pad_length - pad_left
            combined_signal = F.pad(combined_signal, (pad_left, pad_right))
            
            # Double-check the padding was successful
            if combined_signal.shape[1] < min_length:
                logger.warning(f"Padding failed! Shape is still {combined_signal.shape}")
                identified_speakers[speaker_label] = "Unknown"
                continue
        
        try:
            # Extract embedding and compare with reference embeddings
            embedding = extract_embedding(combined_signal, sample_rate)
            scores = {}
            for ref_speaker, ref_embedding in reference_embeddings.items():
                score = F.cosine_similarity(embedding, ref_embedding.to(device), dim=0)
                scores[ref_speaker] = score.item()
            if scores:
                identified_speaker = max(scores, key=scores.get)
                max_score = scores[identified_speaker]
                # Add a confidence threshold
                if max_score < 0.2:  # Adjust this threshold as needed
                    logger.info(f"Speaker {speaker_label} matched {identified_speaker} but score {max_score} below threshold")
                    identified_speaker = "Unknown"
            else:
                identified_speaker = "Unknown"
            identified_speakers[speaker_label] = identified_speaker
            logger.info(f"Speaker {speaker_label} identified as {identified_speaker}")
        except Exception as e:
            logger.error(f"Error in speaker recognition for {speaker_label}: {e}")
            identified_speakers[speaker_label] = "Unknown"
    
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

    audio_file_path = None
    try:
        while True:
            data = await websocket.receive_bytes()
            if not data:
                break
            logger.info("Received audio data")
            
            # Create a temporary WAV file with a unique name
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                temp_audio_file.write(data)
                audio_file_path = temp_audio_file.name
            
            logger.info(f"Audio data saved to temporary file {audio_file_path}")

            # Perform voice activity detection
            vad_result = await asyncio.to_thread(vad_pipeline, audio_file_path)
            logger.info("VAD completed")

            # Check if any speech is detected
            speech_segments = vad_result.get_timeline().support()
            if not speech_segments:
                logger.info("No speech detected in audio")
                await websocket.send_text(json.dumps({}))
                # Clean up the temporary file before continuing
                if os.path.exists(audio_file_path):
                    os.remove(audio_file_path)
                    audio_file_path = None
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
        if audio_file_path and os.path.exists(audio_file_path):
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
