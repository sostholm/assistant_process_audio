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
LOW_PROB_THRESHOLD = float(os.getenv("LOW_PROB_THRESHOLD", 0.5))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("whisper_server")

# Initialize the device
device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)
logger.info(f"Using device: {device}")

# Initialize the Whisper model
model_size = os.getenv("WHISPER_MODEL", "medium.en")
# Use float16 for RTX 4060
compute_type = "float16"
model = WhisperModel(model_size, device=device_str, compute_type=compute_type)
logger.info(f"Loaded Faster Whisper model: {model_size} on {device} with compute_type: {compute_type}")

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
reference_embeddings = {}
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
    
    # Append to a list of embeddings for this user instead of overwriting
    if user.nick_name not in reference_embeddings:
        reference_embeddings[user.nick_name] = []
    reference_embeddings[user.nick_name].append(embedding)
    logger.info(f"Loaded reference embedding for {user.nick_name} (total: {len(reference_embeddings[user.nick_name])})")

def transcribe(audio_file_path):
    # Transcribe the audio file with word timestamps
    segments, info = model.transcribe(audio_file_path, beam_size=5, language="en", vad_filter=True, word_timestamps=True)
    
    processed_segments = []

    for segment in segments:
        current_words = []
        if segment.words:
            for word in segment.words:
                word_text = word.word.strip() # Get the word text
                # Check if probability is below threshold
                if word.probability < LOW_PROB_THRESHOLD:
                    marked_word = f"{word_text}(?)" # Mark the word
                    current_words.append(marked_word)
                    logger.debug(f"Low probability word: '{word_text}' (Prob: {word.probability:.2f}) -> Marked: '{marked_word}'")
                else:
                    current_words.append(word_text) # Keep original word
            
            processed_text = " ".join(current_words)
        else:
            # Fallback if word timestamps are not available for some reason
            processed_text = segment.text
            logger.warning("Segment found without word timestamps.")

        processed_segments.append({
            "start": segment.start,
            "end": segment.end,
            "text": processed_text
        })
        logger.debug(f"Processed segment text: {processed_text}")

    logger.info("Transcription completed with low probability word marking")
    return processed_segments

def diarize(audio_file_path):
    # Process the audio file with the pipeline
    diarization = diarization_pipeline(audio_file_path)
    logger.info("Diarization completed")
    return diarization

def recognize_speakers(diarization_result, audio_file_path):
    identified_speakers = {}
    try:
        signal_full, sample_rate = torchaudio.load(audio_file_path)
        signal_full = signal_full.to(device) # Move full signal to device

        # Resample full audio once if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000).to(device)
            signal_full = resampler(signal_full)
            sample_rate = 16000 # Update sample rate after resampling

        # Convert to mono once if needed
        if signal_full.shape[0] > 1:
            signal_full = torch.mean(signal_full, dim=0, keepdim=True)

    except Exception as e:
        logger.error(f"Error loading or preprocessing audio file {audio_file_path}: {e}", exc_info=True)
        for speaker_label in diarization_result.labels():
            identified_speakers[speaker_label] = "Unknown"
        return identified_speakers

    for speaker_label in diarization_result.labels():
        speaker_segments = diarization_result.label_timeline(speaker_label)
        combined_signal_list = []
        for segment in speaker_segments:
            start_frame = int(segment.start * sample_rate) # sample_rate is now always 16000
            end_frame = int(segment.end * sample_rate)
            end_frame = min(end_frame, signal_full.shape[1])
            start_frame = min(start_frame, end_frame)

            signal_segment = signal_full[:, start_frame:end_frame]
            combined_signal_list.append(signal_segment)

        if not combined_signal_list:
            logger.warning(f"No valid segments found for speaker {speaker_label}")
            identified_speakers[speaker_label] = "Unknown"
            continue

        combined_signal = torch.cat(combined_signal_list, dim=1)

        min_length = 10
        if combined_signal.shape[1] < min_length:
            if combined_signal.shape[1] < 3:
                logger.info(f"Speaker {speaker_label} combined segment too short: {combined_signal.shape[1]} samples")
                identified_speakers[speaker_label] = "Unknown"
                continue

            pad_length = min_length - combined_signal.shape[1]
            logger.info(f"Speaker {speaker_label} combined segment short; padding with {pad_length} zeros")
            pad_left = pad_length // 2
            pad_right = pad_length - pad_left
            combined_signal = F.pad(combined_signal, (pad_left, pad_right))

            if combined_signal.shape[1] < min_length:
                logger.warning(f"Padding failed for speaker {speaker_label}! Shape is still {combined_signal.shape}")
                identified_speakers[speaker_label] = "Unknown"
                continue

        try:
            embedding = extract_embedding(combined_signal, sample_rate) # sample_rate is 16000
            scores = {}
            max_overall_score = -float('inf')
            best_match_speaker = "Unknown"

            for ref_speaker, ref_embeddings_list in reference_embeddings.items():
                max_user_score = -float('inf')
                for ref_embedding in ref_embeddings_list:
                    score = F.cosine_similarity(embedding, ref_embedding.to(device), dim=0)
                    max_user_score = max(max_user_score, score.item())

                scores[ref_speaker] = max_user_score
                if max_user_score > max_overall_score:
                    max_overall_score = max_user_score
                    best_match_speaker = ref_speaker

            confidence_threshold = 0.27
            if max_overall_score >= confidence_threshold:
                identified_speaker = best_match_speaker
                final_score_log = max_overall_score
            else:
                identified_speaker = "Unknown"
                final_score_log = max_overall_score

            identified_speakers[speaker_label] = identified_speaker
            logger.info(f"Speaker {speaker_label} identified as {identified_speaker} with score {final_score_log:.4f} (Threshold: {confidence_threshold})")

        except Exception as e:
            logger.error(f"Error during speaker recognition comparison for {speaker_label}: {e}", exc_info=True)
            identified_speakers[speaker_label] = "Unknown"

    return identified_speakers

def associate_speakers(transcription, diarization_result, identified_speakers):
    # Create a list to hold the final transcription
    final_transcription = []
    # 'transcription' is now a list of dictionaries: [{"start": s, "end": e, "text": t}, ...]
    for segment_info in transcription:
        start_time = segment_info["start"]
        end_time = segment_info["end"]
        text = segment_info["text"] # Text may contain marked words like "word(?)"
        
        # Find overlapping diarization segments
        overlapping_speakers = diarization_result.crop(Segment(start_time, end_time))
        
        if overlapping_speakers:
            # Get the most frequent speaker label in the overlapping segments
            # Ensure labels() returns a list and access the first element safely
            labels = overlapping_speakers.labels()
            if labels:
                speaker_label = labels[0]
                identified_speaker = identified_speakers.get(speaker_label, "Unknown")
            else:
                # Handle cases where crop might return an empty annotation for the segment
                identified_speaker = "Unknown"
                logger.warning(f"No speaker label found for segment: {start_time}-{end_time}")
        else:
            identified_speaker = "Unknown"
            logger.warning(f"No overlapping diarization segment found for transcription segment: {start_time}-{end_time}")
            
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
