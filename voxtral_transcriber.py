from transformers import VoxtralForConditionalGeneration, AutoProcessor
import torch

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
repo_id = "mistralai/Voxtral-Mini-3B-2507"
audio_path = "path/to/mp3.mp3"

print(f"Loading model on {device}...")

# Load processor and model
processor = AutoProcessor.from_pretrained(repo_id)
model = VoxtralForConditionalGeneration.from_pretrained(
    repo_id,
    torch_dtype=torch.bfloat16,
    device_map=device,
    timestamp_granularities=["segment"]
)

print("Model loaded. Processing audio...")

# Create transcription request
inputs = processor.apply_transcription_request(
    language="fr",  # Change to "fr" if the audio is in French
    audio=audio_path,
    model_id=repo_id,
#    timestamp_granularities=["segment"]  # or ["word"] for word-level
)

# Move to device
inputs = inputs.to(device, dtype=torch.bfloat16)

print("Generating transcription...")

# Generate transcription
outputs = model.generate(**inputs, max_new_tokens=500)

# Decode the output
decoded_outputs = processor.batch_decode(
    outputs[:, inputs.input_ids.shape[1]:],
    skip_special_tokens=True
)


# Inspect for segments with timestamps
if hasattr(decoded_outputs[0], "segments"):
    for segment in decoded_outputs[0].segments:
        print(f"Text: {segment['text']}")
        print(f"Start: {segment['start']}s")
        print(f"End: {segment['end']}s")
else:
    print(decoded_outputs)

# Get the transcription text
transcription = decoded_outputs[0]

print("\nTranscription:")
print("=" * 80)
print(transcription)
print("=" * 80)

# Save to file
with open("output.txt", "w", encoding="utf-8") as f:
    f.write(transcription)

print(f"\nTranscription saved to output.txt")
