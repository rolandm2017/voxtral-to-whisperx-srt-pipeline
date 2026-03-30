import whisperx
from whisperx.schema import SingleSegment
from whisperx.audio import load_audio
from whisperx.utils import get_writer

from dataclasses import dataclass
import ast


input_lines_from_voxtral = {}
with open("./src/alignment/myfile.txt", "r") as f:
    voxtral_out: list[str] = f.readlines()
    
    input_lines_from_voxtral = ast.literal_eval(voxtral_out[2])

@dataclass
class VoxtralSegment:
    text: str
    start: float
    end: float

segments_for_wx = []

for key, value in input_lines_from_voxtral.items():
    print(f"{key}: {type(value).__name__} — {str(value)[:100]}")
    if key == "segments":
        print(value)

        segments = [VoxtralSegment(**s) for s in input_lines_from_voxtral['segments']]

        for v, segment in enumerate(segments):
            print(v, segments[v].start, segments[v].text)
            segments_for_wx.append(SingleSegment(start=segments[v].start, end=segments[v].end, text=segments[v].text))
    
device = "cpu"
audio = load_audio("./src/alignment/france1.mp3")

model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
transcription = whisperx.align(segments_for_wx, model_a, metadata, audio, device)

result = {"segments": segments_for_wx, "language": "en"}


srt_writer = get_writer("srt", "./")  # output directory
srt_writer(
    result,
    "france1.mp3",  # just used for naming the output file
    {"max_line_width": None, "max_line_count": None, "highlight_words": False},
)
