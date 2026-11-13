"""Audio enrollment prompting for target-speaker ASR with Whisper."""

SAMPLE_RATE = 16000
# One Whisper encoder output frame covers 2 mel hops of 160 samples = 20 ms.
ENC_HOP = 320
ENC_FRAME_RATE = SAMPLE_RATE // ENC_HOP  # 50 Hz
WHISPER_WINDOW_SEC = 30.0
WHISPER_WINDOW_SAMPLES = int(WHISPER_WINDOW_SEC * SAMPLE_RATE)
WHISPER_ENC_FRAMES = WHISPER_WINDOW_SAMPLES // ENC_HOP  # 1500

# DiCoW v3.2 remote code and weights are pinned to this Hugging Face commit
DICOW_MODEL = "BUT-FIT/DiCoW_v3_2"
DICOW_REVISION = "00a09c622bf58c34fdc76c3b5676f4da706f2a8e"
