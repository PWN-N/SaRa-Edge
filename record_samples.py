import sounddevice as sd
import soundfile as sf
import os

SAMPLE_RATE = 16000
DURATION = 2
CHANNELS = 1
DEVICE = 1

OUTPUT_DIR = "dataset/noise"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("===================================")
print("      KEYWORD DATASET RECORDER")
print("===================================")
print(f"Sample rate : {SAMPLE_RATE} Hz")
print(f"Duration    : {DURATION} seconds")
print(f"Microphone  : Device {DEVICE}")
print()
print("Press Ctrl+C to stop.")
print()

count = 1

try:
    while True:

        filename = input(
            f"Enter filename for recording {count} "
            "(or press Enter for automatic name): "
        ).strip()

        if not filename:
            filename = f"sample_{count:03d}"

        filepath = os.path.join(
            OUTPUT_DIR,
            filename + ".wav"
        )

        print()
        print("Recording starts in 2 seconds...")
        sd.sleep(2000)

        print(">>> SPEAK YOUR KEYWORD NOW <<<")

        audio = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            device=DEVICE
        )

        sd.wait()

        sf.write(
            filepath,
            audio,
            SAMPLE_RATE
        )

        print(f"Saved: {filepath}")
        print()

        count += 1

except KeyboardInterrupt:
    print("\n")
    print("Recording stopped.")
    print(f"Total recordings created: {count - 1}")