import argparse
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf


SAMPLE_RATE = 16000
OUTPUT_FILE = Path("data/test_audio.wav")


def list_devices():
    print("\nAvailable audio devices:\n")
    print(sd.query_devices())


def record_audio(seconds):
    print(f"\nRecording for {seconds} seconds...")
    print("Speak something into your microphone.")

    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    sf.write(
        OUTPUT_FILE,
        audio,
        SAMPLE_RATE,
        subtype="PCM_16"
    )

    print("\nRecording finished.")
    print(f"Saved to: {OUTPUT_FILE}")

    inspect_audio(OUTPUT_FILE)


def inspect_audio(file_path):
    audio, sample_rate = sf.read(
        file_path,
        always_2d=False
    )

    if audio.ndim == 1:
        channels = 1
    else:
        channels = audio.shape[1]

    duration = len(audio) / sample_rate

    rms = np.sqrt(
        np.mean(
            np.square(
                audio.astype(np.float64)
            )
        )
    )

    print("\n========== AUDIO INFORMATION ==========")
    print(f"File        : {file_path}")
    print(f"Sample rate : {sample_rate} Hz")
    print(f"Channels    : {channels}")
    print(f"Samples     : {len(audio)}")
    print(f"Duration    : {duration:.3f} seconds")
    print(f"RMS         : {rms:.6f}")
    print("========================================")

    if sample_rate != 16000:
        print("WARNING: Sample rate is not 16 kHz.")

    if channels != 1:
        print("WARNING: Audio is not mono.")


def main():

    parser = argparse.ArgumentParser(
        description="SaRa-Edge microphone testing"
    )

    parser.add_argument(
        "--devices",
        action="store_true",
        help="Show available audio devices"
    )

    parser.add_argument(
        "--record",
        type=float,
        help="Record audio for given seconds"
    )

    parser.add_argument(
        "--file",
        type=Path,
        help="Inspect an existing WAV file"
    )

    args = parser.parse_args()

    if args.devices:
        list_devices()

    elif args.record:
        record_audio(args.record)

    elif args.file:
        inspect_audio(args.file)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()