import os
import wave

DATASET_DIR = "dataset"

CLASSES = [
    "keyword",
    "unknown",
    "noise"
]


def check_audio_file(filepath):
    try:
        with wave.open(filepath, "rb") as audio:

            channels = audio.getnchannels()
            sample_rate = audio.getframerate()
            frames = audio.getnframes()
            sample_width = audio.getsampwidth()

            duration = frames / sample_rate

            return {
                "channels": channels,
                "sample_rate": sample_rate,
                "duration": duration,
                "sample_width": sample_width
            }

    except Exception as e:
        return f"ERROR: {e}"


print("=" * 60)
print("              DATASET CHECK")
print("=" * 60)

total_files = 0

for class_name in CLASSES:

    folder = os.path.join(DATASET_DIR, class_name)

    print(f"\n[{class_name.upper()}]")

    if not os.path.exists(folder):
        print("❌ Folder not found!")
        continue

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(".wav")
    ]

    print(f"Number of files: {len(files)}")

    total_files += len(files)

    if len(files) == 0:
        print("❌ No WAV files found!")
        continue

    errors = 0

    for filename in files:

        filepath = os.path.join(folder, filename)

        result = check_audio_file(filepath)

        if isinstance(result, str):
            print(f"❌ {filename}: {result}")
            errors += 1
            continue

        # Check expected format
        problems = []

        if result["channels"] != 1:
            problems.append(f"channels={result['channels']}")

        if result["sample_rate"] != 16000:
            problems.append(f"sample_rate={result['sample_rate']}")

        if result["duration"] < 1.5 or result["duration"] > 2.5:
            problems.append(f"duration={result['duration']:.2f}s")

        if problems:
            print(f"⚠️ {filename}: {', '.join(problems)}")
            errors += 1

    if errors == 0:
        print("✅ All files have the expected format.")
    else:
        print(f"⚠️ {errors} file(s) need attention.")


print("\n" + "=" * 60)
print(f"TOTAL WAV FILES: {total_files}")
print("=" * 60)

if total_files == 150:
    print("✅ Dataset size looks correct for your prototype.")
else:
    print("⚠️ Expected around 150 files (50 per class).")

print("\nExpected format:")
print("  Sample rate : 16000 Hz")
print("  Channels    : Mono (1)")
print("  Duration    : ~2 seconds")
print("=" * 60)