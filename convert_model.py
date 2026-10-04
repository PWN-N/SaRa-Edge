from pathlib import Path

MODEL_FILE = "models/kws_dscnn_int8_v2.tflite"
OUTPUT_FILE = "kws_model.h"

model_data = Path(MODEL_FILE).read_bytes()

with open(OUTPUT_FILE, "w") as f:
    f.write("#ifndef KWS_MODEL_H\n")
    f.write("#define KWS_MODEL_H\n\n")
    f.write("#include <stdint.h>\n\n")

    f.write("const unsigned char kws_model[] = {\n")

    for i in range(0, len(model_data), 12):
        chunk = model_data[i:i+12]
        f.write("    ")
        f.write(", ".join(f"0x{b:02x}" for b in chunk))
        f.write(",\n")

    f.write("};\n\n")
    f.write(f"const unsigned int kws_model_len = {len(model_data)};\n\n")
    f.write("#endif\n")

print("Model converted successfully!")
print(f"Model size: {len(model_data) / 1024:.2f} KB")
print(f"Output: {OUTPUT_FILE}")