import io
import sys
import whisper

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def transcribe(audio_path, model_name="base"):
    print(f"Loading Whisper '{model_name}' model...")
    model = whisper.load_model(model_name)

    print(f"Transcribing: {audio_path}")
    result = model.transcribe(audio_path)

    return result["text"]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <audio_file> [model]")
        print("Models: tiny, base (default), small, medium, large")
        print("  tiny   - fastest, least accurate")
        print("  base   - good balance (default)")
        print("  small  - better accuracy")
        print("  medium - high accuracy")
        print("  large  - best accuracy, slowest")
        sys.exit(1)

    audio_file = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "base"

    text = transcribe(audio_file, model_name)

    print("\n--- Transcription ---\n")
    print(text)

    output_file = audio_file.rsplit(".", 1)[0] + ".txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nSaved to: {output_file}")
