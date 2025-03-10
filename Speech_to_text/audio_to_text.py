import speech_recognition as sr
import time
import wave

def save_audio(audio_data, filename="recorded_audio.wav"):
    """Save recorded audio to a file for debugging or training purposes."""
    with wave.open(filename, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(audio_data.get_wav_data())

def record_and_convert():
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        print("Adjusting for background noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Recording... Speak now.")

        try:
            # audio_data = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            audio_data = recognizer.listen(source,timeout=5)

            print("Recording complete. Converting to text...")

            # Save audio for debugging
            save_audio(audio_data)

            # Recognize speech using Google API (supports multiple languages)
            text = recognizer.recognize_google(audio_data, language="en-US")
            print("Transcribed Text:", text)

            return text

        except sr.UnknownValueError:
            print("⚠ Could not understand the audio. Try again.")
            return None
        except sr.RequestError:
            print("⚠ Could not request results from Google Speech Recognition.")
            return None
        except sr.WaitTimeoutError:
            print("⚠ No speech detected within the time limit.")
            return None

# Run the function
record_and_convert()
