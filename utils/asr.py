from faster_whisper import WhisperModel
from pydub import AudioSegment
import io
import wave


model_size = "D:\\PythonProject\\apollo\\weight\\whisper-large-v2-ct2"
model = WhisperModel(model_size, device="cpu", compute_type="int8")

def stt(audio):
        ls_result = []
        segments, _ = model.transcribe(audio, beam_size=1, language='zh',
                                          vad_filter=True,
                                          initial_prompt='以下是普通話句子，國泰產險客服中心敝姓')
           
        for segment in segments:
                text = "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text)
                print(text)
                ls_result.append(text)
 
        return ls_result

def convert_to_wav(audio_file, file_format):
    audio = AudioSegment.from_file(audio_file, format=file_format)
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)
    return wav_io

def merge_audio_files(audio_files):
    combined_audio = io.BytesIO()
    output_wav = wave.open(combined_audio, 'wb')

    for i, audio in enumerate(audio_files):
        audio.seek(0)
        with wave.open(audio, 'rb') as wf:
            if i == 0:
                output_wav.setnchannels(wf.getnchannels())
                output_wav.setsampwidth(wf.getsampwidth())
                output_wav.setframerate(wf.getframerate())
            while True:
                frames = wf.readframes(1024)
                if not frames:
                    break
                output_wav.writeframes(frames)
    
    output_wav.close()
    combined_audio.seek(0)
    return combined_audio


# def merge_audio_files(audio_files):
#     combined_audio = None
    
#     for audio in audio_files:
#         audio.seek(0)
#         segment = AudioSegment.from_file(audio, format="wav")
#         if combined_audio is None:
#             combined_audio = segment
#         else:
#             combined_audio += segment

#     combined_audio_io = io.BytesIO()
#     combined_audio.export(combined_audio_io, format="wav")
#     combined_audio_io.seek(0)
#     return combined_audio_io


if __name__ == "__main__":
    audio_path = 'D:\\PythonProject\\apollo\\data\\onepiece_15s.mp3'
    output_path = 'D:\\PythonProject\\apollo\\result\\onepiece_15s_2.txt'
    ls_result = stt(audio_path)
    with open(output_path, "w", encoding="utf-8") as file:
           for item in ls_result:
                 file.write(item + "\n")
