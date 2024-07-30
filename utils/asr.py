from faster_whisper import WhisperModel

model_size = "D:\\PythonProject\\apollo\\weight\\whisper-large-v2-ct2"
model = WhisperModel(model_size, device="cpu", compute_type="int8")

def stt(audio):
        ls_result = []
        segments, _ = model.transcribe(audio, beam_size=1, language='zh',
                                          vad_filter=True,
                                          initial_prompt='以下是普通話句子，國泰產險客服中心敝姓')
           
        for segment in segments:
                text = "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text)
                ls_result.append(text)

        return ls_result

     
if __name__ == "__main__":
    audio_path = 'D:\\PythonProject\\apollo\\data\\onepiece_15s.mp3'
    output_path = 'D:\\PythonProject\\apollo\\result\\onepiece_15s_2.txt'
    ls_result = stt(audio_path)
    with open(output_path, "w", encoding="utf-8") as file:
           for item in ls_result:
                 file.write(item + "\n")
