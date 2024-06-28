import os
import ray
import librosa
import soundfile as sf
from tqdm import tqdm

def list_audio_files(directory, ext='.wav'):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(ext)]

available_cpus = os.cpu_count()
ray.init(num_cpus=available_cpus)  # Ray 초기화

@ray.remote
def resample_audio(input_file_path, output_directory, target_sr=16000):
    # 오디오 파일 읽기
    audio, sr = librosa.load(input_file_path, sr=None)
    
    # 오디오 리샘플링
    if sr != target_sr:
        audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    else:
        audio_resampled = audio
    
    # 리샘플된 오디오 파일 저장
    output_file_path = os.path.join(output_directory, os.path.basename(input_file_path))
    sf.write(output_file_path, audio_resampled, target_sr)

# 오디오 파일이 위치한 입력 폴더
input_directory = '/workspace/jaeyoung/data/WavText5K/WebCrawl/download'
# 리샘플된 오디오 파일을 저장할 출력 폴더
output_directory = '/workspace/jaeyoung/data/WavText5K/WavText5K_16k'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 폴더에서 오디오 파일 리스트 생성
audio_files = list_audio_files(input_directory)

# Ray를 사용하여 각 파일에 대해 리샘플링 작업을 비동기적으로 실행
futures = [resample_audio.remote(input_path, output_directory) for input_path in tqdm(audio_files) if not os.path.basename(input_path).startswith('._')]
ray.get(futures)  # 모든 작업이 완료될 때까지 대기

ray.shutdown()  # Ray 종료
