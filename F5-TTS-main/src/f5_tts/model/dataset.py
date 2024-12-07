import json #Json 파일 처리를 위한 라이브러리
import random # 난수 생성, 무작위 선택 등을 위한 라이브러리 
from importlib.resources import files #패키지 리소스 접근을 위한도구

import torch #Pytorch 메인라이브러리
import torch.nn.functional as F #PyTorch의 함수형 신경망 연상(패딩 등)
import torchaudio #오디오 처리를 위한 PyTorch 라이브러리
from datasets import Dataset as Dataset_ #Hugging Face의 Dataset 클래스를 확장한 사용자 정의 데이터셋 클래스
from datasets import load_from_disk #Hugging Face의 데이터셋을 로드하는 함수
from torch import nn #PyTorch의 신경망 모듈
from torch.utils.data import Dataset, Sampler #PyTorch의 데이터셋 및 샘플러 모듈
from tqdm import tqdm #진행 상황을 표시하는 프로그램

from f5_tts.model.modules import MelSpec #Mel 스펙트로그램 생성을 위한 모듈
from f5_tts.model.utils import default #기본값 설정을 위한 모듈


class HFDataset(Dataset): #Dataset 클래스를 상속받아 정의되는 클래스 huggign 데이터셋을 pytorch 데이터셋으로 변환
    def __init__( #init 메서드는 클래스 생성자, 객체가 생성될 때 호출되고 데어티셋을 초기화하는 데 필요한 여러 매개변수를 받는다
        self,
        hf_dataset: Dataset, # Hugging Face 데이터셋
        target_sample_rate=24_000, # 목표 샘플링 속도
        n_mel_channels=100, #Mel 스펙트로그램 채널 수
        hop_length=256, # STFT 윈도우 간 이동 크기
        n_fft=1024, # Fast Fourier Transform 크기
        win_length=1024, # STFT 윈도우 길이
        mel_spec_type="vocos", # Mel 스펙트로그램 유형
    ):
        self.data = hf_dataset #클래스 인스턴스의 속성으로, 전달받은 hf_dataset을 저장 , 나중에 데이터 접근시 사용
        #목표 샘플링 속도, STFT 위도우 간 이동 크기 저장, 나중에 오디오 데이터 처리시 사용
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        
        #MelSpec 객체를 생성하여 저장, 이 객체는 mel 스펙토그램 생성에 사용, 생성자에 전달된 매개변수들은 mel 스펙트로그램 생성 시 필요한 설정값들
        self.mel_spectrogram = MelSpec(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        )
    # 메서드는 주어진 인덱스의 오디오 데이터의 프레임 길이를 계산한다.
    def get_frame_len(self, index):
        row = self.data[index] #데이터셋에서 주어진 인덱스의 데이터를 가져온다
        audio = row["audio"]["array"] #오디오 데이터를 가져온다.
        sample_rate = row["audio"]["sampling_rate"] #오디오의 샘플링 속도를 가져온다.
        return audio.shape[-1] / sample_rate * self.target_sample_rate / self.hop_length #오디오 데이터의 길이를 목표 샘플링 속도와 hop_length를 사용하여 프레임 길이를 계산하여 반환한다.

    def __len__(self): #데이터셋의 길이를 반환하는 메서드
        return len(self.data)

    def __getitem__(self, index): #주어진 인덱스의 데이터를 반환하는 메서드
        row = self.data[index] #데이터셋에서 주어진 인덱스의 데이터를 가져온다.
        audio = row["audio"]["array"] #오디오 데이터를 가져온다.

        # logger.info(f"Audio shape: {audio.shape}") #오디오 데이터의 크기를 출력

        sample_rate = row["audio"]["sampling_rate"]#오디오의 샘플링 속도를 가져온다.
        duration = audio.shape[-1] / sample_rate #오디오 데이터의 길이를 계산한다.

        if duration > 30 or duration < 0.3:#오디오 데이터의 길이가 30초 이상 또는 0.3초 미만이면 다음 데이터로 넘어간다.
            return self.__getitem__((index + 1) % len(self.data)) #다음 데이터로 넘어간다.

        audio_tensor = torch.from_numpy(audio).float() #오디오 데이터를 텐서로 변환

        if sample_rate != self.target_sample_rate:   #샘플링 속도가 목표 샘플링 속도와 다르면 재샘플링
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)   #재샘플링 객체 생성
            audio_tensor = resampler(audio_tensor) #재샘플링

        audio_tensor = audio_tensor.unsqueeze(0)  # 't -> 1 t') #오디오 텐서를 채널 차원을 추가하여 텐서 크기를 변경

        mel_spec = self.mel_spectrogram(audio_tensor) #MelSpec 객체를 사용하여 오디오 텐서를 mel 스펙트로그램으로 변환

        mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'

        text = row["text"] #텍스트 데이터를 가져온다.

        return dict(#mel 스펙트로그램과 텍스트 데이터를 반환
            mel_spec=mel_spec,
            text=text,
        )

# 사용자 정의 데이터셋 클래스, 오디오 데이터를 전처리하고 mel 스펙트로그램을 생성하는 데이터셋
class CustomDataset(Dataset): #Dataset 클래스를 상속받아 정의되는 클래스
    def __init__(#init 메서드는 클래스 생성자, 객체가 생성될 때 호출되고 데이터셋을 초기화하는 데 필요한 여러 매개변수를 받는다
        self,
        custom_dataset: Dataset, #사용자 정의 데이터셋
        durations=None, #오디오 데이터의 길이
        target_sample_rate=24_000, #목표 샘플링 속도
        hop_length=256, #STFT 윈도우 간 이동 크기
        n_mel_channels=100, #Mel 스펙트로그램 채널 수
        n_fft=1024, #Fast Fourier Transform 크기            
        win_length=1024, #STFT 윈도우 길이
        mel_spec_type="vocos", #Mel 스펙트로그램 유형
        preprocessed_mel=False, #사전 처리된 mel 스펙트로그램 여부
        mel_spec_module: nn.Module | None = None, #Mel 스펙트로그램 모듈
    ):
        self.data = custom_dataset #클래스 인스턴스의 속성으로, 전달받은 custom_dataset을 저장, 나중에 데이터 접근시 사용
        self.durations = durations #오디오 데이터의 길이를 저장, 나중에 데이터 접근시 사용
        self.target_sample_rate = target_sample_rate #목표 샘플링 속도를 저장, 나중에 데이터 접근시 사용
        self.hop_length = hop_length #STFT 윈도우 간 이동 크기를 저장, 나중에 데이터 접근시 사용
        self.n_fft = n_fft #Fast Fourier Transform 크기를 저장, 나중에 데이터 접근시 사용
        self.win_length = win_length #STFT 윈도우 길이를 저장, 나중에 데이터 접근시 사용
        self.mel_spec_type = mel_spec_type #Mel 스펙트로그램 유형을 저장, 나중에 데이터 접근시 사용
        self.preprocessed_mel = preprocessed_mel #사전 처리된 mel 스펙트로그램 여부를 저장, 나중에 데이터 접근시 사용

        if not preprocessed_mel: #사전 처리된 mel 스펙트로그램이 아니면
            self.mel_spectrogram = default( #MelSpec 객체를 생성하여 저장, 이 객체는 mel 스펙트로그램 생성에 사용
                mel_spec_module, #Mel 스펙트로그램 모듈
                MelSpec( #MelSpec 객체를 생성하여 저장, 이 객체는 mel 스펙트로그램 생성에 사용
                    n_fft=n_fft, #Fast Fourier Transform 크기
                    hop_length=hop_length, #STFT 윈도우 간 이동 크기
                    win_length=win_length, #STFT 윈도우 길이
                    n_mel_channels=n_mel_channels, #Mel 스펙트로그램 채널 수
                    target_sample_rate=target_sample_rate, #목표 샘플링 속도
                    mel_spec_type=mel_spec_type, #Mel 스펙트로그램 유형
                ),
            )

    def get_frame_len(self, index): #주어진 인덱스의 오디오 데이터의 프레임 길이를 계산하는 메서드
        if ( #오디오 데이터의 길이가 제공되면
            self.durations is not None #오디오 데이터의 길이가 제공되면
        ):  # Please make sure the separately provided durations are correct, otherwise 99.99% OOM
            return self.durations[index] * self.target_sample_rate / self.hop_length #오디오 데이터의 길이를 목표 샘플링 속도와 hop_length를 사용하여 프레임 길이를 계산하여 반환
        return self.data[index]["duration"] * self.target_sample_rate / self.hop_length #오디오 데이터의 길이를 목표 샘플링 속도와 hop_length를 사용하여 프레임 길이를 계산하여 반환

    def __len__(self): #데이터셋의 길이를 반환하는 메서드
        return len(self.data) #데이터셋의 길이를 반환

    def __getitem__(self, index): #주어진 인덱스의 데이터를 반환하는 메서드
        row = self.data[index] #데이터셋에서 주어진 인덱스의 데이터를 가져온다.
        audio_path = row["audio_path"] #오디오 파일 경로를 가져온다.
        text = row["text"] #텍스트 데이터를 가져온다.
        duration = row["duration"] #오디오 데이터의 길이를 가져온다.

        if self.preprocessed_mel: #사전 처리된 mel 스펙트로그램이면
            mel_spec = torch.tensor(row["mel_spec"]) #mel 스펙트로그램을 텐서로 변환하여 저장

        else: #사전 처리된 mel 스펙트로그램이 아니면
            audio, source_sample_rate = torchaudio.load(audio_path) #오디오 파일을 로드하여 오디오 데이터와 원본 샘플링 속도를 가져온다.
            if audio.shape[0] > 1: #오디오 데이터가 여러 채널을 가지면
                audio = torch.mean(audio, dim=0, keepdim=True) #채널 차원을 평균하여 하나의 채널로 변환

            if duration > 30 or duration < 0.3: #오디오 데이터의 길이가 30초 이상 또는 0.3초 미만이면
                return self.__getitem__((index + 1) % len(self.data)) #다음 데이터로 넘어간다.

            if source_sample_rate != self.target_sample_rate: #원본 샘플링 속도가 목표 샘플링 속도와 다르면
                resampler = torchaudio.transforms.Resample(source_sample_rate, self.target_sample_rate) #재샘플링 객체 생성
                audio = resampler(audio) #재샘플링

            mel_spec = self.mel_spectrogram(audio) #MelSpec 객체를 사용하여 오디오 텐서를 mel 스펙트로그램으로 변환
            mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t')

        return dict( #mel 스펙트로그램과 텍스트 데이터를 반환
            mel_spec=mel_spec,
            text=text,
        )


# Dynamic Batch Sampler


class DynamicBatchSampler(Sampler[list[int]]): #Sampler 클래스를 상속받아 정의되는 클래스, 동적 배치 샘플러
    """Extension of Sampler that will do the following:
    1.  Change the batch size (essentially number of sequences)
        in a batch to ensure that the total number of frames are less
        than a certain threshold.
    2.  Make sure the padding efficiency in the batch is high.
    """

    def __init__( #init 메서드는 클래스 생성자, 객체가 생성될 때 호출되고 데이터셋을 초기화하는 데 필요한 여러 매개변수를 받는다
        self, sampler: Sampler[int], frames_threshold: int, max_samples=0, random_seed=None, drop_last: bool = False # 샘플러, 프레임 임계값, 최대 샘플 수, 랜덤 시드, 마지막 배치 삭제 여부
    ):
        self.sampler = sampler #샘플러를 저장
        self.frames_threshold = frames_threshold #프레임 임계값을 저장
        self.max_samples = max_samples #최대 샘플 수를 저장

        indices, batches = [], [] #인덱스와 배치를 저장할 리스트 초기화
        data_source = self.sampler.data_source #샘플러의 데이터 소스를 가져온다.

        for idx in tqdm( #진행 상황을 표시하는 프로그램
            self.sampler, desc="Sorting with sampler... if slow, check whether dataset is provided with duration"
        ):
            indices.append((idx, data_source.get_frame_len(idx))) #인덱스와 프레임 길이를 저장
        indices.sort(key=lambda elem: elem[1]) #프레임 길이를 기준으로 정렬

        batch = [] #배치를 저장할 리스트 초기화
        batch_frames = 0 #배치의 프레임 길이를 저장할 변수 초기화
        for idx, frame_len in tqdm( #진행 상황을 표시하는 프로그램
            indices, desc=f"Creating dynamic batches with {frames_threshold} audio frames per gpu" #설명 문자열
        ):
            if batch_frames + frame_len <= self.frames_threshold and (max_samples == 0 or len(batch) < max_samples): #배치의 프레임 길이가 임계값 이하이고 최대 샘플 수가 0 또는 배치의 길이가 최대 샘플 수 미만이면
                batch.append(idx) #배치에 인덱스를 추가
                batch_frames += frame_len #배치의 프레임 길이를 업데이트
            else: #배치의 프레임 길이가 임계값 초과하거나 최대 샘플 수를 초과하면
                if len(batch) > 0: #배치가 비어있지 않으면
                    batches.append(batch) #배치를 추가
                if frame_len <= self.frames_threshold: #프레임 길이가 임계값 이하이면
                    batch = [idx] #배치를 초기화
                    batch_frames = frame_len #배치의 프레임 길이를 업데이트
                else: #프레임 길이가 임계값 초과하면
                    batch = [] #배치를 초기화
                    batch_frames = 0 #배치의 프레임 길이를 초기화

        if not drop_last and len(batch) > 0: #마지막 배치를 삭제하지 않고 배치가 비어있지 않으면
            batches.append(batch) #배치를 추가

        del indices #인덱스 삭제    

        # if want to have different batches between epochs, may just set a seed and log it in ckpt
        # cuz during multi-gpu training, although the batch on per gpu not change between epochs, the formed general minibatch is different
        # e.g. for epoch n, use (random_seed + n)
        random.seed(random_seed) #랜덤 시드 설정
        random.shuffle(batches) #배치를 무작위로 섞는다.    

        self.batches = batches #배치를 저장

    def __iter__(self): #배치를 반복하는 메서드
        return iter(self.batches) #배치를 반환

    def __len__(self): #배치의 길이를 반환하는 메서드
        return len(self.batches) #배치의 길이를 반환


# Load dataset


def load_dataset( #데이터셋을 로드하는 함수 
    dataset_name: str, #데이터셋 이름
    tokenizer: str = "pinyin", #토크나이저 이름
    dataset_type: str = "CustomDataset", #데이터셋 유형
    audio_type: str = "raw", #오디오 유형
    mel_spec_module: nn.Module | None = None, #Mel 스펙트로그램 모듈
    mel_spec_kwargs: dict = dict(), #Mel 스펙트로그램 매개변수
) -> CustomDataset | HFDataset: #CustomDataset 또는 HFDataset 클래스를 반환
    """
    dataset_type    - "CustomDataset" if you want to use tokenizer name and default data path to load for train_dataset
                    - "CustomDatasetPath" if you just want to pass the full path to a preprocessed dataset without relying on tokenizer
    """

    print("Loading dataset ...") #데이터셋 로드 중 메시지 출력

    if dataset_type == "CustomDataset": #CustomDataset 유형인 경우
        rel_data_path = str(files("f5_tts").joinpath(f"../../data/{dataset_name}_{tokenizer}")) #데이터셋 경로 설정
        if audio_type == "raw": #오디오 유형이 raw인 경우
            try: #파일이 존재하면
                train_dataset = load_from_disk(f"{rel_data_path}/raw") #데이터셋 로드
            except:  # noqa: E722
                train_dataset = Dataset_.from_file(f"{rel_data_path}/raw.arrow") #데이터셋 로드
            preprocessed_mel = False #사전 처리된 mel 스펙트로그램 여부 설정
        elif audio_type == "mel": #오디오 유형이 mel인 경우
            train_dataset = Dataset_.from_file(f"{rel_data_path}/mel.arrow") #데이터셋 로드
            preprocessed_mel = True #사전 처리된 mel 스펙트로그램 여부 설정
        with open(f"{rel_data_path}/duration.json", "r", encoding="utf-8") as f: #duration.json 파일 열기
            data_dict = json.load(f) #JSON 파일 로드
        durations = data_dict["duration"] #duration 키의 값을 가져옴
        train_dataset = CustomDataset( #CustomDataset 객체 생성
            train_dataset, #데이터셋
            durations=durations, #오디오 데이터의 길이
            preprocessed_mel=preprocessed_mel, #사전 처리된 mel 스펙트로그램 여부
            mel_spec_module=mel_spec_module, #Mel 스펙트로그램 모듈
            **mel_spec_kwargs, #Mel 스펙트로그램 매개변수
        )

    elif dataset_type == "CustomDatasetPath": #CustomDatasetPath 유형인 경우    
        try: #파일이 존재하면
            train_dataset = load_from_disk(f"{dataset_name}/raw") #데이터셋 로드
        except:  # noqa: E722
            train_dataset = Dataset_.from_file(f"{dataset_name}/raw.arrow") #데이터셋 로드

        with open(f"{dataset_name}/duration.json", "r", encoding="utf-8") as f: #duration.json 파일 열기    
            data_dict = json.load(f) #JSON 파일 로드
        durations = data_dict["duration"] #duration 키의 값을 가져옴    
        train_dataset = CustomDataset( #CustomDataset 객체 생성
            train_dataset, durations=durations, preprocessed_mel=preprocessed_mel, **mel_spec_kwargs #오디오 데이터의 길이, 사전 처리된 mel 스펙트로그램 여부, Mel 스펙트로그램 매개변수

        )

    elif dataset_type == "HFDataset": #HFDataset 유형인 경우
        print(
            "Should manually modify the path of huggingface dataset to your need.\n"
            + "May also the corresponding script cuz different dataset may have different format."
        )
        pre, post = dataset_name.split("_") #데이터셋 이름을 전처리 및 후처리로 분리
        train_dataset = HFDataset( #HFDataset 객체 생성
            load_dataset(f"{pre}/{pre}", split=f"train.{post}", cache_dir=str(files("f5_tts").joinpath("../../data"))), #데이터셋 로드
        )

    return train_dataset #데이터셋을 반환


# collation


def collate_fn(batch): #배치를 정리하는 함수
    mel_specs = [item["mel_spec"].squeeze(0) for item in batch] #mel 스펙트로그램을 정리
    mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs]) #mel 스펙트로그램의 길이를 정리
    max_mel_length = mel_lengths.amax() #mel 스펙트로그램의 최대 길이를 정리

    padded_mel_specs = [] #패딩된 mel 스펙트로그램을 저장할 리스트 초기화
    for spec in mel_specs:  # TODO. maybe records mask for attention here
        padding = (0, max_mel_length - spec.size(-1)) #패딩 크기 계산
        padded_spec = F.pad(spec, padding, value=0) #패딩 적용
        padded_mel_specs.append(padded_spec) #패딩된 mel 스펙트로그램을 저장

    mel_specs = torch.stack(padded_mel_specs) #패딩된 mel 스펙트로그램을 스택

    text = [item["text"] for item in batch] #텍스트를 정리
    text_lengths = torch.LongTensor([len(item) for item in text]) #텍스트의 길이를 정리

    return dict( #mel 스펙트로그램, mel 스펙트로그램의 길이, 텍스트, 텍스트의 길이를 반환
        mel=mel_specs, #mel 스펙트로그램    
        mel_lengths=mel_lengths, #mel 스펙트로그램의 길이
        text=text, #텍스트
        text_lengths=text_lengths, #텍스트의 길이
    )
