# ruff: noqa: E402
# Above allows ruff to ignore E402: module level import not at top of file

import re # 정규 표현식을 사용하기 위한 모듈
import tempfile # 임시 파일 및 디렉토리를 생성하는 데 사용되는 모듈

import click # 명령줄 인터페이스를 만드는 데 사용되는 모듈
import gradio as gr # 웹 기반 인터페이스를 쉽게 만들 수 있는 라이브러리
import numpy as np # 수치 계산을 위한 라이브러리
import soundfile as sf # 오디오 파일을 읽고 쓰는 데 사용되는 라이브러리
import torchaudio # PyTorch에서 오디오 처리를 위한 라이브러리
from cached_path import cached_path # 파일을 캐시하여 다운로드하는 데 사용되는 유틸리티
from transformers import AutoModelForCausalLM, AutoTokenizer # 자연어 처리 모델을 위한 라이브러리

# 조건부임포트 및 설정
try:
    import spaces # 모듈을 임포트하려고 시도

    USING_SPACES = True # 성공하면 usingspaces를 true설정
except ImportError: #실패하면
    USING_SPACES = False #false로 설정 이는 spaces 모듈의 사용 가능 여부를 확인하는 데 사용

# GPU 데코레이터 정의 : gpu_decorator 함수는 함수에 gpu 지원을 추가 하는 데 사용된다.
def gpu_decorator(func):
    if USING_SPACES: # 만약 spaces 모듈이 사용 가능하다면
        return spaces.GPU(func) # spaces.GPU 데코레이터를 함수에 적용
    else: # 그렇지 않다면
        return func # 함수를 그대로 반환

# 모델 및 유틸리티 임포트
from f5_tts.model import DiT, UNetT # f5_tts.model에서 DiT, UNetT 임포트
from f5_tts.infer.utils_infer import ( # f5_tts.infer.utils_infer에서 여러 유틸리티 함수 임포트
    load_vocoder, # 보코더
    load_model, # 모델
    preprocess_ref_audio_text, # 참조 오디오 텍스트 전처리
    infer_process, # 추론 프로세스
    remove_silence_for_generated_wav, # 생성된 오디오에서 무음 없애기
    save_spectrogram, # 스펙트로그램 저장
)

# 보코더 및 모델 로드
vocoder = load_vocoder() # 보코더 로드

# load models
F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4) # F5TTS 모델의 설정을 정의하는 딕셔너리
F5TTS_ema_model = load_model( # F5TTS 모델 로드
    DiT, F5TTS_model_cfg, str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors")) # 허깅페이스 허브에서 F5TTS 모델 로드
)
# E2TTS 모델 로드
E2TTS_model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4) # E2TTS 모델의 설정을 정의하는 딕셔너리
E2TTS_ema_model = load_model( # E2TTS 모델 로드
    UNetT, E2TTS_model_cfg, str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors")) # 허깅페이스 허브에서 E2TTS 모델 로드
)
# 챗모델 상태 및 토크나이저 상태 초기화
chat_model_state = None # 챗모델 상태
chat_tokenizer_state = None # 챗 토크나이저 상태

# 챗모델 응답 생성 함수
@gpu_decorator
def generate_response(messages, model, tokenizer): # generate_response 함수는 주어진 메시지를 사용하여 AI 응답을 생성하는 데 사용된다.
    """Generate response using Qwen""" # 퀸 모델을 사용하여 응답 생성
    text = tokenizer.apply_chat_template( # 챗 템플릿을 적용하여 메시지를 토큰화하는 데 사용된다.
        messages, # 메시지
        tokenize=False, # 토큰화 여부
        add_generation_prompt=True, # 생성 프롬프트 추가 여부
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device) # 토큰화된 메시지를 텐서로 변환하고 모델 장치에 배치
    generated_ids = model.generate( # model.generate를 사용하여 응답 생성
        **model_inputs, # model_inputs를 키워드 인자로 전달
        max_new_tokens=512, # 최대 새로운 토큰 수
        temperature=0.7, # temperature 매개변수
        top_p=0.95, # 상위 p 매개변수
    )

    generated_ids = [ # 생성된 토큰 추출
        output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids) # 입력 토큰과 생성된 토큰 쌍을 순회하며 생성된 토큰 추출
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0] # 생성된 토큰을 디코딩하고 특수 토큰 건너뛰기

# 추론 함수
@gpu_decorator # gpu_decorator 데코레이터를 사용하여 함수에 gpu 지원을 추가
def infer( # infer 함수는 참조 오디오, 참조 텍스트, 생성할 텍스트, 모델 선택, 무음 제거 여부, 크로스 페이드 지속 시간, 속도, 정보 표시 여부를 인자로 받는다.
    ref_audio_orig, ref_text, gen_text, model, remove_silence, cross_fade_duration=0.15, speed=1, show_info=gr.Info
    # ref_audio_orig: 참조 오디오, ref_text: 참조 텍스트, gen_text: 생성할 텍스트, model: 모델 선택, remove_silenc
):
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=show_info) # 참조 오디오 및 텍스트 전처리
    # ref_audio: 참조 오디오, ref_text: 참조 텍스트 = preprocess_ref_audio_text는 참조 오디오와 텍스트를 전처리하는 함수(ref_audio_orig는 참조 오디오, ref_text는 참조 텍스트, show_info는 정보 표시 여부)
    if model == "F5-TTS": # 만약 모델이 F5-TTS라면
        ema_model = F5TTS_ema_model # F5TTS_ema_model 할당
    elif model == "E2-TTS": # 만약 모델이 E2-TTS라면
        ema_model = E2TTS_ema_model # E2TTS_ema_model 할당
    # final_wave는 최종 오디오 신호, final_sample_rate는 최종 오디오 신호의 샘플 레이트, combined_spectrogram는 결합된 스펙트로그램
    final_wave, final_sample_rate, combined_spectrogram = infer_process( # infer_process 함수 호출
        ref_audio, # 참조 오디오
        ref_text, # 참조 텍스트
        gen_text, # 생성할 텍스트
        ema_model, # 모델
        vocoder, # 보코더
        cross_fade_duration=cross_fade_duration, # 크로스 페이드 지속 시간
        speed=speed, # 속도
        show_info=show_info, # 정보 표시 여부
        progress=gr.Progress(), # 진행 상황 표시
    )

    # 무음 제거
    if remove_silence: # 만약 무음 제거 여부가 True라면
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f: # 임시 파일 생성
            sf.write(f.name, final_wave, final_sample_rate) # 오디오 데이터를 임시 파일에 쓰기
            remove_silence_for_generated_wav(f.name) # 무음 제거
            final_wave, _ = torchaudio.load(f.name) # 오디오 데이터 로드
        final_wave = final_wave.squeeze().cpu().numpy() # 오디오 데이터 정리

    # 스펙트로그램 저장
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram: # 임시 파일 생성
        spectrogram_path = tmp_spectrogram.name # 스펙트로그램 경로 저장
        save_spectrogram(combined_spectrogram, spectrogram_path) # 스펙트로그램 저장

    return (final_sample_rate, final_wave), spectrogram_path


# Gradio 블록을 사용하여 텍스트-음성 변환(TTS) 인터페이스를 정의합니다.
with gr.Blocks() as app_tts:
    # TTS 섹션의 제목을 마크다운 형식으로 표시합니다.
    gr.Markdown("# Batched TTS")
    
    # 참조 오디오 파일을 업로드할 수 있는 입력 필드를 정의합니다.
    ref_audio_input = gr.Audio(label="Reference Audio", type="filepath")
    
    # 생성할 텍스트를 입력할 수 있는 텍스트 박스를 정의합니다.
    gen_text_input = gr.Textbox(label="Text to Generate", lines=10)
    
    # 사용할 TTS 모델을 선택할 수 있는 라디오 버튼을 정의합니다.
    model_choice = gr.Radio(choices=["F5-TTS", "E2-TTS"], label="Choose TTS Model", value="F5-TTS")
    
    # 음성을 생성하는 버튼을 정의합니다.
    generate_btn = gr.Button("Synthesize", variant="primary")
    
    # 고급 설정을 위한 아코디언을 정의합니다.
    with gr.Accordion("Advanced Settings", open=False):
        # 참조 텍스트를 입력할 수 있는 텍스트 박스를 정의합니다.
        ref_text_input = gr.Textbox(
            label="Reference Text",
            info="Leave blank to automatically transcribe the reference audio. If you enter text it will override automatic transcription.",
            lines=2,
        )
        
        # 침묵 제거 여부를 선택할 수 있는 체크박스를 정의합니다.
        remove_silence = gr.Checkbox(
            label="Remove Silences",
            info="The model tends to produce silences, especially on longer audio. We can manually remove silences if needed. Note that this is an experimental feature and may produce strange results. This will also increase generation time.",
            value=False,
        )
        
        # 오디오의 속도를 조절할 수 있는 슬라이더를 정의합니다.
        speed_slider = gr.Slider(
            label="Speed",
            minimum=0.3,
            maximum=2.0,
            value=1.0,
            step=0.1,
            info="Adjust the speed of the audio.",
        )
        
        # 오디오 클립 간의 크로스 페이드 지속 시간을 설정할 수 있는 슬라이더를 정의합니다.
        cross_fade_duration_slider = gr.Slider(
            label="Cross-Fade Duration (s)",
            minimum=0.0,
            maximum=1.0,
            value=0.15,
            step=0.01,
            info="Set the duration of the cross-fade between audio clips.",
        )

    # 생성된 음성을 출력할 오디오 컴포넌트를 정의합니다.
    audio_output = gr.Audio(label="Synthesized Audio")
    
    # 생성된 스펙트로그램을 출력할 이미지 컴포넌트를 정의합니다.
    spectrogram_output = gr.Image(label="Spectrogram")

    # 버튼 클릭 시 infer 함수를 호출하여 음성을 생성합니다.
    generate_btn.click(
        infer,  # 호출할 함수
        inputs=[
            ref_audio_input,  # 참조 오디오 입력
            ref_text_input,   # 참조 텍스트 입력
            gen_text_input,   # 생성할 텍스트 입력
            model_choice,     # 선택한 모델
            remove_silence,   # 침묵 제거 여부
            cross_fade_duration_slider,  # 크로스 페이드 지속 시간
            speed_slider,     # 오디오 속도
        ],
        outputs=[audio_output, spectrogram_output],  # 출력 컴포넌트
    )


def parse_speechtypes_text(gen_text):
    """
    주어진 텍스트에서 {speechtype} 형식의 스타일을 파싱하여 텍스트와 스타일을 분리합니다.

    인자:
    gen_text (str): 스타일을 포함한 생성할 텍스트.

    반환:
    list: 각 텍스트 조각과 해당 스타일을 포함하는 딕셔너리의 리스트.
    """
    # Pattern to find {speechtype}
    pattern = r"\{(.*?)\}" # {speechtype} 패턴을 찾기 위한 정규 표현식 패턴을 정의합니다.

    # Split the text by the pattern
    tokens = re.split(pattern, gen_text) # 주어진 텍스트를 패턴에 따라 분리합니다.

    segments = [] # 텍스트 조각과 스타일을 저장할 리스트를 초기화합니다.

    current_style = "Regular" # 현재 스타일을 "Regular"로 초기화합니다.

    for i in range(len(tokens)): # 분리된 토큰을 순회합니다.
        if i % 2 == 0: # 만약 인덱스가 짝수라면 (짝수 인덱스는 텍스트)
            # This is text
            text = tokens[i].strip() # 텍스트의 앞뒤 공백을 제거합니다.
            if text: # 만약 텍스트가 있다면
                segments.append({"style": current_style, "text": text}) # 현재 텍스트와 스타일을 저장합니다.
        else: # 만약 인덱스가 홀수라면 (홀수 인덱스는 스타일)
            # This is style
            style = tokens[i].strip() # 스타일의 앞뒤 공백을 제거합니다.
            current_style = style # 현재 스타일을 업데이트합니다.

    return segments # 텍스트 조각과 스타일을 포함하는 딕셔너리의 리스트를 반환합니다.


with gr.Blocks() as app_multistyle: # 다중 스타일 생성을 위한 Gradio 블록을 정의합니다.
    # New section for multistyle generation
    gr.Markdown( # 다중 스타일 생성을 위한 설명을 표시합니다.
        """
    # Multiple Speech-Type Generation

    This section allows you to generate multiple speech types or multiple people's voices. Enter your text in the format shown below, and the system will generate speech using the appropriate type. If unspecified, the model will use the regular speech type. The current speech type will be used until the next speech type is specified.
    """
    )

    with gr.Row(): # 다중 스타일 생성을 위한 예시 입력을 표시합니다.
        gr.Markdown( # 예시 입력을 표시합니다.
            """
            **Example Input:**                                                                      
            {Regular} Hello, I'd like to order a sandwich please.                                                         
            {Surprised} What do you mean you're out of bread?                                                                      
            {Sad} I really wanted a sandwich though...                                                              
            {Angry} You know what, darn you and your little shop!                                                                       
            {Whisper} I'll just go back home and cry now.                                                                           
            {Shouting} Why me?!                                                                         
            """
        )

        gr.Markdown( # 예시 입력 2를 표시합니다.
            """
            **Example Input 2:**                                                                                
            {Speaker1_Happy} Hello, I'd like to order a sandwich please.                                                            
            {Speaker2_Regular} Sorry, we're out of bread.                                                                                
            {Speaker1_Sad} I really wanted a sandwich though...                                                                             
            {Speaker2_Whisper} I'll give you the last one I was hiding.                                                                     
            """
        )

    gr.Markdown( # 다중 스타일 생성을 위한 설명을 표시합니다.
        "Upload different audio clips for each speech type. The first speech type is mandatory. You can add additional speech types by clicking the 'Add Speech Type' button."
    )

    # Regular speech type (mandatory)
    with gr.Row(): # 기본 스타일을 위한 입력 필드를 정의합니다.
        with gr.Column(): # 기본 스타일을 위한 입력 필드를 정의합니다.
            regular_name = gr.Textbox(value="Regular", label="Speech Type Name") # 기본 스타일 이름을 입력할 수 있는 텍스트 박스를 정의합니다.
            regular_insert = gr.Button("Insert", variant="secondary") # 기본 스타일을 삽입할 수 있는 버튼을 정의합니다.
        regular_audio = gr.Audio(label="Regular Reference Audio", type="filepath") # 기본 스타일 오디오를 업로드할 수 있는 오디오 컴포넌트를 정의합니다.
        regular_ref_text = gr.Textbox(label="Reference Text (Regular)", lines=2)    # 기본 스타일 참조 텍스트를 입력할 수 있는 텍스트 박스를 정의합니다.

    # Additional speech types (up to 99 more)
    max_speech_types = 100 # 최대 스타일 수를 100으로 정의합니다.
    speech_type_rows = [] # 스타일 행을 저장할 리스트를 초기화합니다.
    speech_type_names = [regular_name] # 스타일 이름을 저장할 리스트를 초기화합니다.
    speech_type_audios = [] # 스타일 오디오를 저장할 리스트를 초기화합니다.
    speech_type_ref_texts = [] # 스타일 참조 텍스트를 저장할 리스트를 초기화합니다.
    speech_type_delete_btns = [] # 스타일 삭제 버튼을 저장할 리스트를 초기화합니다.
    speech_type_insert_btns = [] # 스타일 삽입 버튼을 저장할 리스트를 초기화합니다.
    speech_type_insert_btns.append(regular_insert) # 기본 스타일 삽입 버튼을 추가합니다.

    for i in range(max_speech_types - 1): # 최대 스타일 수 - 1 만큼 반복합니다.
        with gr.Row(visible=False) as row: # 스타일 행을 정의합니다.
            with gr.Column(): # 스타일 열을 정의합니다.
                name_input = gr.Textbox(label="Speech Type Name") # 스타일 이름을 입력할 수 있는 텍스트 박스를 정의합니다.
                delete_btn = gr.Button("Delete", variant="secondary") # 스타일을 삭제할 수 있는 버튼을 정의합니다.
                insert_btn = gr.Button("Insert", variant="secondary") # 스타일을 삽입할 수 있는 버튼을 정의합니다.
            audio_input = gr.Audio(label="Reference Audio", type="filepath") # 스타일 오디오를 업로드할 수 있는 오디오 컴포넌트를 정의합니다.
            ref_text_input = gr.Textbox(label="Reference Text", lines=2) # 스타일 참조 텍스트를 입력할 수 있는 텍스트 박스를 정의합니다.
        speech_type_rows.append(row) # 스타일 행을 저장합니다.
        speech_type_names.append(name_input) # 스타일 이름을 저장합니다.
        speech_type_audios.append(audio_input) # 스타일 오디오를 저장합니다.
        speech_type_ref_texts.append(ref_text_input) # 스타일 참조 텍스트를 저장합니다.
        speech_type_delete_btns.append(delete_btn) # 스타일 삭제 버튼을 저장합니다.
        speech_type_insert_btns.append(insert_btn) # 스타일 삽입 버튼을 저장합니다.

    # Button to add speech type
    add_speech_type_btn = gr.Button("Add Speech Type") # 스타일을 추가할 수 있는 버튼을 정의합니다.

    # Keep track of current number of speech types
    speech_type_count = gr.State(value=0) # 현재 스타일 수를 저장할 상태를 정의합니다.

    # Function to add a speech type
    def add_speech_type_fn(speech_type_count): # 스타일을 추가하는 함수를 정의합니다.
        if speech_type_count < max_speech_types - 1: # 만약 현재 스타일 수가 최대 스타일 수 - 1보다 작다면
            speech_type_count += 1  # 스타일 수를 1 증가시킵니다.
            # Prepare updates for the rows
            row_updates = [] # 스타일 행을 업데이트할 리스트를 초기화합니다.
            for i in range(max_speech_types - 1): # 최대 스타일 수 - 1 만큼 반복합니다.
                if i < speech_type_count: # 만약 인덱스가 현재 스타일 수보다 작다면
                    row_updates.append(gr.update(visible=True)) # 스타일 행을 표시합니다.
                else: # 만약 인덱스가 현재 스타일 수보다 크다면
                    row_updates.append(gr.update()) # 스타일 행을 숨깁니다.
        else: # 만약 현재 스타일 수가 최대 스타일 수 - 1보다 크다면
            # Optionally, show a warning
            row_updates = [gr.update() for _ in range(max_speech_types - 1)] # 모든 스타일 행을 표시합니다.
        return [speech_type_count] + row_updates # 현재 스타일 수와 스타일 행을 반환합니다.

    add_speech_type_btn.click( # 스타일을 추가하는 버튼을 클릭하면 스타일을 추가하는 함수를 호출합니다.
        add_speech_type_fn, inputs=speech_type_count, outputs=[speech_type_count] + speech_type_rows 
    ) 

    # Function to delete a speech type
    def make_delete_speech_type_fn(index): 
        def delete_speech_type_fn(speech_type_count):
            # Prepare updates
            row_updates = []

            for i in range(max_speech_types - 1):
                if i == index:
                    row_updates.append(gr.update(visible=False))
                else:
                    row_updates.append(gr.update())

            speech_type_count = max(0, speech_type_count - 1)

            return [speech_type_count] + row_updates

        return delete_speech_type_fn

    # Update delete button clicks
    for i, delete_btn in enumerate(speech_type_delete_btns):
        delete_fn = make_delete_speech_type_fn(i)
        delete_btn.click(delete_fn, inputs=speech_type_count, outputs=[speech_type_count] + speech_type_rows)

    # Text input for the prompt
    gen_text_input_multistyle = gr.Textbox(
        label="Text to Generate",
        lines=10,
        placeholder="Enter the script with speaker names (or emotion types) at the start of each block, e.g.:\n\n{Regular} Hello, I'd like to order a sandwich please.\n{Surprised} What do you mean you're out of bread?\n{Sad} I really wanted a sandwich though...\n{Angry} You know what, darn you and your little shop!\n{Whisper} I'll just go back home and cry now.\n{Shouting} Why me?!",
    )

    def make_insert_speech_type_fn(index):
        def insert_speech_type_fn(current_text, speech_type_name):
            current_text = current_text or ""
            speech_type_name = speech_type_name or "None"
            updated_text = current_text + f"{{{speech_type_name}}} "
            return gr.update(value=updated_text)

        return insert_speech_type_fn

    for i, insert_btn in enumerate(speech_type_insert_btns):
        insert_fn = make_insert_speech_type_fn(i)
        insert_btn.click(
            insert_fn,
            inputs=[gen_text_input_multistyle, speech_type_names[i]],
            outputs=gen_text_input_multistyle,
        )

    # Model choice
    model_choice_multistyle = gr.Radio(choices=["F5-TTS", "E2-TTS"], label="Choose TTS Model", value="F5-TTS")

    with gr.Accordion("Advanced Settings", open=False):
        remove_silence_multistyle = gr.Checkbox(
            label="Remove Silences",
            value=False,
        )

    # Generate button
    generate_multistyle_btn = gr.Button("Generate Multi-Style Speech", variant="primary")

    # Output audio
    audio_output_multistyle = gr.Audio(label="Synthesized Audio")

    @gpu_decorator
    def generate_multistyle_speech(
        regular_audio,
        regular_ref_text,
        gen_text,
        *args,
    ):
        num_additional_speech_types = max_speech_types - 1
        speech_type_names_list = args[:num_additional_speech_types]
        speech_type_audios_list = args[num_additional_speech_types : 2 * num_additional_speech_types]
        speech_type_ref_texts_list = args[2 * num_additional_speech_types : 3 * num_additional_speech_types]
        model_choice = args[3 * num_additional_speech_types + 1]
        remove_silence = args[3 * num_additional_speech_types + 1]

        # Collect the speech types and their audios into a dict
        speech_types = {"Regular": {"audio": regular_audio, "ref_text": regular_ref_text}}

        for name_input, audio_input, ref_text_input in zip(
            speech_type_names_list, speech_type_audios_list, speech_type_ref_texts_list
        ):
            if name_input and audio_input:
                speech_types[name_input] = {"audio": audio_input, "ref_text": ref_text_input}

        # Parse the gen_text into segments
        segments = parse_speechtypes_text(gen_text)

        # For each segment, generate speech
        generated_audio_segments = []
        current_style = "Regular"

        for segment in segments:
            style = segment["style"]
            text = segment["text"]

            if style in speech_types:
                current_style = style
            else:
                # If style not available, default to Regular
                current_style = "Regular"

            ref_audio = speech_types[current_style]["audio"]
            ref_text = speech_types[current_style].get("ref_text", "")

            # Generate speech for this segment
            audio, _ = infer(
                ref_audio, ref_text, text, model_choice, remove_silence, 0, show_info=print
            )  # show_info=print no pull to top when generating
            sr, audio_data = audio

            generated_audio_segments.append(audio_data)

        # Concatenate all audio segments
        if generated_audio_segments:
            final_audio_data = np.concatenate(generated_audio_segments)
            return (sr, final_audio_data)
        else:
            gr.Warning("No audio generated.")
            return None

    generate_multistyle_btn.click(
        generate_multistyle_speech,
        inputs=[
            regular_audio,
            regular_ref_text,
            gen_text_input_multistyle,
        ]
        + speech_type_names
        + speech_type_audios
        + speech_type_ref_texts
        + [
            model_choice_multistyle,
            remove_silence_multistyle,
        ],
        outputs=audio_output_multistyle,
    )

    # Validation function to disable Generate button if speech types are missing
    def validate_speech_types(gen_text, regular_name, *args):
        num_additional_speech_types = max_speech_types - 1
        speech_type_names_list = args[:num_additional_speech_types]

        # Collect the speech types names
        speech_types_available = set()
        if regular_name:
            speech_types_available.add(regular_name)
        for name_input in speech_type_names_list:
            if name_input:
                speech_types_available.add(name_input)

        # Parse the gen_text to get the speech types used
        segments = parse_speechtypes_text(gen_text)
        speech_types_in_text = set(segment["style"] for segment in segments)

        # Check if all speech types in text are available
        missing_speech_types = speech_types_in_text - speech_types_available

        if missing_speech_types:
            # Disable the generate button
            return gr.update(interactive=False)
        else:
            # Enable the generate button
            return gr.update(interactive=True)

    gen_text_input_multistyle.change(
        validate_speech_types,
        inputs=[gen_text_input_multistyle, regular_name] + speech_type_names,
        outputs=generate_multistyle_btn,
    )


with gr.Blocks() as app_chat:
    gr.Markdown(
        """
# Voice Chat
Have a conversation with an AI using your reference voice! 
1. Upload a reference audio clip and optionally its transcript.
2. Load the chat model.
3. Record your message through your microphone.
4. The AI will respond using the reference voice.
"""
    )

    if not USING_SPACES:
        load_chat_model_btn = gr.Button("Load Chat Model", variant="primary")

        chat_interface_container = gr.Column(visible=False)

        @gpu_decorator
        def load_chat_model():
            global chat_model_state, chat_tokenizer_state
            if chat_model_state is None:
                show_info = gr.Info
                show_info("Loading chat model...")
                model_name = "Qwen/Qwen2.5-3B-Instruct"
                chat_model_state = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype="auto", device_map="auto"
                )
                chat_tokenizer_state = AutoTokenizer.from_pretrained(model_name)
                show_info("Chat model loaded.")

            return gr.update(visible=False), gr.update(visible=True)

        load_chat_model_btn.click(load_chat_model, outputs=[load_chat_model_btn, chat_interface_container])

    else:
        chat_interface_container = gr.Column()

        if chat_model_state is None:
            model_name = "Qwen/Qwen2.5-3B-Instruct"
            chat_model_state = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
            chat_tokenizer_state = AutoTokenizer.from_pretrained(model_name)

    with chat_interface_container:
        with gr.Row():
            with gr.Column():
                ref_audio_chat = gr.Audio(label="Reference Audio", type="filepath")
            with gr.Column():
                with gr.Accordion("Advanced Settings", open=False):
                    model_choice_chat = gr.Radio(
                        choices=["F5-TTS", "E2-TTS"],
                        label="TTS Model",
                        value="F5-TTS",
                    )
                    remove_silence_chat = gr.Checkbox(
                        label="Remove Silences",
                        value=True,
                    )
                    ref_text_chat = gr.Textbox(
                        label="Reference Text",
                        info="Optional: Leave blank to auto-transcribe",
                        lines=2,
                    )
                    system_prompt_chat = gr.Textbox(
                        label="System Prompt",
                        value="You are not an AI assistant, you are whoever the user says you are. You must stay in character. Keep your responses concise since they will be spoken out loud.",
                        lines=2,
                    )

        chatbot_interface = gr.Chatbot(label="Conversation")

        with gr.Row():
            with gr.Column():
                audio_input_chat = gr.Microphone(
                    label="Speak your message",
                    type="filepath",
                )
                audio_output_chat = gr.Audio(autoplay=True)
            with gr.Column():
                text_input_chat = gr.Textbox(
                    label="Type your message",
                    lines=1,
                )
                send_btn_chat = gr.Button("Send")
                clear_btn_chat = gr.Button("Clear Conversation")

        conversation_state = gr.State(
            value=[
                {
                    "role": "system",
                    "content": "You are not an AI assistant, you are whoever the user says you are. You must stay in character. Keep your responses concise since they will be spoken out loud.",
                }
            ]
        )

        # Modify process_audio_input to use model and tokenizer from state
        @gpu_decorator
        def process_audio_input(audio_path, text, history, conv_state):
            """Handle audio or text input from user"""

            if not audio_path and not text.strip():
                return history, conv_state, ""

            if audio_path:
                text = preprocess_ref_audio_text(audio_path, text)[1]

            if not text.strip():
                return history, conv_state, ""

            conv_state.append({"role": "user", "content": text})
            history.append((text, None))

            response = generate_response(conv_state, chat_model_state, chat_tokenizer_state)

            conv_state.append({"role": "assistant", "content": response})
            history[-1] = (text, response)

            return history, conv_state, ""

        @gpu_decorator
        def generate_audio_response(history, ref_audio, ref_text, model, remove_silence):
            """Generate TTS audio for AI response"""
            if not history or not ref_audio:
                return None

            last_user_message, last_ai_response = history[-1]
            if not last_ai_response:
                return None

            audio_result, _ = infer(
                ref_audio,
                ref_text,
                last_ai_response,
                model,
                remove_silence,
                cross_fade_duration=0.15,
                speed=1.0,
                show_info=print,  # show_info=print no pull to top when generating
            )
            return audio_result

        def clear_conversation():
            """Reset the conversation"""
            return [], [
                {
                    "role": "system",
                    "content": "You are not an AI assistant, you are whoever the user says you are. You must stay in character. Keep your responses concise since they will be spoken out loud.",
                }
            ]

        def update_system_prompt(new_prompt):
            """Update the system prompt and reset the conversation"""
            new_conv_state = [{"role": "system", "content": new_prompt}]
            return [], new_conv_state

        # Handle audio input
        audio_input_chat.stop_recording(
            process_audio_input,
            inputs=[audio_input_chat, text_input_chat, chatbot_interface, conversation_state],
            outputs=[chatbot_interface, conversation_state],
        ).then(
            generate_audio_response,
            inputs=[chatbot_interface, ref_audio_chat, ref_text_chat, model_choice_chat, remove_silence_chat],
            outputs=[audio_output_chat],
        ).then(
            lambda: None,
            None,
            audio_input_chat,
        )

        # Handle text input
        text_input_chat.submit(
            process_audio_input,
            inputs=[audio_input_chat, text_input_chat, chatbot_interface, conversation_state],
            outputs=[chatbot_interface, conversation_state],
        ).then(
            generate_audio_response,
            inputs=[chatbot_interface, ref_audio_chat, ref_text_chat, model_choice_chat, remove_silence_chat],
            outputs=[audio_output_chat],
        ).then(
            lambda: None,
            None,
            text_input_chat,
        )

        # Handle send button
        send_btn_chat.click(
            process_audio_input,
            inputs=[audio_input_chat, text_input_chat, chatbot_interface, conversation_state],
            outputs=[chatbot_interface, conversation_state],
        ).then(
            generate_audio_response,
            inputs=[chatbot_interface, ref_audio_chat, ref_text_chat, model_choice_chat, remove_silence_chat],
            outputs=[audio_output_chat],
        ).then(
            lambda: None,
            None,
            text_input_chat,
        )

        # Handle clear button
        clear_btn_chat.click(
            clear_conversation,
            outputs=[chatbot_interface, conversation_state],
        )

        # Handle system prompt change and reset conversation
        system_prompt_chat.change(
            update_system_prompt,
            inputs=system_prompt_chat,
            outputs=[chatbot_interface, conversation_state],
        )


with gr.Blocks() as app:
    gr.Markdown(
        """
# E2/F5 TTS

This is a local web UI for F5 TTS with advanced batch processing support. This app supports the following TTS models:

* [F5-TTS](https://arxiv.org/abs/2410.06885) (A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching)
* [E2 TTS](https://arxiv.org/abs/2406.18009) (Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS)

The checkpoints support English and Chinese.

If you're having issues, try converting your reference audio to WAV or MP3, clipping it to 15s, and shortening your prompt.

**NOTE: Reference text will be automatically transcribed with Whisper if not provided. For best results, keep your reference clips short (<15s). Ensure the audio is fully uploaded before generating.**
"""
    )
    gr.TabbedInterface(
        [app_tts, app_multistyle, app_chat, app_credits],
        ["TTS", "Multi-Speech", "Voice-Chat", "Credits"],
    )


@click.command()
@click.option("--port", "-p", default=None, type=int, help="Port to run the app on")
@click.option("--host", "-H", default=None, help="Host to run the app on")
@click.option(
    "--share",
    "-s",
    default=False,
    is_flag=True,
    help="Share the app via Gradio share link",
)
@click.option("--api", "-a", default=True, is_flag=True, help="Allow API access")
def main(port, host, share, api):
    global app
    print("Starting app...")
    app.queue(api_open=api).launch(server_name="0.0.0.0", server_port=7861, share=True, show_api=api)


if __name__ == "__main__":
    if not USING_SPACES:
        main()
    else:
        app.queue().launch()
