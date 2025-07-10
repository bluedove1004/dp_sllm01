
import streamlit as st
import torch
from vllm import LLM, SamplingParams


# 모델 캐싱 최초 한번만 실행
@st.cache_resource
def load_model():
    model_id = "PrunaAI/saltlux-Ko-Llama3-Luxia-8B-bnb-4bit-smashed"
    # LLM 객체 생성
    llm = LLM(
        model=model_id,
        dtype="float16",
        quantization="bitsandbytes",  # bitsandbytes 양자화 사용
        load_format="bitsandbytes",   # load_format을 명시적으로 설정
        max_model_len=512,            # 최대 시퀀스 길이 감소
        gpu_memory_utilization=0.7,   # GPU 메모리 활용도 제한
        max_num_batched_tokens=512,   # 배치된 토큰 수 제한
        max_num_seqs=1,               # 동시에 처리할 시퀀스 수 제한
    )
    return llm

# 모델과 토크나이저 로드
llm = load_model()

# Streamlit 페이지에 모델 설명 출력
st.write('sLLM을 활용한 Streamlit 배포 예제')

# 사용자 입력을 받기 위한 텍스트 입력 상자
user_input = st.text_input('질문을 입력하세요:')

# 모델이 입력에 대해 답변을 생성
if user_input:
    sampling_params = SamplingParams(max_tokens=256)
    outputs = llm.generate([user_input], sampling_params)[0].outputs[0].text
    st.write('모델의 답변:', outputs)
