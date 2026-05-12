import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 환경 변수 로드
load_dotenv()

def get_llm(task_type: str = "text", provider: str = "openai"):
    """
    용도와 제공자에 따라 최적화된 LLM을 반환합니다.
    토큰 비용을 최소화하기 위한 제한(max_tokens) 설정이 포함되어 있습니다.
    """
    temperature = 0.0

    if provider == "openai":
        if task_type == "generation":
            # 💡 [추가된 부분] 기사 작성(Toulmin) 작업은 글을 길게 써야 하므로 토큰을 넉넉하게 줍니다!
            return ChatOpenAI(
                model="gpt-4o",  # 똑똑하고 자연스러운 기사 작성을 위해 gpt-4o 사용
                temperature=0.3, # 글을 약간 더 자연스럽게 쓰도록 온도 살짝업
                max_tokens=2000
            )
            
        elif task_type == "vision":
            # 비전 작업은 gpt-4o를 사용하되, 응답 길이를 150 토큰으로 엄격히 제한
            return ChatOpenAI(
                model="gpt-4o", 
                temperature=temperature,
                max_tokens=150
            )
            
        else:
            # 텍스트 검증(verification/triage)은 단순 판정/짧은 이유만 필요하므로 100 토큰 제한!
            return ChatOpenAI(
                model="gpt-4o-mini", 
                temperature=temperature,
                max_tokens=100 
            )
            
    elif provider == "vllm":
        # 향후 완전 무료인 연구실 로컬 서버(Llama-3 등) 연결용
        api_base = os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")
        return ChatOpenAI(
            model="meta-llama/Meta-Llama-3-8B-Instruct", 
            openai_api_base=api_base, 
            openai_api_key="EMPTY",
            temperature=temperature,
            max_tokens=100
        )
    else:
        raise ValueError(f"❌ 지원하지 않는 제공자입니다: {provider}")