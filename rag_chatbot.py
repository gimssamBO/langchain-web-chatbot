# -*- coding: utf-8 -*-
"""
rag_chatbot.py — 문서 기반 RAG 챗봇 (Streamlit UI - OpenAI 버전)

실행: python -m streamlit run rag_chatbot.py
"""

import os
import pathlib
import traceback
from dotenv import load_dotenv

# ── [보완 1] SSL 인증서 경로 충돌 방지 ───────────────────────────────────────
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]
if "REQUESTS_CA_BUNDLE" in os.environ:
    del os.environ["REQUESTS_CA_BUNDLE"]

# 프로젝트 루트 경로 (.env, data/ 등 기준점)
ROOT = pathlib.Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")  # .env 파일에서 환경변수 로드

import streamlit as st

# ── 페이지 설정 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="문서 기반 RAG 챗봇",
    page_icon="📚",
    layout="wide",
)

# ── 상수 ─────────────────────────────────────────────────────────────────────
DOCS_DIR = ROOT / "data" / "docs"   # PDF 문서 폴더 경로

CHUNK_SIZE = 500        # 문서를 나누는 청크 크기 (글자 수)
CHUNK_OVERLAP = 50      # 청크 간 겹치는 글자 수 (문맥 손실 방지)
TOP_K = 4               # 검색 시 가져올 유사 청크 개수

# 사용할 OpenAI 모델 (.env에서 덮어쓸 수 있음)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# LLM에게 전달하는 역할·답변 규칙 지침
SYSTEM_PROMPT = """너는 회사 내부 문서를 기반으로 정보를 안내하는 친절한 AI 상담원이다.

답변 규칙:
1. 반드시 제공된 문서 내용(context)을 근거로 답하라.
2. 문서에 없는 내용은 "해당 내용은 문서에서 찾을 수 없습니다."라고 솔직히 말하라.
3. 답변 끝에 어떤 문서를 참고했는지 출처를 밝혀라. (예: 📄 출처: 환불교환정책.pdf p.2)
4. 친절하고 명확한 한국어로 답하라.
"""

# ── 헬퍼 함수 ────────────────────────────────────────────────────────────────

def require_openai_key():
    """OPENAI_API_KEY가 없으면 안내 메시지 표시 후 앱 중단"""
    key = os.getenv("OPENAI_API_KEY", "")
    if not key or key.startswith("여기에"):
        st.error(
            "❌ OPENAI_API_KEY가 설정되지 않았습니다.\n\n"
            "1. `.env.example`을 복사해 `.env`로 저장\n"
            "2. `.env` 파일에 `OPENAI_API_KEY=sk-...` 입력\n"
            "3. 앱 재시작"
        )
        st.stop()


@st.cache_resource(show_spinner="📚 문서를 읽고 벡터 인덱스를 구축하는 중...")
def build_vectorstore():
    """
    PDF → 청크 분할 → 임베딩 → FAISS 색인 파이프라인
    """
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings

    require_openai_key()

    # DOCS_DIR 존재 여부 확인 및 자동 생성
    if not DOCS_DIR.exists():
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        st.error(f"❌ `{DOCS_DIR}` 폴더가 없어 생성했습니다. 해당 폴더에 PDF 파일을 넣어주세요.")
        st.stop()

    # DOCS_DIR 안의 PDF 파일 목록 수집
    pdf_files = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        st.error(f"❌ `{DOCS_DIR.resolve()}` 에 PDF 파일이 없습니다.")
        st.stop()

    # ① PDF 로드: 절대 경로(resolve)를 전달하여 [Errno 2] 방지
    all_docs = []
    for pdf in pdf_files:
        pdf_path = str(pdf.resolve())  # 절대 경로 문자열로 변환
        pages = PyPDFLoader(pdf_path).load()
        for page in pages:
            page.metadata["source_name"] = pdf.name   # 출처 표시용 파일명 저장
        all_docs.extend(pages)

    # ② 청크 분할
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)

    # ③ 임베딩 & FAISS 색인
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore, [p.name for p in pdf_files], len(chunks)


def format_source(doc) -> str:
    """검색된 청크의 출처를 '파일명 p.페이지' 형식으로 반환"""
    meta = doc.metadata
    name = meta.get("source_name") or pathlib.Path(meta.get("source", "")).name
    page = meta.get("page", "")
    page_str = f" p.{page + 1}" if page != "" else ""
    return f"📄 {name}{page_str}"


def retrieve_and_answer(question: str, vectorstore, chat_history: list) -> tuple[str, list]:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

    docs = vectorstore.similarity_search(question, k=TOP_K)

    context_parts = []
    for i, doc in enumerate(docs, 1):
        src = format_source(doc)
        context_parts.append(f"[{i}] {src}\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for turn in chat_history[-6:]:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))

    user_msg = f"""다음 문서 내용을 참고해 질문에 답하라.

=== 참고 문서 ===
{context}

=== 질문 ===
{question}"""
    messages.append(HumanMessage(content=user_msg))

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)
    response = llm.invoke(messages)

    sources = list(dict.fromkeys(format_source(d) for d in docs))
    return response.content, sources


# ── Streamlit UI ──────────────────────────────────────────────────────────────

def main():
    if st.query_params.get("reset"):
        st.session_state.messages = []
        st.query_params.clear()
        st.rerun()

    with st.sidebar:
        st.title("📚 RAG 챗봇")
        st.caption("문서 기반 정보 검색 챗봇 (OpenAI)")
        st.divider()

        api_key_input = st.text_input(
            "OpenAI API Key (선택)",
            type="password",
            placeholder=".env 설정시 비워두세요",
        )
        if api_key_input:
            os.environ["OPENAI_API_KEY"] = api_key_input

        st.divider()

        if DOCS_DIR.exists():
            st.subheader("📂 색인된 문서")
            for pdf in sorted(DOCS_DIR.glob("*.pdf")):
                st.markdown(f"- {pdf.name}")
        else:
            st.warning(f"`{DOCS_DIR}` 폴더가 없습니다.")

        st.divider()

        if st.button("🗑️ 대화 초기화"):
            st.session_state.messages = []
            st.rerun()

    st.markdown(
        '<h1><a href="/?reset=true" style="text-decoration:none; color:inherit;">'
        '📚 문서 기반 RAG 챗봇</a></h1>',
        unsafe_allow_html=True,
    )
    st.caption("회사 내부 문서(매뉴얼·정책·핸드북)를 기반으로 정확한 정보를 제공합니다.")

    # ── [보완 2] 상세 에러 스택 트레이스 출력 ───────────────────────────────
    try:
        vectorstore, doc_names, chunk_count = build_vectorstore()
        st.success(f"✅ {len(doc_names)}개 문서 색인 완료 ({chunk_count:,}개 청크)")
    except SystemExit:
        return
    except Exception as e:
        st.error(f"❌ 벡터 인덱스 구축 실패: {e}")
        # 어떤 파일/코드에서 Errno 2가 났는지 구체적인 경로 표시
        st.code(traceback.format_exc(), language="python")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 참고 문서"):
                    for src in msg["sources"]:
                        st.markdown(src)

    if not st.session_state.messages:
        st.markdown("#### 💡 이런 걸 물어보세요")
        example_questions = [
            "환불 신청은 어떻게 하나요?",
            "로봇청소기 필터 청소 방법을 알려주세요",
            "멤버십 등급별 혜택은 무엇인가요?",
            "스마트워치 배터리 교체 주기는?",
            "신입 직원 온보딩 절차를 알려주세요",
        ]
        cols = st.columns(len(example_questions))
        for col, q in zip(cols, example_questions):
            if col.button(q, use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()

    if "pending_question" in st.session_state:
        user_input = st.session_state.pop("pending_question")
    else:
        user_input = st.chat_input("문서에 대해 질문하세요...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("문서에서 관련 내용을 검색하는 중..."):
                try:
                    answer, sources = retrieve_and_answer(
                        user_input,
                        vectorstore,
                        st.session_state.messages[:-1],
                    )
                    st.markdown(answer)
                    if sources:
                        with st.expander("📎 참고 문서"):
                            for src in sources:
                                st.markdown(src)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                except Exception as e:
                    error_msg = f"❌ 오류 발생: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )


if __name__ == "__main__":
    main()