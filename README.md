# 📚 문서 기반 RAG 챗봇

회사 내부 문서(PDF)를 자동으로 색인하고, 자연어 질문에 **문서 근거**와 함께 답변하는 RAG(Retrieval-Augmented Generation) 챗봇입니다.

---

## 주요 기능

| 기능             | 설명                                                     |
| ---------------- | -------------------------------------------------------- |
| 📄 PDF 자동 색인 | `data/docs/` 폴더의 PDF 전체를 자동으로 벡터 인덱싱      |
| 🔍 RAG 검색      | 질문과 유사한 문서 청크 Top-4 검색 후 Gemini로 답변 생성 |
| 📎 출처 표시     | 답변에 참고한 PDF 파일명과 페이지 번호 표시              |
| 💬 대화 기억     | 최근 6턴 대화 이력 유지 (문맥 이어서 질문 가능)          |
| 🏠 홈 이동       | 제목 클릭 시 첫 화면(대화 초기화)으로 이동               |

---

## 색인 문서

```
data/docs/
├── 환불교환정책.pdf
├── 멤버십정책.pdf
├── 제품매뉴얼_로봇청소기.pdf
├── 제품매뉴얼_스마트워치.pdf
└── 직원핸드북.pdf
```

PDF를 추가하거나 교체하면 앱 재시작 시 자동으로 재색인됩니다.

---

## 로컬 실행

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. API 키 설정

```bash
cp .env.example .env
```

`.env` 파일을 열어 Google API 키를 입력합니다.

```env
GOOGLE_API_KEY=발급받은_키_입력
```

> Google AI Studio에서 무료 발급: https://aistudio.google.com/apikey

### 3. 앱 실행

```bash
python -m streamlit run rag_chatbot.py
```

브라우저에서 http://localhost:8501 접속

---

## GitHub 업로드

### 1. Git 초기화 & 첫 커밋 (최초 1회)

```bash
git init
git add .
git commit -m "Initial commit"
```

### 2. GitHub 레포지토리 연결

GitHub에서 새 레포지토리 생성 후:

```bash
git remote add origin https://github.com/본인계정/agai-main.git
git branch -M main
git push -u origin main
```

### 3. 이후 변경사항 업로드

```bash
git add .
git commit -m "변경 내용 설명"
git push
```

> **주의:** `.env` 파일은 `.gitignore`에 등록되어 있어 API 키가 GitHub에 올라가지 않습니다.

---

## Streamlit Cloud 배포

GitHub 레포가 준비된 후 아래 순서로 무료 웹 서비스를 배포할 수 있습니다.

### 1. Streamlit Cloud 접속

https://share.streamlit.io 에서 GitHub 계정으로 로그인

### 2. 앱 생성

**"Create app"** 클릭 후 아래와 같이 입력:

| 항목           | 값                   |
| -------------- | -------------------- |
| Repository     | `본인계정/agai-main` |
| Branch         | `main`               |
| Main file path | `rag_chatbot.py`     |

### 3. API 키 설정 (중요)

**"Advanced settings" → Secrets** 탭에 입력:

```toml
GOOGLE_API_KEY = "발급받은_키_입력"
```

> `.env` 파일은 GitHub에 올라가지 않으므로 Secrets에서 대신 설정합니다.

### 4. 배포

**"Deploy!"** 클릭 → 1~3분 후 `https://xxx.streamlit.app` URL 발급

### GitHub 레포 접근 권한 오류 시

Streamlit Cloud가 레포를 찾지 못하면:

1. Streamlit Cloud → **Settings → Connections → GitHub → Configure** 클릭
2. `agai-main` 레포를 허용 목록에 추가

---

## 기술 스택

- **LLM** — Google Gemini 2.5 Flash
- **임베딩** — `models/gemini-embedding-001` (768차원)
- **벡터 DB** — FAISS
- **프레임워크** — LangChain + Streamlit
- **PDF 처리** — PyPDF

---

## 프로젝트 구조

```
agai-main/
├── rag_chatbot.py       # 메인 앱 (RAG + Streamlit UI)
├── data/
│   ├── docs/            # 색인 대상 PDF 문서
│   └── ...              # 기타 실습 데이터
├── .env.example         # 환경변수 예시 (복사 후 .env로 사용)
├── requirements.txt     # 패키지 목록
└── README.md
```

---

## 사용 예시

```
Q: 환불 신청은 어떻게 하나요?
A: 환불 신청은 구매일로부터 7일 이내에 고객센터를 통해 ...
   📄 출처: 환불교환정책.pdf p.2

Q: 로봇청소기 필터 청소 주기는?
A: 헤파 필터는 2주에 한 번 청소를 권장하며 ...
   📄 출처: 제품매뉴얼_로봇청소기.pdf p.5
```
