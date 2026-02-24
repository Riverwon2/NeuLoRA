# NeuLoRA 🧠✨
**The Thinking Tutor that Connects to Your Neuron**

LangGraph 기반의 멀티턴 RAG 챗봇 프로젝트입니다.  
PDF/TXT 문서를 적재하고, 대화 맥락 + 벡터 검색 + (필요 시) 웹 검색을 조합해 답변을 생성합니다.

---

## 🚀 Key Feature

- 📚 **문서 기반 질의응답(RAG)**: ChromaDB + LangChain/LangGraph 기반 검색-생성 파이프라인
- 🧭 **질문 라우팅**: 질문 특성에 따라 검색 필요 여부를 그래프에서 분기
- 🧪 **관련성 점검**: 검색 문맥이 부적절할 경우 웹 검색(Tavily)로 보강
- 🧠 **대화 기억 저장**: 대화 누적 시 요약 메모리를 별도 컬렉션에 저장
- 💬 **웹 챗 UI**: FastAPI + React(Vite) 기반 실시간 질의응답 화면
- 🧮 **수식 렌더링 지원**: 프론트에서 KaTeX 기반 LaTeX 표시

---

## 🗂️ Project Structure

```text
YAI-NLP/
├─ LangGraph/
│  ├─ LangGraph.py                 # 메인 그래프 파이프라인 (노드/분기/실행)
│  ├─ api.py                       # FastAPI 서버 (chat/upload/status/documents/reset)
│  ├─ stream.py                    # Streamlit 데모 엔트리
│  ├─ LangGraph.ipynb              # 실험/프로토타이핑 노트북
│  ├─ chroma_db/                   # Chroma 로컬 영속 저장소
│  └─ frontend/                    # React + Vite 프론트엔드
│     ├─ src/
│     │  ├─ App.jsx                # 앱 상태/요청/스트리밍 UI 로직
│     │  ├─ App.css                # 전체 스타일
│     │  ├─ main.jsx               # 앱 부트스트랩 (KaTeX CSS 포함)
│     │  └─ components/
│     │     ├─ ChatArea.jsx        # 메시지 렌더/입력/업로드 UI
│     │     ├─ Sidebar.jsx         # 상태/문서/리셋 패널
│     │     └─ Toast.jsx           # 토스트 컴포넌트
│     ├─ package.json
│     └─ vite.config.js
├─ rag/
│  ├─ base.py                      # 공통 체인/임베딩/프롬프트 구성
│  ├─ chroma.py                    # Chroma 연결형 RetrievalChain
│  ├─ ingest.py                    # PDF/TXT 적재 파이프라인
│  ├─ pdf.py                       # PDF 처리 유틸
│  ├─ utils.py                     # 문서 포맷팅 유틸
│  └─ graph_utils.py               # 그래프 실행/세션 도우미
├─ requirements.txt
├─ setup.sh
└─ README.md
```

---

## ⚙️ Technical Stack

- **Backend**: FastAPI, LangGraph, LangChain
- **Frontend**: React, Vite
- **Vector DB**: ChromaDB
- **LLM/Embedding**: Hugging Face Endpoint + `Qwen/Qwen2.5-7B-Instruct`, `BAAI/bge-m3`
- **양자화**: bitsandbytes (4bit NF4 / 8bit)
- **Math Rendering**: KaTeX (`react-markdown` + `remark-math` + `rehype-katex`)

---

## 🔐 Environmental Variables

루트의 `.env` 파일에 아래 값을 설정하세요.

| 변수 | 필수 | 설명 |
|---|---|---|
| `HF_API_KEY` | ✅ (API 모드 시) | Hugging Face API 토큰 (LLM/임베딩 호출) |
| `TAVILY_API_KEY` | 선택 | 웹 검색 보강 기능 사용 시 필요 |
| `EMBEDDING_MODE` | 선택 | `local`(기본) / `api` |
| `LLM_MODE` | 선택 | `api`(기본) / `vessel`(로컬 GPU, 3090 등) |
| `LLM_8BIT` | 선택 | vessel 시 `1` 또는 `true` 이면 8bit 양자화 (24GB VRAM 권장) |

예시:

```env
HF_API_KEY=hf_xxxxxxxxxxxx
TAVILY_API_KEY=tvly-xxxxxxxxxxxx
EMBEDDING_MODE=local
# vessel(로컬 GPU) 사용 시:
# LLM_MODE=vessel
# LLM_8BIT=1
```

---

## 🏃 Run

### 1) Python 의존성 설치

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 2) FastAPI 백엔드 실행

```bash
cd LangGraph
pip install fastapi uvicorn python-multipart
uvicorn api:app --reload --port 8800
```

### 3) React 프론트 실행

```bash
cd LangGraph/frontend
npm install
npm run dev
```

- Frontend: `http://localhost:5173`
- API: `http://localhost:8800`

### 4) Streamlit 데모 (선택)

```bash
cd LangGraph
streamlit run stream.py
```

---

## 🌐 Remote(SSH) 환경에서 실행하기

원격 서버에 SSH로 접속해 실행할 때도 로컬과 동일한 순서로 진행하면 됩니다.

### 가상환경

- **프로젝트 루트**에서 활성화합니다. `venv/bin` 디렉터리 안으로 들어갈 필요 없습니다.

```bash
cd /path/to/NeuLoRA
source venv/bin/activate
```

- 터미널을 열 때마다(또는 SSH 접속 시) 위 명령을 한 번 실행하면 해당 세션에서 가상환경이 적용됩니다.

### .env 파일

- `.env`는 Git에 포함되지 않으므로, **원격 서버에서 프로젝트를 새로 클론한 경우** 해당 서버 루트에 `.env`를 직접 만들고 `HF_API_KEY` 등 필요한 변수를 넣어야 합니다.
- 이미 해당 서버에 `.env`가 있다면 SSH 접속 여부와 관계없이 그대로 사용하면 됩니다.

### 원격에서 브라우저로 접속할 때

- 서버가 아닌 **내 PC 브라우저**로 접속하려면, 서버가 모든 인터페이스에서 listen 하도록 실행합니다.

**백엔드:**

```bash
cd LangGraph
uvicorn api:app --reload --host 0.0.0.0 --port 8800
```

**프론트엔드 (Vite):**  
`LangGraph/frontend`에서 `npm run dev` 시 `--host`를 주면 외부 접속이 가능합니다.

```bash
cd LangGraph/frontend
npm run dev -- --host
```

- 접속 주소: `http://<서버IP>:5173` (프론트), `http://<서버IP>:8800` (API)
- 방화벽에서 5173, 8800 포트가 열려 있어야 합니다. (예: `sudo ufw allow 5173,8800`)

---

## 🧩 API endpoints

| Method | Path | 설명 |
|---|---|---|
| `POST` | `/api/chat` | 질문 전송 후 답변 반환 (`message`, 선택 `thread_id`) |
| `POST` | `/api/upload` | PDF/TXT 업로드 및 벡터 DB 적재 |
| `GET` | `/api/status` | 모델/연결 상태 조회 |
| `GET` | `/api/documents` | 컬렉션 목록 및 문서 개수 |
| `POST` | `/api/reset` | 새 세션 ID 발급 |

---

## 🤖 Current model setting

- **라우팅/판단/요약**: `Qwen/Qwen2.5-14B-Instruct`
- **답변 생성(RAG)**: `Qwen/Qwen2.5-14B-Instruct`
- **임베딩**: `BAAI/bge-m3`

---

## 🧪 LoRA feature

> ⚠️ **MultiLoRA를 이용한 맞춤형 질의응답 튜터**  
> 데이터셋 구성, 학습 전략, 어댑터 병합 여부는 추후 확정 예정입니다.

---

## 📌 참고 사항

- `.env` 파일은 Git에 포함하지 않습니다.
- ChromaDB 데이터는 `LangGraph/chroma_db/`에 로컬 저장됩니다.
- 실험용 파일/노트북은 운영 코드와 분리하여 관리하는 것을 권장합니다.
