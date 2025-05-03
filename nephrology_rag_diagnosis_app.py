import streamlit as st
from PIL import Image, ImageOps
import pytesseract
from pdf2image import convert_from_bytes
import io
import re
import json
import csv
import base64
from io import BytesIO
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss

# Tesseract 경로 지정 (Windows용)
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

st.set_page_config(page_title="신장내과 진단 시스템", layout="centered")
st.title("🔍 신장내과 질병 예측 시스템 (4단계: 질의응답)")

st.subheader("🧠 질병 관련 질문을 입력해 주세요")
user_question = st.text_input("예: 'eGFR이 55인데 CKD인가요?' 또는 '단백뇨 수치가 높으면 어떤 의미인가요?'")

if user_question:
    st.info("🔍 사용자의 질문을 벡터화하여 FAISS 인덱스에서 유사 문서를 검색합니다.")

    # 모델 및 인덱스 로드
    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
    question_embedding = model.encode([user_question])
    index = faiss.read_index("nephro_faiss.index")

    with open("nephro_chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # 검색 수행
    D, I = index.search(question_embedding, k=2)

    matched_docs = []
    for idx, score in zip(I[0], D[0]):
        matched_docs.append({
            "title": f"관련 문서 #{idx+1}",
            "similarity": round(1.0 / (1.0 + score), 2),  # L2 거리 → 유사도 근사치
            "snippet": chunks[idx][:200] + "..." if len(chunks[idx]) > 200 else chunks[idx]
        })

    st.markdown("### 🔗 검색된 문서 및 유사도")
    for doc in matched_docs:
        st.write(f"📄 **{doc['title']}** — 유사도: {int(doc['similarity'] * 100)}%")
        st.caption(f"➡️ {doc['snippet']}")

    # 모의 LLM 응답 예시
    llm_answer = "(예시 응답) 입력하신 검사 수치는 CKD와 관련이 있을 수 있습니다. 보다 정확한 진단을 위해 의료진의 평가가 필요합니다."

    st.markdown("### 🤖 AI 응답 예시")
    st.success(llm_answer)

    # 보고서 저장
    if st.button("📄 질의응답 보고서 저장 (DOCX)"):
        doc = Document()
        doc.add_heading("RAG 기반 질의응답 결과 리포트", level=1)
        doc.add_paragraph(f"사용자 질문: {user_question}")

        doc.add_heading("검색된 문서", level=2)
        for doc_data in matched_docs:
            doc.add_paragraph(f"- {doc_data['title']} (유사도: {int(doc_data['similarity']*100)}%)")
            doc.add_paragraph(f"  ⤷ {doc_data['snippet']}")

        doc.add_heading("AI 응답", level=2)
        doc.add_paragraph(llm_answer)

        buffer = BytesIO()
        doc.save(buffer)
        st.download_button("DOCX 다운로드", data=buffer.getvalue(), file_name="질의응답_리포트.docx")

    st.markdown("---")
    st.markdown("📌 현재는 로컬 임베딩 기반 검색이며, 향후 LLM 연동 시 응답 정확도가 향상됩니다.")

st.divider()
st.markdown("""
🔄 계속해서 OCR, 수치 입력, 진단 결과, 리포트 저장 등 기존 기능을 함께 활용하실 수 있습니다.
""")
