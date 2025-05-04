import streamlit as st
from PIL import Image
import io
import re
import json
import csv
import base64
from io import BytesIO
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
from google.cloud import vision
import os

# Streamlit 설정
st.set_page_config(page_title="신장내과 진단 시스템", layout="centered")
st.title("🧜️ 신장내과 지방 예측 시스템")

st.markdown("""
이 시스템은 신장내과 지방군(예: CKD, AKI, Nephrotic Syndrome 등)에 대해
협악검사 수치를 기반으로 지방을 보조하고,
지방에 관한 질문에 구격 기반 응답을 제공하는 **AI 기반 지방 지원 도구**입니다.

**기능 소개:**
1. 수치 직접 입력 또는 검사 이미지 업로드(Google OCR 기반)
2. 뢰 기반 지방 및 리포트 저장
3. RAG 기반 질의응답 (유사 문서 및 AI 설명)
""")

st.header("🦢 검사 수치 입력")

with st.form("manual_input_form"):
    eGFR = st.number_input("eGFR (mL/min/1.73m²)", min_value=0.0, max_value=200.0, step=0.1)
    creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.0, max_value=20.0, step=0.1)
    albumin = st.number_input("Albumin (g/dL)", min_value=0.0, max_value=6.0, step=0.1)
    proteinuria = st.number_input("단백조 (mg/dL)", min_value=0.0, max_value=5000.0, step=1.0)
    submitted = st.form_submit_button("지방 결과 확인")

def run_rag_from_input(eGFR, creatinine, albumin, proteinuria):
    user_question = f"eGFR: {eGFR}, Creatinine: {creatinine}, Albumin: {albumin}, 단백조: {proteinuria} 이 수치로 어느 지방이 의심됩니까?"
    st.subheader("🔍 RAG 기반 결과 분석")
    st.write(f"\2753 자동 생성 질의: {user_question}")

    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
    question_embedding = model.encode([user_question])
    index = faiss.read_index("nephro_faiss.index")

    with open("nephro_chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    D, I = index.search(question_embedding, k=2)
    matched_docs = []
    for idx, score in zip(I[0], D[0]):
        matched_docs.append({
            "title": f"관련 문서 #{idx+1}",
            "similarity": round(1.0 / (1.0 + score), 2),
            "snippet": chunks[idx][:200] + "..." if len(chunks[idx]) > 200 else chunks[idx]
        })

    st.markdown("### 🔗 검사된 문서 및 유사도")
    for doc in matched_docs:
        st.write(f"📄 **{doc['title']}** — 유사도: {int(doc['similarity'] * 100)}%")
        st.caption(f"➤️ {doc['snippet']}")

    llm_answer = "(예시 응답) 입력된 수치는 CKD, Nephrotic Syndrome 등과 관련이 있을 수 있습니다. 정확한 진단은 의료진 상담이 필요합니다."
    st.markdown("### 🤖 AI 응답 예시")
    st.success(llm_answer)

if submitted:
    st.subheader("🩺 예비 지방 결과")
    result = []
    if eGFR < 60:
        result.append("🔸 CKD 가능성 있음 (eGFR < 60)")
    if creatinine > 1.3:
        result.append("🔸 신기능 저하 가능성 (Creatinine > 1.3)")
    if albumin < 3.5:
        result.append("🔸 저알부민형증 가능성")
    if proteinuria > 150:
        result.append("🔸 단백조 의심 (정상 기준 초과)")
    if result:
        for r in result:
            st.write(r)
    else:
        st.success("정상 범위로 판단됩니다.")

    run_rag_from_input(eGFR, creatinine, albumin, proteinuria)

st.markdown("---")

st.subheader("📷 이미지 OCR 업로드 (Google Vision API)")
uploaded_file = st.file_uploader("이미지 파일 업로드 (png, jpg, jpeg)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    images = [image]

    extracted_text = ""
    client = vision.ImageAnnotatorClient()
    for img in images:
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        content = img_byte_arr.getvalue()
        image_for_vision = vision.Image(content=content)
        response = client.text_detection(image=image_for_vision)
        texts = response.text_annotations
        if texts:
            extracted_text += texts[0].description + "\n"

    st.text_area("📁 OCR 인식 결과:", extracted_text, height=200)

    def extract_value(name):
        match = re.search(fr"{name}[:=\s]*([0-9]+\.?[0-9]*)", extracted_text, re.IGNORECASE)
        return float(match.group(1)) if match else 0.0

    eGFR_val = extract_value("eGFR")
    creat_val = extract_value("Creatinine")
    alb_val = extract_value("Albumin")
    prot_val = extract_value("단백조|Proteinuria")

    st.write("**📌 추출된 검사 수치:**")
    st.write(f"eGFR: {eGFR_val}, Creatinine: {creat_val}, Albumin: {alb_val}, 단백조: {prot_val}")

    run_rag_from_input(eGFR_val, creat_val, alb_val, prot_val)

st.markdown("---")

st.subheader("🤖 질의응답 시스템 (RAG 기반)")
user_question = st.text_input("예: 'eGFR이 55인데 CKD인가요?' 또는 '단백뇨 수치가 높으면 어떤 의미인가요?'")

if user_question:
    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
    question_embedding = model.encode([user_question])
    index = faiss.read_index("nephro_faiss.index")

    with open("nephro_chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    D, I = index.search(question_embedding, k=2)
    matched_docs = []
    for idx, score in zip(I[0], D[0]):
        matched_docs.append({
            "title": f"관련 문서 #{idx+1}",
            "similarity": round(1.0 / (1.0 + score), 2),
            "snippet": chunks[idx][:200] + "..." if len(chunks[idx]) > 200 else chunks[idx]
        })

    st.markdown("### 🔗 검사된 문서 및 유사도")
    for doc in matched_docs:
        st.write(f"📄 **{doc['title']}** — 유사도: {int(doc['similarity'] * 100)}%")
        st.caption(f"➤️ {doc['snippet']}")

    llm_answer = "(예시 응답) 입력하신 검사 수치는 CKD와 관련이 있을 수 있습니다. 보다 정확한 진단을 위해 의료진의 평가가 필요합니다."

    st.markdown("### 🤖 AI 응답 예시")
    st.success(llm_answer)

    if st.button("📄 질의응답 보고서 저장 (DOCX)"):
        doc = Document()
        doc.add_heading("RAG 기반 질의응답 결과 리포트", level=1)
        doc.add_paragraph(f"사용자 질문: {user_question}")

        doc.add_heading("검사된 문서", level=2)
        for doc_data in matched_docs:
            doc.add_paragraph(f"- {doc_data['title']} (유사도: {int(doc_data['similarity']*100)}%)")
            doc.add_paragraph(f"  ⤷ {doc_data['snippet']}")

        doc.add_heading("AI 응답", level=2)
        doc.add_paragraph(llm_answer)

        buffer = BytesIO()
        doc.save(buffer)
        st.download_button("DOCX 다운로드", data=buffer.getvalue(), file_name="질의응답_리포트.docx")
