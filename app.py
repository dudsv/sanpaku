import streamlit as st
import requests
import tempfile
import os
import base64
import cv2
import numpy as np
import mediapipe as mp

st.set_page_config(page_title="Verificação SANPAKU com IA (Together)", layout="centered")
st.title("🧠🎧👁️ SANPAKU com IA via Together AI")

# Substitua abaixo pela sua chave da Together AI
together_api_key = "tgp_v1_enAp8dJ6-Km1fipMg8laq7rwQSK2y3okvFcstaeQ4nM"

def transcrever_com_together(audio_path):
    url = "https://api.together.xyz/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {together_api_key}"}
    with open(audio_path, "rb") as f:
        files = {"file": f}
        data = {"model": "whisper-large-v3"}
        response = requests.post(url, headers=headers, files=files, data=data)
    if response.status_code != 200:
        st.error("Erro na transcrição do áudio:")
        st.code(response.text)
        return "Erro na transcrição"
    return response.json().get("text", "Sem texto retornado")

def gerar_relatorio_com_together(prompt):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {together_api_key}",
        "Content-Type": "application/json"
    }
    json_data = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1200
    }
    response = requests.post(url, headers=headers, json=json_data)
    res = response.json()
    if "choices" in res:
        return res["choices"][0]["message"]["content"]
    else:
        st.error("Erro na resposta da IA:")
        st.json(res)
        return "Erro na geração do relatório"

def detectar_olhar(image_file):
    mp_face_mesh = mp.solutions.face_mesh
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(rgb_image)
        if not results.multi_face_landmarks:
            return "Nenhum rosto detectado"
        landmarks = results.multi_face_landmarks[0].landmark
        top = landmarks[159].y
        bottom = landmarks[145].y
        iris = landmarks[468].y
        center = (top + bottom) / 2
        if iris < center - 0.01:
            return "SANPAKU Inferior"
        elif iris > center + 0.01:
            return "SANPAKU Superior"
        else:
            return "Olhar Normal"

# Formulário simples
st.subheader("🧠 MENTE")
perguntas = [
    "Você tem se sentido disperso?",
    "Costuma quebrar objetos?",
    "Tem dificuldades para relaxar?"
]
respostas = [f"- {q} {st.text_input(q)}" for q in perguntas]

# Foto
foto = st.file_uploader("📸 Envie a foto do rosto", type=["jpg", "jpeg", "png"])
resultado_olhar = detectar_olhar(foto) if foto else ""

# Áudio
st.markdown("📥 Envie o áudio gravado (.wav ou .mp3)")
audio = st.file_uploader("Áudio", type=["wav", "mp3"])
audio_path = None
if audio:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio.read())
        audio_path = tmp.name
        st.audio(audio_path)

if st.button("📄 Gerar Relatório"):
    if not foto or not audio_path:
        st.warning("Por favor, envie a imagem e o áudio.")
    else:
        texto = transcrever_com_together(audio_path)
        prompt = (
            "Você é uma IA especialista em interpretação simbólica no estilo SANPAKU.\n"
            "Baseie-se nas respostas abaixo para gerar uma leitura emocional.\n"
            "MENTE:\n" + "\n".join(respostas) + "\n\n"
            "CORES (voz):\nTranscrição: " + texto + "\n\n"
            "OLHAR:\n" + resultado_olhar + "\n\n"
            "Gere um relatório com perfil simbólico e recomendações personalizadas."
        )
        resultado = gerar_relatorio_com_together(prompt)
        st.success("Relatório gerado com sucesso!")
        st.markdown("### Relatório Final")
        st.markdown(resultado)