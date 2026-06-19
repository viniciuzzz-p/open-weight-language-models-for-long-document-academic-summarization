import time
import pandas as pd
import os
import csv
import google.auth
import google.auth.transport.requests
from google.oauth2 import service_account
import requests

# --- CONFIGURAÇÃO ---
MODEL        = "gemini-2.5-pro"
SERVICE_ACCOUNT_FILE = "/home/vinicius/Área de Trabalho/Artigos/GCP/sefaz-ai-poc-funcap.json"  # ajuste o nome se necessário

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

dataset_path = os.path.join(project_root, "datasets", "teses1_prompt_fixo.csv") 
prompt_path  = os.path.join(project_root, "prompts", "prompt_fixo.txt")
output_path  = os.path.join(project_root, "output_csv", "novas_teste_gemini25pro.csv")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# --- AUTENTICAÇÃO VIA SERVICE ACCOUNT ---
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)

def get_access_token():
    """Renova o token se necessário e retorna o access token."""
    request = google.auth.transport.requests.Request()
    credentials.refresh(request)
    return credentials.token

# Pega project_id direto do JSON
import json
with open(SERVICE_ACCOUNT_FILE) as f:
    sa_info = json.load(f)
PROJECT_ID = sa_info["project_id"]

API_URL = (
    f"https://us-central1-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}"
    f"/locations/us-central1/publishers/google/models/{MODEL}:generateContent"
)

def call_gemini(prompt_text: str) -> str:
    token = get_access_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 8192,
        },
    }
    resp = requests.post(API_URL, headers=headers, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]

# --- CARREGA DATASET ---
df = pd.read_csv(dataset_path)

col = f"{MODEL}_response"
if os.path.exists(output_path):
    print("Carregando progresso anterior...")
    df_out = pd.read_csv(output_path)
    df[col] = df_out[col] if col in df_out.columns else ""
else:
    df[col] = ""

# --- CARREGA PROMPT ---
with open(prompt_path, "r", encoding="utf-8") as f:
    prompt_template = f.read()

print(f"Dataset: {len(df)} linhas | Modelo: {MODEL}\n")

# --- LOOP PRINCIPAL ---
for index, row in df.iterrows():
    if pd.notna(df.loc[index, col]) and df.loc[index, col] != "":
        print(f"[{index}] já processado, pulando.")
        continue

    text_input    = str(row["text"]).replace("\x00", "")
    final_content = prompt_template.replace("{{TEXT}}", text_input)

    for attempt in range(3):
        try:
            print(f"[{index}] Gerando...")
            df.loc[index, col] = call_gemini(final_content)
            break
        except Exception as e:
            print(f"  Erro tentativa {attempt+1}: {e}")
            time.sleep(5)

    df[["title", col]].to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"[{index}] ✔ salvo.")

print(f"\n✅ Concluído! Arquivo salvo em: {output_path}")