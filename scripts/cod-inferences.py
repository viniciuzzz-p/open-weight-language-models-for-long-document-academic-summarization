from openai import OpenAI
import time
import pandas as pd
import re
import json
import os
import csv

llms = [
    "gemma-3-4b-it", 
    "gemma-3-12b-it",                 		
    "gemma-3-27b-it",                 		
    "meta-llama-3.1-8B-Instruct",
    "llama-3.2-1B-Instruct",  
    "llama-3.2-3B-Instruct",    
    "llama-3.3-70B-Instruct",
    "DeepSeek-R1-Distill-Llama-8B", 
    "DeepSeek-R1-Distill-Llama-70B",  		
    "DeepSeek-R1-Distill-Qwen-1.5B",  
    "DeepSeek-R1-Distill-Qwen-7B",
    "DeepSeek-R1-Distill-Qwen-14B", 
    "DeepSeek-R1-Distill-Qwen-32B",   	
    "mistral-small-3.2-24B-Instruct-2506",
    "Phi-4-mini-instruct",
    "qwen/qwen3-4b-2507",
    "qwen/qwen3-30b-a3b-2507"
]

client = OpenAI(base_url="http://192.168.68.113:1234/v1", api_key="lm-studio", timeout=1800.0)

# --- CONFIGURAÇÃO DE CAMINHOS ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
prompt = ["prompt_cod"]

for tp_prompt in prompt:    
    dataset_name = "teses1" 
    
    # Caminhos
    csv_path = os.path.join(project_root, "datasets")
    dataset_base = f"{dataset_name}.csv"
    csv_output_path = os.path.join(project_root, "output_csv")
    os.makedirs(csv_output_path, exist_ok=True) 
    csv_prompt = f"{dataset_name}_{tp_prompt}.csv"

    # Carrega DF
    try:
        full_path = os.path.join(csv_path, dataset_base)
        print(f"Lendo dataset: {full_path}")
        df = pd.read_csv(full_path)
    except FileNotFoundError:
        print("ERRO: Dataset não encontrado.")
        continue

    # Carrega DF (Output/Progresso)
    if os.path.exists(os.path.join(csv_output_path, csv_prompt)):
        print(f"Carregando progresso anterior...")
        df_out = pd.read_csv(os.path.join(csv_output_path, csv_prompt))
        df = df_out
    
    # Carrega o Template do Prompt
    prompt_path = os.path.join(project_root, "prompts", f"{tp_prompt}.txt")
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            base_prompt_template = f.read()
    except:
        print(f"ERRO: Arquivo {prompt_path} não encontrado.")
        break

    for llm_name in llms:
        print(f"\n--- Processando com: {llm_name} ---")
        
        # Inicializa colunas
        if llm_name not in df.columns:
            df[llm_name] = ""                
            df[llm_name+"_response"] = ""    
            df[llm_name+"_time"] = 0.0       
            df[llm_name+"_n_tokens_in"] = 0  
            df[llm_name+"_n_tokens_out"] = 0 

        for index, row in df.iterrows():
            # Pula se já tiver resposta válida
            if pd.notna(df.loc[index, llm_name+"_response"]) and df.loc[index, llm_name+"_response"] != "":
                continue

            # --- 1. PREPARAÇÃO DO TEXTO (MODO HARDWARE MAXIMO) ---
            text_input = str(row['text']).replace('\x00', '')
            
            # --- 2. INJEÇÃO NO PROMPT ---
            # Substitui a tag pelo texto massivo
            final_content = base_prompt_template.replace("{{TEXT}}", text_input)

            messages = [{"role": "user", "content": final_content}]

            # --- 3. CONFIGURAÇÃO DE PARADA (STOP TOKENS) ---
            stop_map = {
                "gemma": ["<end_of_turn>", "<eos>"],
                "llama": ["<|eot_id|>", "<|end_of_text|>", "</s>"]
            }
            
            current_stops = None
            for k, v in stop_map.items():
                if k in llm_name.lower():
                    current_stops = v
                    break
            
            # --- 4. INFERÊNCIA ---
            max_retries = 3
            completion = None
            start_time = time.time()

            for attempt in range(max_retries):
                try:
                    print(f"Gerando Linha {index}... (Input: ~{len(text_input)//4} tokens)")
                    completion = client.chat.completions.create(
                        model=llm_name,
                        messages=messages,
                        temperature=0.3, 
                        max_completion_tokens=8192, # Alto para caber os 5 passos do resumo
                        stop=current_stops,
                    )
                    break
                except Exception as e:
                    print(f"Erro tentativa {attempt+1}: {e}")
                    time.sleep(5)

            # --- 5. SALVAMENTO ---
            if completion:
                duration = time.time() - start_time
                raw_response = completion.choices[0].message.content

                df.loc[index, llm_name+"_response"] = raw_response
                df.loc[index, llm_name+"_time"] = duration
                df.loc[index, llm_name+"_n_tokens_in"] = completion.usage.prompt_tokens
                df.loc[index, llm_name+"_n_tokens_out"] = completion.usage.completion_tokens
                
                # Save parcial a cada linha
                df.to_csv(os.path.join(csv_output_path, csv_prompt), index=False, quoting=csv.QUOTE_ALL)
            else:
                print(f"Falha total na linha {index}")