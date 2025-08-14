import os
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import logging

def testar_modelo_simples():
    """Testa o modelo treinado"""
    model_path = "/home/nees/Documents/VSCodigo/TestandoHuggingFace/qa_model_treinado"
    try:
        # Carregar modelo
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        
        # Perguntas de teste
        perguntas = [
            "Quanto é 2 + 2?",
            "Se eu tenho 10 maçãs e como 3, quantas sobram?",
            "Qual é a área de um quadrado de lado 5?",
            "Encontre a soma de 20 números pares consecutivos a partir de 1."

        ]
        
        print("Respostas do modelo:")
        print("-" * 40)
        
        for pergunta in perguntas:
            # Preparar prompt
            prompt = f"{pergunta}{tokenizer.sep_token}"
            inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
            
            # Gerar resposta
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decodificar
            resposta_completa = tokenizer.decode(outputs[0], skip_special_tokens=True)
            resposta = resposta_completa.replace(pergunta, "").strip()
            
            print(f"P: {pergunta}")
            print(f"R: {resposta[:200]}...")  # Limitar tamanho
            print()
        
    except Exception as e:
        print(f"Erro no teste: {e}")

testar_modelo_simples()