from transformers import pipeline
import os
from dotenv import load_dotenv

load_dotenv(override=True)

frases = [
    "Adorei o produto!",
    "Não gostei do atendimento.",
    "Produto ok, normal. Produto mediano, sem mais."
]
modelo = './fine_tuned_model'  # Caminho para o modelo treinado

# Mapeamento de rótulos
label_mapping = {"LABEL_0": "positivo", "LABEL_1": "negativo", "LABEL_2": "neutro"}

# Carregar modelo treinado
classificador = pipeline('text-classification', model=modelo)

# Classificar e mostrar resultados
for frase in frases:
    resultado = classificador(frase)
    label_original = resultado[0]['label']
    sentimento = label_mapping.get(label_original, label_original)  # Mapear para nome descritivo
    confianca = round(resultado[0]['score'] * 100, 2)
    print(f"'{frase}' → {sentimento} ({confianca}%)")