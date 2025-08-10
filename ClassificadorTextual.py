import requests
import os
from transformers import pipeline
token2 = os.getenv('TOKEN_HF')

modelo = "nlptown/bert-base-multilingual-uncased-sentiment"

API_URL = f"https://router.huggingface.co/hf-inference/models/{modelo}"

headers = {
    "Authorization": f"Bearer {token2}",
}

reviews = [
    "Até então não tenho do que reclamar. Estou usando pra estudo, está bem tranquilo até aqui.",
    "Acho que vale o custo benefício, caso seja para usos básicos.",
    "A bateria do notebook está descarregando muito rápido.",
    "Eu estava com muito medo de me arrepender da compra. Mas eu realmente gostei! Ótimo demais, comprem!",
    "Muito bom, recomendo!",
    "Não comprem, caro demais pelo que oferece.",
    "Super custo benefício, pelo preço que paguei superou todas as minhas expectativas.",
    "Excelente, zero arrependimentos. Muito muito bom.",
    "Esperava um pouco mais. Mas é um produto bom. Não coloquei mais estrelas pois não usei direito.",
]

classificacoes = pipeline(task='text-classification', model=modelo, top_k=None)

for review in reviews:
    print('Avaliação: ', review)
    resposta = classificacoes(review)
    prop = round(resposta[0][0]['score'] * 100, 2)   
    print('Nível de sentimento: ', resposta[0][0]['label'], '(', prop, '%)')
    print('-'*100)
