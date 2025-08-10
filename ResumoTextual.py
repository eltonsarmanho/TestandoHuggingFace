from transformers import pipeline
import os
import dotenv
from dotenv import load_dotenv

load_dotenv(override=True)
input_text = "O corpo está mais propenso a sentir dores com exercícios de alta intensidade | Foto: Getty Images O problema está em saber identificar qual é qual. Em algumas situações, é difícil diferenciar uma da outra, reconhece Juan Francisco Marco, professor do Centro de Ciência do Esporte, Treinamento e Fitness Alto Rendimento, na Espanha. A dor boa é aquela que associamos ao exercício físico, que não limita (o movimento) e permite continuar (a se exercitar) até o momento em que o músculo fica realmente esgotado e não trabalha mais, explica. É importante detectar qual é o tipo de dor que você está sentindo, para evitar ter problemas mais sérios | Foto: Getty Images Para Francisco Sánchez Diego, diretor do centro de treinamento Corpore 10, a dor boa se sente no grupo muscular que você trabalhou, tanto durante o treinamento como nos dias seguintes."

classifier = pipeline("summarization",token=os.getenv("HF_TOKEN"),
                      model='t5-small')
resposta  = classifier(input_text)
print(resposta[0]['summary_text'])