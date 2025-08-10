from transformers import pipeline

tradutor = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-pt")

print(tradutor("Hello world!"))