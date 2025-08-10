from transformers import pipeline
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import pathlib
from transformers import Trainer, TrainingArguments

load_dotenv(override=True)

# Caminho para o dataset
caminho = pathlib.Path(__file__).parent.parent.resolve()

# Carregar o conjunto de dados CSV
dataset = load_dataset('csv', data_files={
    'train': str(caminho) + '/Dataset/DatasetSentimento.csv',
    'test': str(caminho) + '/Dataset/DatasetSentimentoTeste.csv'
})

# Exibir os primeiros exemplos do conjunto de dados
print(dataset['train'][0])

# Mapeamento dos rótulos para valores numéricos
label_mapping = {"positivo": 0, "negativo": 1, "neutro": 2}

# Função para mapear os rótulos
def encode_labels(example):
    example['label'] = label_mapping[example['label']]
    return example

# Aplicar o mapeamento ao conjunto de dados
encoded_datasets = dataset.map(encode_labels)

# Carregar tokenizer e modelo
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-multilingual-cased", num_labels=3)  # Ajuste o número de rótulos

# Função para tokenizar os textos
def tokenize_function(examples):
    return tokenizer(examples['texto'], padding="max_length", truncation=True)

# Tokenizar o conjunto de dados
tokenized_datasets = encoded_datasets.map(tokenize_function, batched=True)

# Configurar argumentos de treinamento
training_args = TrainingArguments(
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
)

# Configurar o Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=tokenizer,
)

# Treinar o modelo
trainer.train()
trainer.save_model("./fine_tuned_model")
results = trainer.evaluate()
print(results)