"""
Versão simplificada do treinamento de modelo QA
Compatível com versões recentes do transformers
"""

import os
import torch
import pandas as pd
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def treinar_modelo_qa_simples():
    """Função simplificada para treinar modelo QA"""
    
    print("=" * 60)
    print("TREINAMENTO SIMPLIFICADO - MODELO QA")
    print("=" * 60)
    
    # Configurações
    model_name = "pucpr/bioBERTpt-squad-v1.1-portuguese"
    model_name = "microsoft/DialoGPT-small"
    max_length = 256
    output_dir = "/home/nees/Documents/VSCodigo/TestandoHuggingFace/qa_model_treinado"
    
    try:
        # 1. Carregar dataset
        print("\n1. Carregando dataset...")
        df = pd.read_parquet("hf://datasets/rhaymison/orca-math-portuguese-64k/data/train-00000-of-00001.parquet")
        
        print(df.info())
        
        # Usar apenas uma pequena amostra para demonstração
        sample_size = 1000
        
        # Selecionar amostras usando pandas
        train_df = df.iloc[:sample_size].copy()
        val_df = df.iloc[sample_size:sample_size + 100].copy()
        
        # Converter para Dataset
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        
        dataset_small = DatasetDict({
            'train': train_dataset,
            'test': val_dataset
        })
        
        print(f"Dataset carregado: {len(dataset_small['train'])} treino, {len(dataset_small['test'])} test")
        
        # 2. Carregar modelo e tokenizer
        print("\n2. Carregando modelo e tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Adicionar tokens especiais
        special_tokens = {"pad_token": "<pad>", "sep_token": "<sep>"}
        tokenizer.add_special_tokens(special_tokens)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Usar FP32 para estabilidade
            low_cpu_mem_usage=True
        )
        
        # Redimensionar embeddings
        model.resize_token_embeddings(len(tokenizer))
        
        print("Modelo e tokenizer carregados com sucesso!")
        
        # 3. Preprocessar dados
        print("\n3. Preprocessando dados...")
        
        def preprocess_function(examples):
            # Combinar pergunta e resposta
            texts = []
            for question, answer in zip(examples['question'], examples['answer']):
                text = f"{question}{tokenizer.sep_token}{answer}{tokenizer.eos_token}"
                texts.append(text)
            
            # Tokenizar com padding e truncation habilitados
            result = tokenizer(
                texts,
                truncation=True,
                padding=True,  # Habilitar padding
                max_length=max_length,
                return_tensors=None
            )
            
            # Labels = input_ids para language modeling
            result["labels"] = result["input_ids"].copy()
            return result
        
        # Aplicar preprocessamento
        tokenized_datasets = dataset_small.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset_small['train'].column_names,
            desc="Tokenizando"
        )
        
        print("Dados preprocessados!")
        
        # 4. Configurar treinamento
        print("\n4. Configurando treinamento...")
        
        # Argumentos mais simples e estáveis
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=1,  # Apenas 1 época para demo
            per_device_train_batch_size=1,  # Reduzir batch size
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,  # Compensar batch size menor
            warmup_steps=20,
            max_steps=50,  # Reduzir steps para demo mais rápida
            logging_steps=5,
            eval_steps=25,
            save_steps=25,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=False,  # Desabilitar para simplificar
            fp16=False,  # Desabilitar FP16 para evitar problemas
            dataloader_drop_last=True,
            report_to=[],  # Sem logging externo
            remove_unused_columns=False,
            logging_dir=None  # Sem logging detalhado
        )
        
        # Data collator simplificado
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # 5. Criar trainer
        print("\n5. Criando trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['test'],
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        
        # 6. Treinar
        print("\n6. Iniciando treinamento...")
        print("   (Treinamento rápido - apenas para demonstração)")
        
        trainer.train()
        
        # 7. Salvar modelo
        print("\n7. Salvando modelo...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        print(f"\n✅ TREINAMENTO CONCLUÍDO!")
        print(f"Modelo salvo em: {output_dir}")
        
       
        
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    treinar_modelo_qa_simples()