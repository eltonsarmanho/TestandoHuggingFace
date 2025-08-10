"""
Script para demonstrar o uso do modelo treinado de classificação de sentimentos
Execute após o treinamento do modelo estar completo
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import os

from ClassificadorSentimentos import ClassificadorSentimentos
# Exemplo de uso
def run_demo():
    """
    Demonstra como usar a classe ClassificadorSentimentos
    """
    print("🚀 Exemplo de uso do Classificador de Sentimentos\n")
    
    # Criar instância do classificador
    classificador = ClassificadorSentimentos()
    
    if not classificador.esta_carregado():
        print("❌ Modelo não pôde ser carregado. Execute o treinamento primeiro.")
        return
    
    # Textos de exemplo
    textos_exemplo = [
        "Adorei este produto! É exatamente o que eu esperava.",
        "Terrível! Não funciona e o atendimento é péssimo.",
        "O produto é ok, nada demais mas serve para o que preciso.",
        "Fantástico! Recomendo para todos os meus amigos.",
        "Não gostei, mas também não foi tão ruim assim."
    ]
    
    print("📊 Classificando textos de exemplo:\n")
    
    for i, texto in enumerate(textos_exemplo, 1):
        resultado = classificador.classificar(texto)
        
        if "erro" in resultado:
            print(f"❌ Erro: {resultado['erro']}")
            continue
        
        print(f"Texto {i}: {texto}")
        print(f"Sentimento: {resultado['sentimento']}")
        print(f"Confiança: {resultado['confianca']}%")
        print("-" * 60)

if __name__ == "__main__":
    run_demo()
