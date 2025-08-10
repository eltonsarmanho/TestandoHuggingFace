from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import os

class ClassificadorSentimentos:
    """
    Classe para classificação de sentimentos usando modelo treinado
    """
    
    def __init__(self, caminho_modelo="./fine_tuned_model"):
        """
        Inicializa o classificador carregando o modelo treinado
        """
        self.label_mapping = {0: "positivo", 1: "negativo", 2: "neutro"}
        self.caminho_modelo = caminho_modelo
        self.tokenizer = None
        self.model = None
        self.classificador = None
        
        self.carregar_modelo()
    
    def carregar_modelo(self):
        """
        Carrega o modelo e tokenizer treinados
        """
        try:
            if os.path.exists(self.caminho_modelo):
                self.tokenizer = AutoTokenizer.from_pretrained(self.caminho_modelo)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.caminho_modelo)
                
                # Criar pipeline
                self.classificador = pipeline(
                    "text-classification",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    return_all_scores=True
                )
                
                print(f"✅ Modelo carregado com sucesso!")
            else:
                print(f"❌ Modelo não encontrado em: {self.caminho_modelo}")
                print("Execute primeiro o treinamento do modelo.")
                
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
    
    def classificar(self, texto):
        """
        Classifica um texto e retorna o sentimento com maior confiança
        
        Args:
            texto (str): Texto a ser classificado
            
        Returns:
            dict: Dicionário com sentimento e confiança
        """
        if self.classificador is None:
            return {"erro": "Modelo não carregado"}
        
        try:
            resultado = self.classificador(texto)
            
            # Converter resultado para formato mais legível
            resultado_formatado = []
            for item in resultado[0]:
                label_id = int(item['label'].split('_')[-1])
                sentimento = self.label_mapping[label_id]
                confianca = item['score']
                resultado_formatado.append({
                    'sentimento': sentimento,
                    'confianca': round(confianca * 100, 2)
                })
            
            # Retornar o resultado com maior confiança
            melhor_resultado = max(resultado_formatado, key=lambda x: x['confianca'])
            
            return {
                'texto': texto,
                'sentimento': melhor_resultado['sentimento'],
                'confianca': melhor_resultado['confianca'],
                'todas_pontuacoes': resultado_formatado
            }
            
        except Exception as e:
            return {"erro": f"Erro na classificação: {e}"}
    
    def classificar_batch(self, textos):
        """
        Classifica uma lista de textos
        
        Args:
            textos (list): Lista de textos para classificar
            
        Returns:
            list: Lista com resultados da classificação
        """
        resultados = []
        for texto in textos:
            resultado = self.classificar(texto)
            resultados.append(resultado)
        return resultados
    
    def esta_carregado(self):
        """
        Verifica se o modelo foi carregado corretamente
        
        Returns:
            bool: True se modelo está carregado, False caso contrário
        """
        return self.classificador is not None

