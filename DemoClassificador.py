"""
Script para demonstrar o uso do modelo treinado de classificação de sentimentos
Execute após o treinamento do modelo estar completo
"""

from TreinamentoModelo.ClassificadorSentimentos import ClassificadorSentimentos

def main():
    print("🎯 Demonstração do Classificador de Sentimentos")
    print("=" * 50)
    
    # Criar classificador
    classificador = ClassificadorSentimentos()
    
    if not classificador.esta_carregado():
        print("\n❌ ERRO: Modelo não encontrado!")
        print("📋 Para usar este script, você precisa:")
        print("   1. Executar o treinamento: python TreinamentoModelo/ModeloClassificacaoTextual.py")
        print("   2. Aguardar o modelo ser salvo em './TreinamentoModelo/fine_tuned_model/'")
        print("   3. Executar novamente este script")
        return
    
    print("\n✅ Modelo carregado com sucesso!")
    
    # Exemplos de classificação
    exemplos = [
        "Este produto é incrível! Superou todas as expectativas!",
        "Péssimo atendimento, não recomendo para ninguém.",
        "O produto é razoável, nem bom nem ruim.",
        "Entrega rápida e produto de qualidade excelente!",
        "Tive alguns problemas, mas no geral foi ok.",
        "Terrível! Dinheiro jogado fora, produto veio quebrado.",
        "Bom custo-benefício, atende às necessidades básicas."
    ]
    
    print("\n📊 Testando com exemplos predefinidos:")
    print("-" * 50)
    
    for i, texto in enumerate(exemplos, 1):
        resultado = classificador.classificar(texto)
        
        print(f"\n{i}. Texto: {texto}")
        print(f"   🏷️  Sentimento: {resultado['sentimento'].upper()}")
        print(f"   📈 Confiança: {resultado['confianca']}%")
        
        # Mostrar distribuição de todas as pontuações
        print("   📋 Distribuição completa:")
        for pontuacao in sorted(resultado['todas_pontuacoes'], 
                               key=lambda x: x['confianca'], reverse=True):
            emoji = "🥇" if pontuacao['sentimento'] == resultado['sentimento'] else "  "
            print(f"      {emoji} {pontuacao['sentimento']}: {pontuacao['confianca']}%")
    
    # Modo interativo
    print("\n" + "=" * 50)
    print("🎮 MODO INTERATIVO")
    print("Digite textos para classificar (ou 'sair' para terminar)")
    print("-" * 50)
    
    while True:
        try:
            texto = input("\n💬 Digite um texto: ").strip()
            
            if texto.lower() in ['sair', 'exit', 'quit', '']:
                print("\n👋 Obrigado por usar o classificador!")
                break
            
            if len(texto) < 3:
                print("❌ Texto muito curto. Digite algo mais substantivo.")
                continue
            
            resultado = classificador.classificar(texto)
            
            print(f"\n📝 Texto analisado: {texto}")
            print(f"🏷️  Sentimento: {resultado['sentimento'].upper()}")
            print(f"📈 Confiança: {resultado['confianca']}%")
            
            # Interpretação da confiança
            if resultado['confianca'] >= 80:
                interpretacao = "muito alta 🎯"
            elif resultado['confianca'] >= 60:
                interpretacao = "alta ✅"
            elif resultado['confianca'] >= 40:
                interpretacao = "moderada ⚖️"
            else:
                interpretacao = "baixa ⚠️"
            
            print(f"🔍 Interpretação: Confiança {interpretacao}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Saindo...")
            break
        except Exception as e:
            print(f"❌ Erro: {e}")

if __name__ == "__main__":
    main()
