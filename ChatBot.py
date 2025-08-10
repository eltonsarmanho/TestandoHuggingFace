from transformers import pipeline, AutoTokenizer
import sys

# Define o nome do modelo
model_name = "Felladrin/Llama-68M-Chat-v1"

print(f"Loading model: {model_name}...")
try:
    # Carrega o tokenizador para formatar o prompt corretamente
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Inicializa o pipeline de geração de texto
    # Adicionamos parâmetros para melhorar a qualidade da geração e evitar repetição
    chatbot = pipeline(
        task='text-generation',
        model=model_name,
        tokenizer=tokenizer,
        max_new_tokens=256,  # Controla o máximo de tokens novos, não o total
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15
    )
    print("Model loaded successfully!")

    # Mensagem do sistema
    mensagem_sistema = "You are a helpful and concise AI assistant."

    # Prepara a lista de mensagens no formato que o tokenizador espera
    conversa = [
        {"role": "system", "content": mensagem_sistema}
    ]

    while True:
        pergunta = input("Write your question: ")
        if pergunta.lower() in ['exit', 'quit', 'sair']:
            print("Exiting chat. Goodbye!")
            break

        # Adiciona a pergunta do usuário ao histórico
        conversa.append({"role": "user", "content": pergunta})

        # Usa o método do tokenizador para aplicar o template de chat
        # Isso cria a string formatada corretamente com <|im_start|>, etc.
        prompt_formatado = tokenizer.apply_chat_template(conversa, tokenize=False, add_generation_prompt=True)

        # Gera a resposta
        # Passamos o prompt já formatado para o pipeline
        resposta_completa = chatbot(prompt_formatado)

        # A resposta gerada está dentro do dicionário
        texto_gerado = resposta_completa[0]["generated_text"]
        
        # Extrai APENAS a nova resposta do assistente
        # A resposta começa depois do prompt que enviamos
        nova_resposta = texto_gerado[len(prompt_formatado):]

        print(f"Assistant: {nova_resposta}")

        # Adiciona a resposta do assistente ao histórico para a próxima rodada
        conversa.append({"role": "assistant", "content": nova_resposta})


except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure you have the 'transformers' library installed and the model is available.")
    print("You can install the library with: pip install transformers torch")
    sys.exit(1)