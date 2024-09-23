import boto3   ##Biblioteca AWS SDK para Python, usada para interagir com os serviços da AWS.
import json    ##Biblioteca padrão do Python para manipulação de dados JSON.

# Dados do prompt que serão enviados ao modelo para geração de texto
prompt_data = """
Act as a Shakespeare and write a poem on Generative AI
"""

# Cria um cliente para o serviço "bedrock-runtime" da AWS usando o boto3
bedrock = boto3.client(service_name="bedrock-runtime")

# Dicionário contendo os parâmetros para a geração de texto, incluindo o prompt, comprimento máximo do texto gerado, temperatura e top-p.
payload = {
    "prompt": "[INST]" + prompt_data + "[/INST]",  # Prompt formatado para o modelo
    "max_gen_len": 512,  # Comprimento máximo do texto gerado
    "temperature": 0.5,  # Parâmetro de temperatura para controlar a aleatoriedade da geração
    "top_p": 0.9  # Parâmetro de top-p para controle de amostragem
}

# Converte o payload para uma string JSON
body = json.dumps(payload)

# ID do modelo que será invocado
model_id = "meta.llama3-8b-instruct-v1:0"

# Chama o método invoke_model do cliente bedrock para enviar o payload ao modelo e obter a resposta.
response = bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json"
)

# Lê o corpo da resposta e converte de volta para um dicionário Python.
response_body = json.loads(response.get("body").read())

# Extrai o texto gerado da resposta
response_text = response_body['generation']

# Imprime o texto gerado
print(response_text)


# Este código utiliza o serviço AWS Bedrock para invocar um modelo de linguagem natural e gerar um texto baseado no 
# prompt fornecido. Ele é projetado para pedir ao modelo que escreva um poema no estilo de Shakespeare sobre IA Generativa. 
# O código configura os parâmetros de geração, envia o pedido ao modelo, processa a resposta e imprime o texto gerado.