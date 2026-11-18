from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver  # NOVO
from langgraph.types import Command, interrupt  # NOVO

load_dotenv()
llm = ChatOpenAI(model="gpt-5-mini")

# ---------- Ferramenta 1: encomendas ----------
ENCOMENDAS = {
    "TC-55821": {"produto": "Televisão 55 polegadas", "estado": "atrasada", "data_prevista": "14/11/2026", "transportadora": "CTT Expresso"},
    "TC-10234": {"produto": "Portátil Lenovo IdeaPad 5", "estado": "entregue", "data_prevista": "03/11/2026", "transportadora": "DPD"},
    "TC-60112": {"produto": "Auscultadores sem fios", "estado": "em trânsito", "data_prevista": "12/11/2026", "transportadora": "CTT Expresso"},
    "TC-60450": {"produto": "Máquina de café", "estado": "em preparação", "data_prevista": "15/11/2026", "transportadora": "DPD"},
    "TC-59987": {"produto": "Monitor 27 polegadas", "estado": "cancelada", "data_prevista": "—", "transportadora": "—"},
}

@tool
def consultar_encomenda(numero_encomenda: str) -> str:
    """Consulta o estado de uma encomenda da TechCasa.

    Args:
        numero_encomenda: O número da encomenda, no formato TC-XXXXX.
    """
    encomenda = ENCOMENDAS.get(numero_encomenda.strip().upper())
    if encomenda is None:
        return f"A encomenda {numero_encomenda} não foi encontrada."
    return (
        f"Encomenda {numero_encomenda}: {encomenda['produto']}, estado: {encomenda['estado']}, "
        f"data prevista: {encomenda['data_prevista']}, transportadora: {encomenda['transportadora']}."
    )

# ---------- Ferramenta 2: políticas (RAG) ----------
texto_politicas = Path("politicas_techcasa.md").read_text(encoding="utf-8")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
vector_store = InMemoryVectorStore(OpenAIEmbeddings(model="text-embedding-3-small"))
vector_store.add_texts(splitter.split_text(texto_politicas))

@tool
def politicas_techcasa(pergunta: str) -> str:
    """Pesquisa nas políticas oficiais da TechCasa: garantia, devoluções e reembolsos,
    portes e prazos de entrega (incluindo Açores e Madeira), métodos de pagamento e
    prestações, apoio técnico básico e prazos de resposta do apoio ao cliente.

    Args:
        pergunta: O que procurar nas políticas, por exemplo "garantia de portáteis".
    """
    resultados = vector_store.similarity_search(pergunta, k=4)
    return "\n\n".join(doc.page_content for doc in resultados)

tools = [consultar_encomenda, politicas_techcasa]
llm_com_ferramentas = llm.bind_tools(tools)

# ---------- Classificação ----------
class Classificacao(BaseModel):
    categoria: Literal["encomenda", "devolucao", "tecnico", "informacao", "reclamacao", "spam"]
    urgencia: Literal["baixa", "media", "alta"]
    nome_cliente: str = Field(description="Nome do cliente, ou texto vazio se não for indicado")
    numero_encomenda: str = Field(description="Número da encomenda (TC-XXXXX), ou texto vazio se não for indicado")
    resumo: str = Field(description="Resumo do pedido numa frase")
    escalar_para_humano: bool
    motivo_escalamento: str = Field(description="Motivo do escalamento, ou texto vazio se não escalar")

CLASSIFICADOR_PROMPT = """És o sistema de triagem de emails da TechCasa, uma loja online de eletrónica em Portugal.

Categoria:
- encomenda: estado ou atraso de uma encomenda ou entrega
- devolucao: devoluções, trocas ou reembolsos
- tecnico: problema técnico com um produto
- informacao: perguntas sobre produtos, preços ou políticas
- reclamacao: cliente insatisfeito com o atendimento ou o serviço
- spam: publicidade ou emails sem relação com a loja
Classifica pelo assunto principal do email. A insatisfação do cliente reflete-se na urgência e no escalamento, não na categoria. Usa "reclamacao" só quando a queixa é sobre o atendimento ou o serviço em geral, sem outro assunto principal.

Escalar para humano APENAS se:
- o cliente ameaça com ação legal, Livro de Reclamações ou redes sociais
- o pedido envolve um reembolso acima de 200€
- o cliente diz que já contactou a loja 2 ou mais vezes sem resposta

Nunca inventes dados: se o nome ou o número da encomenda não estiverem no email, deixa vazio."""

classificador = llm.with_structured_output(Classificacao)

SYSTEM_PROMPT = (
    "És o assistente de apoio ao cliente da TechCasa, uma loja online de eletrónica em Portugal. "
    "Respondes em português de Portugal, de forma curta e profissional.\n"
    "Se o cliente perguntar pelo estado ou entrega de uma encomenda, usa a ferramenta consultar_encomenda. "
    "Se a encomenda não for encontrada, pede ao cliente que confirme o número.\n"
    "Para perguntas sobre garantia, devoluções, reembolsos, portes, pagamentos ou problemas técnicos, "
    "usa a ferramenta politicas_techcasa.\n"
    "Usa apenas informação do cliente ou das ferramentas. Não inventes prazos, processos nem funcionalidades da loja.\n"
    "Só consegues consultar encomendas e políticas e escrever esta resposta. Não ofereças nem prometas fazer mais nada "
    "(contactar transportadoras, verificar stock ou preços, processar compras, reembolsos, cancelamentos ou garantias). "
    "Quando o cliente tiver de dar um passo, explica-lhe o que deve fazer segundo as políticas, em vez de te ofereceres para o fazer. "
    "Se não tiveres a informação de que o cliente precisa, diz-lho com honestidade."
)

# ---------- Estado ----------
class Estado(MessagesState):
    classificacao: dict  # guardada como dicionário (era um objeto Classificacao)
    destino: str
    resposta_final: str

# ---------- Nós ----------
def classificar(state: Estado):
    email = state["messages"][0].content
    resultado = classificador.invoke([SystemMessage(content=CLASSIFICADOR_PROMPT), HumanMessage(content=email)])
    return {"classificacao": resultado.model_dump()}  # converte o objeto Pydantic num dicionário

def assistente(state: Estado):
    if state["classificacao"]["escalar_para_humano"]:
        contexto = (
            "Este caso VAI ser encaminhado para um membro da equipa, que vai dar seguimento. "
            "Podes dizer ao cliente que a equipa vai entrar em contacto."
        )
    else:
        contexto = (
            "Este caso NÃO vai ser encaminhado para ninguém: a tua resposta é a única que o cliente vai receber. "
            "Não digas que o caso foi encaminhado nem que alguém vai entrar em contacto."
        )
    mensagens = [SystemMessage(content=SYSTEM_PROMPT + "\n" + contexto)] + state["messages"]
    return {"messages": [llm_com_ferramentas.invoke(mensagens)]}

def fila_humana(state: Estado):
    c = state["classificacao"]
    proposta = state["messages"][-1].content
    decisao = interrupt({
        "email": state["messages"][0].content,
        "categoria": c["categoria"],
        "urgencia": c["urgencia"],
        "motivo": c["motivo_escalamento"],
        "resposta_proposta": proposta,
    })
    if decisao["acao"] == "aprovar":
        return {"destino": "fila_humana", "resposta_final": proposta}
    if decisao["acao"] == "editar":
        return {"destino": "fila_humana", "resposta_final": decisao["texto"]}
    return {"destino": "rejeitado", "resposta_final": ""}

def resposta_automatica(state: Estado):
    return {"destino": "resposta_automatica", "resposta_final": state["messages"][-1].content}

def ignorar_spam(state: Estado):
    return {"destino": "ignorado"}

# ---------- Decisões ----------
def depois_de_classificar(state: Estado) -> Literal["ignorar_spam", "assistente"]:
    if state["classificacao"]["categoria"] == "spam":
        return "ignorar_spam"
    return "assistente"

def depois_do_assistente(state: Estado) -> Literal["tools", "fila_humana", "resposta_automatica"]:
    if state["messages"][-1].tool_calls:
        return "tools"
    if state["classificacao"]["escalar_para_humano"]:
        return "fila_humana"
    return "resposta_automatica"

# ---------- Grafo ----------
builder = StateGraph(Estado)
builder.add_node("classificar", classificar)
builder.add_node("assistente", assistente)
builder.add_node("tools", ToolNode(tools))
builder.add_node("fila_humana", fila_humana)
builder.add_node("resposta_automatica", resposta_automatica)
builder.add_node("ignorar_spam", ignorar_spam)

builder.add_edge(START, "classificar")
builder.add_conditional_edges("classificar", depois_de_classificar)
builder.add_conditional_edges("assistente", depois_do_assistente)
builder.add_edge("tools", "assistente")
for no_final in ["fila_humana", "resposta_automatica", "ignorar_spam"]:
    builder.add_edge(no_final, END)
graph = builder.compile(checkpointer=InMemorySaver())  # NOVO: guarda o estado durante a pausa

# ---------- Teste interativo ----------
if __name__ == "__main__":
    emails = [
        "Olá, chamo-me Ricardo Matos. Encomendei uma televisão há 3 semanas (encomenda TC-55821) e ainda não chegou. Já enviei 2 emails e ninguém me respondeu. Se não tiver resposta até sexta vou fazer queixa no Livro de Reclamações.",
        "Bom dia, gostava de saber se o portátil Lenovo IdeaPad 5 tem garantia de 3 anos e se posso pagar em prestações. Obrigada, Sofia Costa",
    ]
    for i, email in enumerate(emails, 1):
        config = {"configurable": {"thread_id": f"email-{i}"}}
        r = graph.invoke({"messages": [HumanMessage(content=email)]}, config)

        if "__interrupt__" in r:
            pedido = r["__interrupt__"][0].value
            print("\n" + "=" * 70)
            print("CASO ESCALADO PARA REVISÃO HUMANA")
            print(f"Categoria: {pedido['categoria']} | Urgência: {pedido['urgencia']}")
            print("Motivo:", pedido["motivo"])
            print("\nRESPOSTA PROPOSTA PELO AGENTE:\n" + pedido["resposta_proposta"])
            print("=" * 70)

            # Só aceita uma decisão explícita: Enter ou outra letra voltam a perguntar
            acao = ""
            while acao not in ("a", "e", "r"):
                acao = input("Aprovar (a), editar (e) ou rejeitar (r)? ").strip().lower()

            if acao == "a":
                decisao = {"acao": "aprovar"}
            elif acao == "e":
                decisao = {"acao": "editar", "texto": input("Escreve a resposta final:\n")}
            else:
                decisao = {"acao": "rejeitar"}
            r = graph.invoke(Command(resume=decisao), config)

        print("\n" + "-" * 70)
        print("EMAIL:", email[:70] + "...")
        print("DESTINO:", r["destino"])
        print("RESPOSTA FINAL:", r.get("resposta_final") or "(nenhuma)")