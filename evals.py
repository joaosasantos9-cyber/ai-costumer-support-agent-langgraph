import csv
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from agente import graph, llm

# ---------- O juiz (LLM-as-judge) ----------
class Avaliacao(BaseModel):
    informacao_inventada: bool = Field(description="True se a resposta contém factos (prazos, preços, processos, políticas) que não estão no email nem nos resultados das ferramentas")
    acoes_impossiveis: bool = Field(description="True se a resposta oferece ou promete ações que o assistente não consegue fazer")
    contradiz_destino: bool = Field(description="True se a resposta contradiz o destino do email")
    explicacao: str = Field(description="Explicação curta da avaliação")

JUIZ_PROMPT = """És um avaliador rigoroso de respostas de apoio ao cliente da TechCasa.

O assistente só consegue fazer três coisas: consultar o estado de encomendas, consultar as políticas da loja e escrever a resposta ao cliente. Não consegue contactar transportadoras, alterar entregas, verificar stock ou preços, processar compras, cancelamentos, reembolsos ou garantias, nem telefonar ao cliente.

Destino do email:
- fila_humana: um membro da equipa vai tratar o caso, por isso a resposta PODE dizer que a equipa vai entrar em contacto.
- resposta_automatica: ninguém da equipa vai tratar este caso, por isso a resposta NÃO pode dizer que o caso foi encaminhado nem que alguém vai tratar dele ou entrar em contacto.

O que É permitido (não conta como erro):
- Repetir informação que está nos resultados das ferramentas, incluindo prazos gerais das políticas (por exemplo, o prazo de resposta do apoio por email).
- Indicar o formato do número de encomenda (TC-XXXXX) ao pedir esse número ao cliente.
- Pedir ao cliente mais informação ou que responda a este email: a resposta dele chega como um novo email.
- Explicar ao cliente o que ele deve fazer, desde que decorra diretamente das políticas.

Avalia a resposta com base apenas no email, nos resultados das ferramentas e no destino."""

juiz = llm.with_structured_output(Avaliacao)

# ---------- Correr e avaliar um email ----------
def avaliar(linha):
    r = graph.invoke({"messages": [HumanMessage(content=linha["chatInput"])]})
    c = r["classificacao"]
    ferramentas = [ch["name"] for m in r["messages"] for ch in (getattr(m, "tool_calls", None) or [])]
    observacoes = [m.content for m in r["messages"] if m.type == "tool"]
    politicas_esperada = (linha.get("politicas_esperadas") or linha.get("politicas_esperada") or "").strip()

    resultado = {
        "email": linha["chatInput"][:45],
        "categoria": c.categoria == linha["categoria_esperada"].strip(),
        "escalar": c.escalar_para_humano == (linha["escalar_esperado"].strip() == "sim"),
        "encomendas": ("consultar_encomenda" in ferramentas) == (linha["ferramenta_esperada"].strip() == "sim"),
        "politicas": None if politicas_esperada == "qualquer" else ("politicas_techcasa" in ferramentas) == (politicas_esperada == "sim"),
        "destino": r["destino"],
        "juiz": None,
        "explicacao_juiz": "",
        "resposta": "",
    }

    if r["destino"] != "ignorado":
        resposta = r["messages"][-1].content
        texto = (
            f"EMAIL:\n{linha['chatInput']}\n\n"
            f"RESULTADOS DAS FERRAMENTAS:\n{chr(10).join(observacoes) or 'nenhum'}\n\n"
            f"DESTINO: {r['destino']}\n\n"
            f"RESPOSTA:\n{resposta}"
        )
        a = juiz.invoke([SystemMessage(content=JUIZ_PROMPT), HumanMessage(content=texto)])
        resultado["juiz"] = not (a.informacao_inventada or a.acoes_impossiveis or a.contradiz_destino)
        resultado["explicacao_juiz"] = a.explicacao
        resultado["resposta"] = resposta
    return resultado

def simbolo(valor):
    return "—" if valor is None else ("✅" if valor else "❌")

# ---------- Correr tudo ----------
if __name__ == "__main__":
    with open("dataset.csv", encoding="utf-8-sig") as f:
        linhas = list(csv.DictReader(f))

    resultados = []
    for i, linha in enumerate(linhas, 1):
        print(f"A avaliar {i}/{len(linhas)}...")
        resultados.append(avaliar(linha))

    print("\n" + "=" * 70)
    for i, r in enumerate(resultados, 1):
        print(f"{i:>2}. {r['email']}... | cat {simbolo(r['categoria'])} | esc {simbolo(r['escalar'])} "
              f"| enc {simbolo(r['encomendas'])} | pol {simbolo(r['politicas'])} | juiz {simbolo(r['juiz'])}")
        if r["juiz"] is False:
            print("      Juiz:", r["explicacao_juiz"])

    print("\nRESUMO")
    for metrica in ["categoria", "escalar", "encomendas", "politicas", "juiz"]:
        valores = [r[metrica] for r in resultados if r[metrica] is not None]
        print(f"  {metrica}: {sum(valores)}/{len(valores)}")

    with open("evals_resultados.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=resultados[0].keys())
        writer.writeheader()
        writer.writerows(resultados)
    print("\nResultados guardados em evals_resultados.csv")