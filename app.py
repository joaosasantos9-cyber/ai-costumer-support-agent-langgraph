import os
import uuid
import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from agente import graph

st.set_page_config(page_title="TechCasa Support Agent", page_icon="📨")
st.title("📨 TechCasa: agente de apoio ao cliente")
st.caption(
    "Cola o email de um cliente. O agente classifica-o, consulta encomendas e políticas "
    "e propõe uma resposta. Os casos escalados precisam de aprovação humana."
)

# ---------- Password (só ativa se APP_PASSWORD estiver definida) ----------
SENHA = os.environ.get("APP_PASSWORD")
if SENHA and not st.session_state.get("autenticado"):
    senha = st.text_input("Password", type="password")
    if senha == SENHA:
        st.session_state.autenticado = True
        st.rerun()
    elif senha:
        st.error("Password errada.")
    st.stop()

    
EXEMPLOS = {
    "Ricardo: encomenda atrasada e ameaça de reclamação": "Olá, chamo-me Ricardo Matos. Encomendei uma televisão há 3 semanas (encomenda TC-55821) e ainda não chegou. Já enviei 2 emails e ninguém me respondeu. Se não tiver resposta até sexta vou fazer queixa no Livro de Reclamações.",
    "Sofia: garantia e prestações": "Bom dia, gostava de saber se o portátil Lenovo IdeaPad 5 tem garantia de 3 anos e se posso pagar em prestações. Obrigada, Sofia Costa",
    "Ana: reembolso de 749€": "Quero o reembolso do portátil (encomenda TC-10234). Custou 749€ e veio com o ecrã partido. Ana Ferreira",
    "Spam": "PARABÉNS!!! Ganhou um iPhone 17 grátis! Clique já neste link para reclamar o seu prémio antes que expire: www.premio-gratis.xyz",
}

# ---------- Memória da sessão (sobrevive aos cliques) ----------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
    st.session_state.resultado = None

def correr(entrada):
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    st.session_state.resultado = graph.invoke(entrada, config)

# ---------- Entrada ----------
exemplo = st.selectbox("Escolhe um exemplo ou escreve o teu", ["(escrever o meu)"] + list(EXEMPLOS))
email = st.text_area("Email do cliente", value=EXEMPLOS.get(exemplo, ""), height=150)

if st.button("Analisar email", type="primary", disabled=not email.strip()):
    st.session_state.thread_id = str(uuid.uuid4())  # cada email é uma conversa nova
    with st.spinner("O agente está a trabalhar..."):
        correr({"messages": [HumanMessage(content=email)]})

# ---------- Resultado ----------
r = st.session_state.resultado
if r:
    c = r["classificacao"]
    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Categoria", c["categoria"])
    col2.metric("Urgência", c["urgencia"])
    col3.metric("Escalar", "Sim" if c["escalar_para_humano"] else "Não")
    if c["motivo_escalamento"]:
        st.info(f"**Motivo do escalamento:** {c['motivo_escalamento']}")

    ferramentas = [ch["name"] for m in r["messages"] for ch in (getattr(m, "tool_calls", None) or [])]
    st.caption("Ferramentas usadas: " + (", ".join(ferramentas) if ferramentas else "nenhuma"))

    if "__interrupt__" in r:
        # O grafo está em pausa à espera de um humano
        pedido = r["__interrupt__"][0].value
        st.warning("Caso escalado: a resposta só é enviada depois de aprovada por um humano.")
        texto = st.text_area(
            "Resposta proposta pelo agente (podes editar antes de enviar)",
            value=pedido["resposta_proposta"],
            height=250,
            key=f"revisao_{st.session_state.thread_id}",
        )
        b1, b2 = st.columns(2)
        if b1.button("✅ Enviar resposta", use_container_width=True):
            if texto.strip() == pedido["resposta_proposta"].strip():
                decisao = {"acao": "aprovar"}
            else:
                decisao = {"acao": "editar", "texto": texto}
            correr(Command(resume=decisao))
            st.rerun()
        if b2.button("❌ Rejeitar", use_container_width=True):
            correr(Command(resume={"acao": "rejeitar"}))
            st.rerun()
    else:
        destino = r["destino"]
        if destino == "ignorado":
            st.error("Spam: o email foi ignorado e nenhuma resposta foi enviada.")
        elif destino == "rejeitado":
            st.error("Rejeitado na revisão humana: nenhuma resposta foi enviada.")
        else:
            st.success("Resposta enviada ao cliente" + (" (revista por um humano)" if destino == "fila_humana" else " (automática)"))
            st.write(r.get("resposta_final", ""))