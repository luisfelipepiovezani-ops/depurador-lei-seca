"""
Depurador de Lei Seca — Motor de IA e Dados (versão para nuvem / upload)

Esta versão foi adaptada para rodar no Streamlit Community Cloud (ou qualquer
servidor remoto): em vez de ler uma pasta local 'meus_materiais/', o usuário
ENVIA o CSV e o PDF pela própria página. Assim não é preciso subir dados pro
repositório nem instalar nada na máquina local.

Demais melhorias (mantidas da versão anterior):
  - SDK Gemini ATUAL (google-genai). O antigo google-generativeai foi descontinuado.
  - Modelo atual (gemini-1.5-flash foi DESLIGADO). Trocável na constante MODELO.
  - Chamadas de IA em LOTE (batching) -> mais rápido e barato.
  - Cache do Streamlit -> não reprocessa PDF/IA a cada clique.
  - Chave da API via st.secrets / variável de ambiente (nunca no código).
  - Extração de artigos por REGEX (do "Art. N" até o próximo), sem .find() frágil.
  - Normalização dos artigos antes do value_counts().
  - pypdf no lugar de PyPDF2 (legado).
  - Erros tratados de forma específica, sem except silencioso.

Dependências (ver requirements.txt):
    streamlit, pandas, pypdf, google-genai
"""

from __future__ import annotations

import io
import json
import os
import re

import pandas as pd
import streamlit as st
from pypdf import PdfReader

from google import genai
from google.genai import types

# --------------------------------------------------------------------------- #
# Configuração
# --------------------------------------------------------------------------- #

MODELO = "gemini-2.5-flash"   # troque por um mais novo (ex.: "gemini-3.5-flash") se quiser
TAMANHO_LOTE = 12             # quantos comentários por chamada de IA no mapeamento


def obter_chave_api() -> str | None:
    """Lê a chave do st.secrets ou da variável de ambiente. Nunca hardcoded."""
    chave = None
    try:
        chave = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        chave = None
    return chave or os.environ.get("GEMINI_API_KEY")


@st.cache_resource(show_spinner=False)
def obter_cliente() -> genai.Client:
    """Cria o cliente Gemini uma única vez (cacheado entre reruns)."""
    chave = obter_chave_api()
    if not chave:
        raise RuntimeError(
            "Chave da API do Gemini não encontrada. "
            "No Streamlit Cloud, adicione GEMINI_API_KEY em Settings > Secrets."
        )
    return genai.Client(api_key=chave)


# --------------------------------------------------------------------------- #
# Leitura de dados (recebem BYTES do upload -> cacheável e testável)
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner=False)
def carregar_csv(conteudo: bytes) -> pd.DataFrame:
    """Lê o CSV enviado (separador ';'). Cacheado pelo conteúdo do arquivo."""
    try:
        return pd.read_csv(io.BytesIO(conteudo), sep=";", dtype=str, keep_default_na=False)
    except pd.errors.ParserError as e:
        raise ValueError(
            "Não consegui ler o CSV. O separador esperado é ';'. "
            f"Detalhe: {e}"
        ) from e


@st.cache_data(show_spinner=False)
def extrair_texto_pdf(conteudo: bytes) -> str:
    """Extrai todo o texto do PDF enviado usando pypdf. Cacheado pelo conteúdo."""
    leitor = PdfReader(io.BytesIO(conteudo))
    partes = [(pagina.extract_text() or "") for pagina in leitor.pages]
    return "\n".join(partes)


# --------------------------------------------------------------------------- #
# Processamento de texto (funções puras — sem Streamlit, fáceis de testar)
# --------------------------------------------------------------------------- #

_RE_ARTIGO = re.compile(r"art(?:igo)?\.?\s*0*(\d+)", re.IGNORECASE)
_RE_CABECALHO_ARTIGO = re.compile(r"(?im)^\s*art(?:igo)?\.?\s*0*(\d+)\s*[º°ªo.\-]?")


def normalizar_artigo(texto: str | None) -> str | None:
    """Converte variações ('Artigo 5º', 'art 5', 'Art. 005') na forma canônica 'Art. 5'."""
    if not texto:
        return None
    m = _RE_ARTIGO.search(texto)
    if not m:
        return None
    return f"Art. {int(m.group(1))}"


def extrair_artigos(texto_pdf: str) -> dict[str, str]:
    """
    Quebra o texto do PDF em {'Art. N': corpo_do_artigo}, do cabeçalho de cada
    artigo até o próximo. Se o mesmo número aparecer mais de uma vez, fica com o
    trecho MAIS LONGO (o artigo de fato, não uma citação cruzada).
    """
    matches = list(_RE_CABECALHO_ARTIGO.finditer(texto_pdf))
    artigos: dict[str, str] = {}
    for i, m in enumerate(matches):
        inicio = m.start()
        fim = matches[i + 1].start() if i + 1 < len(matches) else len(texto_pdf)
        chave = f"Art. {int(m.group(1))}"
        corpo = texto_pdf[inicio:fim].strip()
        if len(corpo) > len(artigos.get(chave, "")):
            artigos[chave] = corpo
    return artigos


# --------------------------------------------------------------------------- #
# Camada de IA (Gemini) — cacheada e em lote
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner=False)
def _gerar(prompt: str, json_mode: bool = False, modelo: str = MODELO) -> str:
    """Chamada única ao Gemini, cacheada por (prompt, json_mode, modelo)."""
    cliente = obter_cliente()
    config = types.GenerateContentConfig(
        response_mime_type="application/json" if json_mode else "text/plain"
    )
    resposta = cliente.models.generate_content(model=modelo, contents=prompt, config=config)
    return (resposta.text or "").strip()


def _parse_json_seguro(bruto: str):
    """Tenta json.loads; se vier com cercas de código, limpa antes."""
    limpo = re.sub(r"^```(?:json)?|```$", "", bruto.strip(), flags=re.MULTILINE).strip()
    return json.loads(limpo)


def mapear_comentarios_para_artigos(comentarios: list[str]) -> list[str | None]:
    """Para cada comentário, descobre o artigo principal — EM LOTE (TAMANHO_LOTE por chamada)."""
    resultado: list[str | None] = [None] * len(comentarios)

    for inicio in range(0, len(comentarios), TAMANHO_LOTE):
        lote = comentarios[inicio : inicio + TAMANHO_LOTE]
        itens = [{"i": j, "comentario": c} for j, c in enumerate(lote)]
        prompt = (
            "Você analisa comentários de professores sobre questões de concurso. "
            "Para CADA item, identifique o artigo principal da lei a que ele se refere.\n"
            "Responda APENAS com um JSON: uma lista de objetos no formato "
            '{"i": <indice>, "artigo": "Art. N"} — use null em "artigo" se não houver artigo claro.\n\n'
            f"Itens:\n{json.dumps(itens, ensure_ascii=False)}"
        )
        try:
            dados = _parse_json_seguro(_gerar(prompt, json_mode=True))
            for obj in dados:
                idx = inicio + int(obj["i"])
                if 0 <= idx < len(resultado):
                    resultado[idx] = normalizar_artigo(obj.get("artigo"))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            st.warning(f"Falha ao mapear o lote iniciado em {inicio}: {e}")

    return resultado


def analisar_pegadinhas(artigo: str, comentarios: list[str]) -> str:
    """Gera mnemônico + 'pegadinhas da banca' para um artigo, com base nos comentários reais."""
    amostra = "\n---\n".join(comentarios[:20])
    prompt = (
        f"Você é um professor de concursos. Com base nos comentários reais abaixo sobre o {artigo}, "
        "produza: (1) um mnemônico curto e fácil de memorizar; (2) as principais 'pegadinhas' "
        "que as bancas costumam usar nesse artigo. Seja direto e prático.\n\n"
        f"Comentários:\n{amostra}"
    )
    return _gerar(prompt)


# --------------------------------------------------------------------------- #
# Orquestração do caderno
# --------------------------------------------------------------------------- #


def montar_caderno(df: pd.DataFrame, coluna_comentario: str, texto_pdf: str, top_n: int) -> str:
    """Junta estatística de incidência, texto da lei e análise da banca num caderno .txt."""
    comentarios = df[coluna_comentario].fillna("").astype(str).tolist()

    artigos_por_linha = mapear_comentarios_para_artigos(comentarios)
    df = df.assign(artigo=artigos_por_linha)

    incidencia = df["artigo"].dropna().value_counts().head(top_n)
    corpo_por_artigo = extrair_artigos(texto_pdf)

    linhas: list[str] = ["CADERNO DE RESUMO — LEI SECA POR INCIDÊNCIA", "=" * 48, ""]
    for artigo, qtd in incidencia.items():
        comentarios_artigo = df.loc[df["artigo"] == artigo, coluna_comentario].tolist()
        linhas += [
            f"## {artigo}  ({qtd} questão(ões))",
            "",
            "Texto da lei:",
            corpo_por_artigo.get(artigo, "[Artigo não localizado no PDF]"),
            "",
            "Análise da banca:",
            analisar_pegadinhas(artigo, comentarios_artigo),
            "",
            "-" * 48,
            "",
        ]
    return "\n".join(linhas)


# --------------------------------------------------------------------------- #
# Interface Streamlit (com upload de arquivos)
# --------------------------------------------------------------------------- #


def main() -> None:
    st.set_page_config(page_title="Depurador de Lei Seca", page_icon="📘")
    st.title("📘 Depurador de Lei Seca")
    st.caption("Envie a planilha de questões (.csv) e o texto da lei (.pdf) para gerar o caderno.")

    col1, col2 = st.columns(2)
    arq_csv = col1.file_uploader("Planilha de questões (.csv)", type=["csv"])
    arq_pdf = col2.file_uploader("Texto da lei (.pdf)", type=["pdf"])

    if arq_csv is None or arq_pdf is None:
        st.info("Envie os dois arquivos para continuar.")
        return

    try:
        df = carregar_csv(arq_csv.getvalue())
        texto_pdf = extrair_texto_pdf(arq_pdf.getvalue())
    except (ValueError, Exception) as e:  # noqa: BLE001 — mostra o erro real ao usuário
        st.error(f"Erro ao ler os arquivos: {e}")
        return

    candidatas = [c for c in df.columns if "coment" in c.lower()] or list(df.columns)
    coluna = st.selectbox("Coluna com o comentário do professor", df.columns,
                          index=list(df.columns).index(candidatas[0]))
    top_n = st.slider("Quantos artigos mais cobrados incluir?", 5, 50, 15)

    if st.button("Gerar caderno", type="primary"):
        with st.spinner("Mapeando artigos e gerando análises com IA..."):
            try:
                caderno = montar_caderno(df, coluna, texto_pdf, top_n)
            except RuntimeError as e:  # ex.: chave da API ausente
                st.error(str(e))
                return

        st.success("Caderno gerado!")
        st.download_button(
            "📥 Baixar caderno (.txt)",
            data=caderno.encode("utf-8"),
            file_name="caderno_lei_seca.txt",
            mime="text/plain",
        )
        st.text_area("Prévia", caderno, height=400)


if __name__ == "__main__":
    main()
