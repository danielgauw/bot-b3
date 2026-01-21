import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv

# --- CORREÇÃO DE CAMINHO ABSOLUTO (A BLINDAGEM) ---
# Isso garante que o Cron ache os arquivos (.env, json, etc)
DIRETORIO_BASE = os.path.dirname(os.path.abspath(__file__))
CAMINHO_ENV = os.path.join(DIRETORIO_BASE, '.env')
CAMINHO_TRADES = os.path.join(DIRETORIO_BASE, 'trades_simulados.json')
CAMINHO_CARTEIRA = os.path.join(DIRETORIO_BASE, 'carteira_alvo.json')

# Carrega o .env explicitamente do caminho certo
load_dotenv(CAMINHO_ENV)

# Bibliotecas de Dados
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, ADXIndicator
from ta.volatility import AverageTrueRange

# Bibliotecas de IA
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
import telebot

# --- IMPORTAÇÃO DA BUSCA ---
try:
    from duckduckgo_search import DDGS
except ImportError:
    try:
        from ddgs import DDGS
    except ImportError:
        DDGS = None

# --- CONFIGURAÇÃO DE CHAVES ---
# (Pega direto do arquivo .env ou do sistema)
if os.getenv("GOOGLE_API_KEY"):
    os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

bot = telebot.TeleBot(TELEGRAM_TOKEN)

# --- 1. O HARD SCREEN (MATEMÁTICA) ---
def validar_setup_v2(ticker):
    try:
        df = yf.download(ticker, period="1y", progress=False)
        if df.empty: return False, None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Filtro de liquidez/atualização (5 dias)
        if (datetime.now() - df.index[-1].to_pydatetime()).days > 5:
            return False, None

        df['SMA200'] = SMAIndicator(df['Close'], window=200).sma_indicator()
        df['SMA50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
        df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
        adx = ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
        df['ADX'] = adx.adx()
        atr = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14)
        df['ATR'] = atr.average_true_range()

        atual = df.iloc[-1]

        tendencia = (atual['Close'] > atual['SMA200']) and (atual['Close'] > atual['SMA50'])
        forca = atual['ADX'] > 20
        pullback = (atual['RSI'] < 60) and (atual['RSI'] > 35)

        if tendencia and forca and pullback:
            return True, df
        
        return False, None

    except Exception as e:
        print(f"Erro no screener ({ticker}): {e}")
        return False, None

# --- 2. FERRAMENTA DE BUSCA ---
@tool("News Search")
def search_news(query: str):
    """Busca notícias recentes."""
    if DDGS is None:
        return "Erro: Instale 'pip install -U duckduckgo-search'"
        
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, region='br-pt', max_results=3))
        
        if not results:
            return "Nenhuma notícia encontrada. Seguir análise técnica."
            
        return str(results)
        
    except Exception as e:
        return f"Erro na busca ({str(e)}). Assumir risco neutro."

# --- 3. AGENTES (Gemini 2.0 Flash) ---
MODELO_IA = "gemini/gemini-2.0-flash"

analista_risco = Agent(
    role='Risk Manager',
    goal='Ler notícias. Se houver erro na busca ou nenhuma notícia, APROVAR.',
    backstory='Você analisa riscos. Se a ferramenta de busca falhar, você assume que não há notícias ruins e libera.',
    tools=[search_news],
    llm=MODELO_IA,
    verbose=True
)

manager = Agent(
    role='CIO',
    goal='Decidir trade.',
    backstory='Decide compra/venda. Se o Risk Manager liberar, você define entrada e stop.',
    llm=MODELO_IA,
    verbose=True
)

# --- 4. TAREFAS ---
t_risco = Task(
    description='Busque notícias de {ticket}. Se der erro, responda "Sem notícias relevantes".',
    expected_output='Resumo curto.',
    agent=analista_risco
)

t_manager = Task(
    description='''O ativo {ticket} passou na matemática (Preço: {price}, ATR: {atr}).
    Retorne APENAS JSON:
    {{
        "ticker": "{ticket}",
        "decisao": "COMPRA" ou "CANCELAR",
        "entrada": float,
        "stop": float,
        "alvo": float,
        "confianca": "ALTA" ou "MEDIA",
        "motivo": "resumo curto"
    }}''',
    expected_output='JSON Válido.',
    agent=manager,
    context=[t_risco]
)

equipe = Crew(
    agents=[analista_risco, manager],
    tasks=[t_risco, t_manager],
    process=Process.sequential
)

# --- FUNÇÃO CORRIGIDA: REGISTRAR TRADE ---
def registrar_trade(sinal):
    # Usa o CAMINHO_TRADES absoluto definido lá em cima
    historico = []
    
    if os.path.exists(CAMINHO_TRADES):
        with open(CAMINHO_TRADES, "r") as f:
            try:
                historico = json.load(f)
            except:
                pass
    
    # Evita duplicatas do dia
    hoje = datetime.now().strftime("%Y-%m-%d")
    for trade in historico:
        if trade['ticker'] == sinal['ticker'] and trade['data'].startswith(hoje):
            return 

    novo_trade = {
        "data": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": sinal['ticker'],
        "entrada": sinal['entrada'],
        "stop": sinal['stop'],
        "alvo": sinal['alvo'],
        "status": "ABERTO",
        "resultado_financeiro": 0.0,
        "confianca": sinal['confianca']
    }
    
    historico.append(novo_trade)
    
    with open(CAMINHO_TRADES, "w") as f:
        json.dump(historico, f, indent=4)
        
    print(f"📝 Trade simulado registrado no caderno: {sinal['ticker']}")

# --- 5. TELEGRAM ---
def enviar_alerta(sinal):
    if not bot: return
    emoji = "🟢" if sinal.get('confianca') == "ALTA" else "🟡"
    msg = f"""
🚀 **SINAL: {sinal.get('ticker')}**
📊 **Status:** `STRONG BUY` {emoji}
💰 **Entrada:** `R$ {sinal.get('entrada')}`
🛑 **Stop:** `R$ {sinal.get('stop')}`
🏁 **Alvo:** `R$ {sinal.get('alvo')}`
📝 **Motivo:** {sinal.get('motivo')}
    """
    try:
        bot.send_message(TELEGRAM_CHAT_ID, msg, parse_mode="Markdown")
    except Exception as e:
        print(f"Erro Telegram: {e}")

# --- 6. EXECUÇÃO ---
def rodar_robo():
    print("--- INICIANDO ROBÔ DE SWING TRADE (V7 - SMART & BLINDADO) ---")
    
    # Usa o CAMINHO_CARTEIRA absoluto
    if not os.path.exists(CAMINHO_CARTEIRA):
        print(f"Erro: {CAMINHO_CARTEIRA} não encontrado.")
        # Cria um arquivo padrão se não existir para não quebrar
        with open(CAMINHO_CARTEIRA, "w") as f:
            json.dump(["WEGE3.SA", "VALE3.SA", "PETR4.SA", "ITUB4.SA", "BBAS3.SA"], f)
        
    with open(CAMINHO_CARTEIRA, "r") as f:
        carteira = json.load(f)
        
    for ticker in carteira:
        print(f"\n🔎 Analisando {ticker}...")
        aprovado, df = validar_setup_v2(ticker)
        
        if aprovado:
            print(f"✅ {ticker} Aprovado na Matemática! Chamando IA...")
            inputs = {
                'ticket': ticker, 
                'atr': f"{df['ATR'].iloc[-1]:.2f}",
                'price': f"{df['Close'].iloc[-1]:.2f}"
            }
            try:
                print("⏳ Aguardando 20s para respeitar limite do Google...")
                time.sleep(20)
                
                resultado = equipe.kickoff(inputs=inputs)
                
                # Tratamento de erro da string JSON
                if hasattr(resultado, 'raw'):
                    texto_limpo = resultado.raw
                else:
                    texto_limpo = str(resultado)
                
                texto_limpo = texto_limpo.replace('```json', '').replace('```', '').strip()
                sinal = json.loads(texto_limpo)
                
                if sinal['decisao'] == "COMPRA":
                    print(f"🚀 COMPRA CONFIRMADA: {ticker}")
                    enviar_alerta(sinal)
                    registrar_trade(sinal)
                else:
                    print(f"❌ {ticker} vetado pela IA.")
            except Exception as e:
                print(f"Erro IA: {e}")
        else:
            pass
            
    print("--- FIM DA EXECUÇÃO ---")

if __name__ == "__main__":
    rodar_robo()