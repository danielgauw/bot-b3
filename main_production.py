import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv

# Bibliotecas de Dados
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, ADXIndicator
from ta.volatility import AverageTrueRange

# Bibliotecas de IA
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import telebot

# --- CONFIGURAÇÃO ---
load_dotenv()

# VACINA: Garante que a chave do Google seja vista pelos novos sistemas do CrewAI
if os.getenv("GOOGLE_API_KEY"):
    os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

bot = telebot.TeleBot(TELEGRAM_TOKEN)

# --- 1. O HARD SCREEN (MATEMÁTICA V2) ---
def validar_setup_v2(ticker):
    try:
        df = yf.download(ticker, period="1y", progress=False)
        if df.empty: return False, None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Ignora dados muito velhos
        if (datetime.now() - df.index[-1].to_pydatetime()).days > 5:
            return False, None

        # Indicadores
        df['SMA200'] = SMAIndicator(df['Close'], window=200).sma_indicator()
        df['SMA50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
        df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
        adx = ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
        df['ADX'] = adx.adx()
        atr = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14)
        df['ATR'] = atr.average_true_range()

        atual = df.iloc[-1]

        # Regras
        tendencia = (atual['Close'] > atual['SMA200']) and (atual['Close'] > atual['SMA50'])
        forca = atual['ADX'] > 20
        pullback = (atual['RSI'] < 60) and (atual['RSI'] > 35)

        if tendencia and forca and pullback:
            return True, df
        
        return False, None

    except Exception as e:
        print(f"Erro no screener ({ticker}): {e}")
        return False, None

# --- 2. FERRAMENTAS ---

@tool("News Search")
def search_news(query: str):
    """Busca notícias recentes."""
    search = DuckDuckGoSearchRun()
    return search.run(query)

# --- 3. AGENTES (Versão Compatível CrewAI Novo) ---

# Usamos o formato novo de string
MODELO_IA = "gemini/gemini-pro"

analista_risco = Agent(
    role='Risk Manager',
    goal='VETAR a operação se houver notícias ruins.',
    backstory='Você é um gestor de risco conservador. Se houver más notícias recentes (3 dias), você VETA.',
    tools=[search_news],
    llm=MODELO_IA,
    verbose=True
)

manager = Agent(
    role='CIO',
    goal='Decidir entrada, Stop e Alvo.',
    backstory='Você decide a compra baseada no risco e define os preços técnicos usando ATR.',
    llm=MODELO_IA,
    verbose=True
)

# --- 4. TAREFAS ---

t_risco = Task(
    description='Busque notícias urgentes de {ticket} no Brasil. Resuma os riscos.',
    expected_output='Resumo de riscos e veredito: SEGURO ou PERIGOSO.',
    agent=analista_risco
)

t_manager = Task(
    description='''O ativo {ticket} passou na matemática. Preço: {price}, ATR: {atr}.
    Decida com base no risco.
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
    except Exception:
        pass

# --- 6. EXECUÇÃO ---

def rodar_robo():
    print("--- INICIANDO ROBÔ DE SWING TRADE (VERSÃO FINAL) ---")
    
    if not os.path.exists("carteira_alvo.json"):
        print("Erro: carteira_alvo.json não encontrado.")
        return
        
    with open("carteira_alvo.json", "r") as f:
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
                resultado = equipe.kickoff(inputs=inputs)
                texto_limpo = str(resultado).replace('```json', '').replace('```', '').strip()
                sinal = json.loads(texto_limpo)
                
                if sinal['decisao'] == "COMPRA":
                    print(f"🚀 COMPRA CONFIRMADA: {ticker}")
                    enviar_alerta(sinal)
                else:
                    print(f"❌ {ticker} vetado pela IA.")
            except Exception as e:
                print(f"Erro IA: {e}")
        else:
            pass
            
    print("--- FIM DA EXECUÇÃO ---")

if __name__ == "__main__":
    rodar_robo()