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
from langchain_google_genai import ChatGoogleGenerativeAI
import telebot

# --- CONFIGURAÇÃO ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

bot = telebot.TeleBot(TELEGRAM_TOKEN)

# Configuração do Modelo Gemini (Google)
google_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro", 
    api_key=GOOGLE_API_KEY, 
    temperature=0.1
)

# --- 1. O HARD SCREEN (MATEMÁTICA V2 - VALIDADA) ---
def validar_setup_v2(ticker):
    """
    Aplica o filtro matemático que obteve 46% de Win Rate no Backtest.
    Critérios: Tendência Alta + Força (ADX) + Pullback (RSI).
    """
    try:
        # Baixa dados (período maior para médias longas)
        df = yf.download(ticker, period="1y", progress=False)
        if df.empty: return False, None
        
        # Limpeza MultiIndex (Correção para yfinance novo)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Validação de Frescor (Dados atrasados > 3 dias são ignorados)
        # Útil para feriados ou fins de semana
        if (datetime.now() - df.index[-1].to_pydatetime()).days > 4:
            return False, None

        # Cálculos V2
        df['SMA200'] = SMAIndicator(df['Close'], window=200).sma_indicator()
        df['SMA50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
        df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
        adx = ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
        df['ADX'] = adx.adx()
        
        # ATR para Stop Técnico
        atr = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14)
        df['ATR'] = atr.average_true_range()

        atual = df.iloc[-1]

        # REGRAS DO BACKTEST
        # 1. Tendência: Preço acima das médias E médias alinhadas
        tendencia = (atual['Close'] > atual['SMA200']) and (atual['Close'] > atual['SMA50'])
        
        # 2. Força: O mercado não pode estar lateral (ADX > 20)
        forca = atual['ADX'] > 20
        
        # 3. Oportunidade: RSI recuou mas não morreu (35 < RSI < 60)
        pullback = (atual['RSI'] < 60) and (atual['RSI'] > 35)

        if tendencia and forca and pullback:
            return True, df
        
        return False, None

    except Exception as e:
        print(f"Erro no screener ({ticker}): {e}")
        return False, None

# --- 2. FERRAMENTAS PARA IA (TOOLS) ---

@tool("News Search")
def search_news(query: str):
    """Busca notícias recentes para identificar riscos sistêmicos ou de governança."""
    search = DuckDuckGoSearchRun()
    return search.run(query)

# --- 3. AGENTES (O COMITÊ DE INVESTIMENTO) ---

# Agente 1: O Gestor de Risco (Paranoico)
analista_risco = Agent(
    role='Risk Manager',
    goal='VETAR a operação se houver notícias ruins (corrupção, processos, política, resultados ruins).',
    backstory='Você é pago para proteger o capital. Se houver dúvida ou notícia ruim recente (últimos 3 dias), você VETA. Você é extremamente conservador.',
    tools=[search_news],
    llm=google_llm,
    verbose=True
)

# Agente 2: O CIO (Decisor)
manager = Agent(
    role='CIO',
    goal='Decidir a entrada e definir Stop/Alvo baseados no ATR.',
    backstory='''Você recebe um ativo que JÁ PASSOU na matemática. Sua função é:
    1. Ler o parecer do Analista de Risco.
    2. Se o risco for alto, cancele.
    3. Se for seguro, defina os preços:
       - Stop Loss = Preço Atual - (2.0 * ATR)
       - Alvo = Preço Atual + (4.0 * ATR)
    ''',
    llm=google_llm,
    verbose=True
)

# --- 4. TAREFAS ---

t_risco = Task(
    description='Busque notícias urgentes e recentes de {ticket} no Brasil. Há algo grave que possa derrubar a ação nos próximos dias? Resuma os riscos.',
    expected_output='Resumo de riscos. Veredito final: SEGURO ou PERIGOSO.',
    agent=analista_risco
)

t_manager = Task(
    description='''O ativo {ticket} passou no filtro matemático. O preço atual é {price} e o ATR é {atr}.
    Baseado no risco identificado, decida.
    
    Retorne APENAS um JSON neste formato exato (sem ```json):
    {{
        "ticker": "{ticket}",
        "decisao": "COMPRA" ou "CANCELAR",
        "entrada": float,
        "stop": float,
        "alvo": float,
        "confianca": "ALTA" ou "MEDIA",
        "motivo": "resumo curto em pt-br"
    }}''',
    expected_output='JSON Válido.',
    agent=manager,
    context=[t_risco] # O Manager recebe o output do Risco
)

equipe = Crew(
    agents=[analista_risco, manager],
    tasks=[t_risco, t_manager],
    process=Process.sequential
)

# --- 5. SISTEMA DE ENVIO (TELEGRAM DASHBOARD) ---

def enviar_alerta(sinal):
    if not bot: return
    
    # Emoji de Confiança
    emoji_conf = "🟢" if sinal['confianca'] == "ALTA" else "🟡"
    
    msg = f"""
🚀 **SINAL DETECTADO: {sinal['ticker']}**
───────────────────────
📊 **Status:** `STRONG BUY` {emoji_conf}
🎯 **Confiança:** {sinal['confianca']}
───────────────────────
💰 **Entrada:** `R$ {sinal['entrada']:.2f}`
🛑 **Stop:** `R$ {sinal['stop']:.2f}`
🏁 **Alvo:** `R$ {sinal['alvo']:.2f}`
───────────────────────
📝 **Motivo:** {sinal['motivo']}
    """
    try:
        bot.send_message(TELEGRAM_CHAT_ID, msg, parse_mode="Markdown")
    except Exception as e:
        print(f"Erro ao enviar Telegram: {e}")

# --- 6. EXECUÇÃO PRINCIPAL ---

def rodar_robo():
    print("--- INICIANDO ROBÔ DE SWING TRADE (PRODUÇÃO) ---")
    
    # 1. Carrega Universo
    if not os.path.exists("carteira_alvo.json"):
        print("Erro: carteira_alvo.json não encontrado. Rode o gerador primeiro.")
        return
        
    with open("carteira_alvo.json", "r") as f:
        carteira = json.load(f)
        
    # 2. Loop de Varredura
    for ticker in carteira:
        print(f"\n🔎 Analisando {ticker}...")
        
        # PASSO A: Hard Screen (Matemática)
        aprovado, df = validar_setup_v2(ticker)
        
        if aprovado:
            print(f"✅ {ticker} passou no filtro Matemático! Acionando IA...")
            
            # Prepara dados para a IA
            atr_atual = df['ATR'].iloc[-1]
            preco_atual = df['Close'].iloc[-1]
            
            inputs = {
                'ticket': ticker, 
                'atr': f"{atr_atual:.2f}",
                'price': f"{preco_atual:.2f}"
            }
            
            # PASSO B: IA Agents
            try:
                resultado = equipe.kickoff(inputs=inputs)
                
                # Limpeza JSON (Tratamento de erro de formatação da IA)
                texto_limpo = str(resultado)
                texto_limpo = texto_limpo.replace('```json', '').replace('```', '').strip()
                
                sinal = json.loads(texto_limpo)
                
                if sinal['decisao'] == "COMPRA":
                    print(f"🚀 COMPRA CONFIRMADA: {ticker}")
                    enviar_alerta(sinal)
                else:
                    print(f"❌ {ticker} vetado pela IA: {sinal.get('motivo')}")
                    
            except Exception as e:
                print(f"Erro na IA com {ticker}: {e}")
        else:
            # print(f"Neutro: {ticker}") 
            pass
            
    print("--- FIM DA EXECUÇÃO ---")

if __name__ == "__main__":
    rodar_robo()