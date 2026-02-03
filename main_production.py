import os
import json
import time
import math
from datetime import datetime, timedelta
from dotenv import load_dotenv

# --- INFRAESTRUTURA BLINDADA ---
DIRETORIO_BASE = os.path.dirname(os.path.abspath(__file__))
CAMINHO_ENV = os.path.join(DIRETORIO_BASE, '.env')
CAMINHO_TRADES = os.path.join(DIRETORIO_BASE, 'trades_simulados.json')
CAMINHO_CARTEIRA = os.path.join(DIRETORIO_BASE, 'carteira_alvo.json')

load_dotenv(CAMINHO_ENV)

import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, ADXIndicator
from ta.volatility import AverageTrueRange
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
import telebot

# --- CONFIGURAÇÕES V8 (GESTÃO DE RISCO) ---
# Item 2 do Roadmap: Risco Fixo em Reais
RISCO_MAXIMO_POR_TRADE = 150.00  # Quanto aceito perder em Reais se der Stop
CAPITAL_VIRTUAL = 10000.00       # Base para travas de segurança

# --- IMPORTAÇÃO DA BUSCA ---
try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

# --- CHAVES ---
if os.getenv("GOOGLE_API_KEY"):
    os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
bot = telebot.TeleBot(TELEGRAM_TOKEN)

# --- FUNÇÕES AUXILIARES V8 ---

def get_volume_projetado(volume_atual, media_volume):
    """
    Item 1 do Roadmap: Inteligência de Dados (Volume).
    Corrige o problema do 'Volume Ratio 0.00' projetando o volume
    com base no tempo decorrido do pregão.
    """
    agora = datetime.now()
    # Define horário do pregão (10h às 17h)
    abertura = agora.replace(hour=10, minute=0, second=0, microsecond=0)
    fechamento = agora.replace(hour=17, minute=0, second=0, microsecond=0)
    
    if agora < abertura: return 0.0
    
    minutos_totais = (fechamento - abertura).total_seconds() / 60
    minutos_passados = (agora - abertura).total_seconds() / 60
    
    if minutos_passados <= 0: return 0.0
    
    # % do dia que já passou (máximo 1.0)
    pct_decorrido = min(minutos_passados / minutos_totais, 1.0)
    
    # Proteção: Se for muito cedo (primeiros 10 min), a projeção é instável.
    # Retornamos 1.0 (neutro) para não bloquear trades.
    if pct_decorrido < 0.02: return 1.0
    
    volume_esperado_ate_agora = media_volume * pct_decorrido
    
    if volume_esperado_ate_agora == 0: return 0.0
    
    return volume_atual / volume_esperado_ate_agora

def calcular_posicao(preco_entrada, stop_loss):
    """
    Item 2 do Roadmap: Position Sizing Dinâmico.
    Calcula quantos lotes comprar baseado no risco financeiro,
    e não mais um lote fixo.
    """
    risco_por_acao = preco_entrada - stop_loss
    
    # Segurança contra divisão por zero
    if risco_por_acao <= 0.01: return 100 
    
    # Qtd = Risco Financeiro ($150) / Risco Unitário
    qtd = int(RISCO_MAXIMO_POR_TRADE / risco_por_acao)
    
    # Trava de Segurança: Nunca alocar mais de 30% do capital em um único ativo
    qtd_max_capital = int((CAPITAL_VIRTUAL * 0.30) / preco_entrada)
    
    return min(qtd, qtd_max_capital)

# --- 1. CORE QUANTITATIVO (Validar Setup) ---
def validar_setup_v8(ticker):
    try:
        # Baixa mais dados para garantir médias
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if df.empty: return False, None, {}
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Filtro de Data (Evita dados velhos)
        if (datetime.now() - df.index[-1].to_pydatetime()).days > 5:
            return False, None, {}

        # Indicadores
        df['SMA200'] = SMAIndicator(df['Close'], window=200).sma_indicator()
        df['SMA50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
        
        df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
        df['ADX'] = ADXIndicator(df['High'], df['Low'], df['Close'], window=14).adx()
        df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
        
        # Volume Média 20
        df['Vol_SMA20'] = df['Volume'].rolling(window=20).mean()

        atual = df.iloc[-1]

        # Regras de Setup
        tendencia = (atual['Close'] > atual['SMA200']) and (atual['Close'] > atual['SMA50'])
        forca = atual['ADX'] > 20
        # Ajuste Fino V8 no RSI
        pullback = (atual['RSI'] < 65) and (atual['RSI'] > 35)

        aprovado = tendencia and forca and pullback

        # Features V8 (Calcula Volume Projetado)
        vol_ratio = get_volume_projetado(atual['Volume'], atual['Vol_SMA20'])

        features = {
            "preco_entrada": float(atual['Close']),
            "rsi": float(atual['RSI']),
            "adx": float(atual['ADX']),
            "atr_absoluto": float(atual['ATR']),
            "distancia_sma200_pct": float((atual['Close'] - atual['SMA200']) / atual['SMA200']) * 100,
            "volume_ratio_projetado": float(vol_ratio),
            "dia_semana": df.index[-1].weekday()
        }

        return aprovado, df, features

    except Exception as e:
        print(f"Erro no screener ({ticker}): {e}")
        return False, None, {}

# --- 2. FERRAMENTAS IA ---
@tool("News Search")
def search_news(query: str):
    """Busca notícias recentes."""
    if DDGS is None: return "Erro: Lib DDGS ausente."
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, region='br-pt', max_results=3))
        return str(results) if results else "Sem notícias relevantes."
    except Exception as e:
        return f"Erro busca: {str(e)}"

# --- 3. AGENTES ---
MODELO = "gemini/gemini-2.0-flash"

analista = Agent(
    role='Risk Analyst',
    goal='Filtrar riscos graves.',
    backstory='Você protege o capital identificando riscos de cauda.',
    tools=[search_news],
    llm=MODELO,
    verbose=True
)

manager = Agent(
    role='Portfolio Manager',
    goal='Tomar decisão final de trade.',
    backstory='Você decide a entrada técnica validada pelo risco.',
    llm=MODELO,
    verbose=True
)

# --- 4. TAREFAS ---
t_risco = Task(
    description='Busque notícias urgentes de {ticket} e filtre ruídos.',
    expected_output='Resumo de riscos.',
    agent=analista
)

t_manager = Task(
    description='''
    Ativo: {ticket}. Preço Técnico: {price}.
    Analise o risco reportado. Decida COMPRA ou CANCELAR.
    
    Retorne JSON estrito:
    {{
        "ticker": "{ticket}",
        "decisao": "COMPRA" ou "CANCELAR",
        "entrada": float,
        "stop": float,
        "alvo": float,
        "motivo": "resumo curto"
    }}''',
    expected_output='JSON Válido.',
    agent=manager,
    context=[t_risco]
)

equipe = Crew(
    agents=[analista, manager],
    tasks=[t_risco, t_manager],
    process=Process.sequential
)

# --- 5. REGISTRO (Data Warehouse) ---
def registrar_trade(sinal, qtd_acoes):
    historico = []
    if os.path.exists(CAMINHO_TRADES):
        try:
            with open(CAMINHO_TRADES, "r") as f:
                historico = json.load(f)
        except: pass
    
    # Evita duplicatas do dia
    hoje = datetime.now().strftime("%Y-%m-%d")
    for t in historico:
        if t['ticker'] == sinal['ticker'] and t['data'].startswith(hoje):
            return 

    novo_trade = {
        "data": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": sinal['ticker'],
        "entrada": sinal['entrada'],
        "stop": sinal['stop'],
        "alvo": sinal['alvo'],
        "qtd_acoes": qtd_acoes, # Nova feature V8
        "status": "ABERTO",
        "resultado_financeiro": 0.0,
        "resultado_pct": 0.0,
        "features_tecnicas": sinal.get('features_ml', {})
    }
    
    historico.append(novo_trade)
    
    with open(CAMINHO_TRADES, "w") as f:
        json.dump(historico, f, indent=4)

def enviar_alerta(sinal, qtd, risco_est):
    if not bot: return
    
    emoji = "🟢"
    ft = sinal.get('features_ml', {})
    
    msg = f"""
🦅 **ROBÔ V8 - SINAL CONFIRMADO**
🚀 **COMPRA:** `{sinal['ticker']}` {emoji}

💰 **Entrada:** `R$ {sinal['entrada']}`
📦 **Lote:** `{qtd} ações`
⚠️ **Risco Est.:** `R$ {risco_est:.2f}`

🛑 **Stop:** {sinal['stop']}
🏁 **Alvo:** {sinal['alvo']}

📉 **Dados V8:**
• Vol Proj: {ft.get('volume_ratio_projetado', 0):.2f}x
• RSI: {ft.get('rsi', 0):.1f}
    """
    try:
        bot.send_message(TELEGRAM_CHAT_ID, msg, parse_mode="Markdown")
    except Exception as e:
        print(f"Erro Telegram: {e}")

def rodar_robo():
    print("--- INICIANDO ROBÔ V8 (BLINDADO) ---")
    
    # Lista Definitiva (Blue Chips + Liquidez)
    ativos = ["WEGE3.SA", "VALE3.SA", "PETR4.SA", "ITUB4.SA", "PRIO3.SA", "CSNA3.SA", "SUZB3.SA", "GGBR4.SA"]
    
    if os.path.exists(CAMINHO_CARTEIRA):
        with open(CAMINHO_CARTEIRA, "w") as f: json.dump(ativos, f)
        
    for ticker in ativos:
        # --- BLINDAGEM V8: Limpeza de Variáveis ---
        # Garante que dados do ativo anterior não contaminem o atual
        preco_real_agora = None
        sinal = None
        
        print(f"\n🔎 Analisando {ticker}...")
        aprovado, df, features = validar_setup_v8(ticker)
        
        if aprovado:
            print(f"✅ {ticker} Aprovado no Filtro V8.")
            inputs = {'ticket': ticker, 'price': f"{features['preco_entrada']:.2f}"}
            
            try:
                print("⏳ Aguardando delay API...")
                time.sleep(15)
                
                resultado = equipe.kickoff(inputs=inputs)
                
                raw_out = getattr(resultado, 'raw', str(resultado))
                texto_limpo = raw_out.replace('```json', '').replace('```', '').strip()
                sinal = json.loads(texto_limpo)
                
                if sinal['decisao'] == "COMPRA":
                    # --- V8 SNIPER CHECK (BLINDADO) ---
                    print("🔄 Buscando preço em tempo real...")
                    try:
                        ticker_obj = yf.Ticker(ticker)
                        # Força download fresco
                        hist = ticker_obj.history(period="1d")
                        if not hist.empty:
                            preco_real_agora = float(hist['Close'].iloc[-1])
                        else:
                            raise Exception("Dados vazios no refresh")
                            
                        print(f"📉 Preço IA: {sinal['entrada']} -> Preço REAL: {preco_real_agora:.2f}")
                        
                        # Atualiza entrada com dado fresco
                        sinal['entrada'] = round(preco_real_agora, 2)
                        
                    except Exception as e:
                        print(f"❌ FALHA CRÍTICA NO PREÇO ({ticker}): {e}")
                        print("⚠️ TRADE ABORTADO POR SEGURANÇA.")
                        continue # Pula o trade se não confirmar o preço. NÃO USA O ANTIGO.

                    # --- GESTÃO DE RISCO V8 ---
                    qtd = calcular_posicao(sinal['entrada'], sinal['stop'])
                    risco_total = (sinal['entrada'] - sinal['stop']) * qtd
                    
                    sinal['features_ml'] = features
                    # Garante que o ticker gravado é o do loop atual
                    sinal['ticker'] = ticker 
                    
                    enviar_alerta(sinal, qtd, risco_total)
                    registrar_trade(sinal, qtd)
                    print(f"🚀 Ordem Executada: {ticker}")
                else:
                    print(f"❌ {ticker} vetado pela IA.")
                    
            except Exception as e:
                print(f"Erro Crítico: {e}")
        else:
            print(f"⏹️ {ticker} Neutro.")
            
    print("--- FIM DA ROTINA V8 ---")

if __name__ == "__main__":
    rodar_robo()