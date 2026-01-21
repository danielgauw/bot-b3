import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv

# --- INFRAESTRUTURA BLINDADA ---
DIRETORIO_BASE = os.path.dirname(os.path.abspath(__file__))
CAMINHO_ENV = os.path.join(DIRETORIO_BASE, '.env')
CAMINHO_TRADES = os.path.join(DIRETORIO_BASE, 'trades_simulados.json')
CAMINHO_CARTEIRA = os.path.join(DIRETORIO_BASE, 'carteira_alvo.json')

load_dotenv(CAMINHO_ENV)

# Bibliotecas de Dados
import yfinance as yf
import pandas as pd
import numpy as np
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
if os.getenv("GOOGLE_API_KEY"):
    os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

bot = telebot.TeleBot(TELEGRAM_TOKEN)

# --- 1. O HARD SCREEN & FEATURE ENGINEERING (O CÉREBRO MATEMÁTICO) ---
def validar_setup_v2(ticker):
    """
    Retorna:
    1. Aprovado (Bool): Se passou no filtro básico.
    2. DF (DataFrame): Os dados brutos.
    3. Features (Dict): A 'Foto' técnica do mercado para auditoria/ML.
    """
    try:
        # Baixa dados suficientes para médias longas e cálculo de volume
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        if df.empty: return False, None, {}
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Filtro de Data (Evita dados velhos)
        if (datetime.now() - df.index[-1].to_pydatetime()).days > 5:
            return False, None, {}

        # --- CÁLCULO DE INDICADORES TÉCNICOS ---
        # 1. Tendência
        df['SMA200'] = SMAIndicator(df['Close'], window=200).sma_indicator()
        df['SMA50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
        
        # 2. Momentum & Força
        df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
        adx = ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
        df['ADX'] = adx.adx()
        
        # 3. Volatilidade (Risco)
        atr = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14)
        df['ATR'] = atr.average_true_range()
        
        # 4. Volume (Liquidez)
        # Preenche NaN com 0 para não quebrar cálculo
        df['Volume'] = df['Volume'].fillna(0)
        df['Vol_SMA20'] = df['Volume'].rolling(window=20).mean()

        # Dados do último candle fechado
        atual = df.iloc[-1]

        # --- REGRAS DE FILTRO (GATEKEEPER) ---
        # Tendência de alta clássica (Preço acima das médias)
        tendencia = (atual['Close'] > atual['SMA200']) and (atual['Close'] > atual['SMA50'])
        
        # Força da tendência (Evita mercado lateral morto)
        forca = atual['ADX'] > 20
        
        # Pullback saudável (Evita comprar topo eufórico > 70 ou faca caindo < 30)
        pullback = (atual['RSI'] < 65) and (atual['RSI'] > 35)

        aprovado = tendencia and forca and pullback

        # --- FEATURE ENGINEERING (A FOTO DO MERCADO) ---
        # Aqui capturamos TUDO que estava acontecendo no momento do trade
        # Esses dados são vitais para entender POR QUE o robô errou ou acertou.
        
        try:
            vol_ratio = float(atual['Volume'] / atual['Vol_SMA20']) if atual['Vol_SMA20'] > 0 else 0.0
        except:
            vol_ratio = 0.0

        features = {
            "preco_entrada": float(atual['Close']),
            "rsi": float(atual['RSI']),
            "adx": float(atual['ADX']),
            "atr_absoluto": float(atual['ATR']),
            "atr_percentual": float(atual['ATR'] / atual['Close']) * 100, # Volatilidade em %
            
            # Distância das Médias (Mean Reversion Risk)
            # Se for muito alto (ex: > 15%), risco de correção é iminente
            "distancia_sma200_pct": float((atual['Close'] - atual['SMA200']) / atual['SMA200']) * 100,
            "distancia_sma50_pct": float((atual['Close'] - atual['SMA50']) / atual['SMA50']) * 100,
            
            # Volume Ratio
            # < 1.0 = Volume abaixo da média (Fraco)
            # > 1.5 = Volume forte (Institucional)
            "volume_ratio": vol_ratio,
            
            # Contexto Temporal
            "dia_semana": df.index[-1].weekday(), # 0=Seg, 4=Sex
            "mes": df.index[-1].month
        }

        return aprovado, df, features

    except Exception as e:
        print(f"Erro no screener ({ticker}): {e}")
        return False, None, {}

# --- 2. FERRAMENTA DE BUSCA ---
@tool("News Search")
def search_news(query: str):
    """Busca notícias recentes."""
    if DDGS is None: return "Erro: Biblioteca DDGS ausente."
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, region='br-pt', max_results=3))
        if not results: return "Sem notícias relevantes."
        return str(results)
    except Exception as e:
        return f"Erro busca: {str(e)}"

# --- 3. AGENTES (IA) ---
MODELO_IA = "gemini/gemini-2.0-flash"

analista_risco = Agent(
    role='Risk Manager',
    goal='Identificar notícias de alto risco (falências, corrupção, quedas bruscas).',
    backstory='Você protege o capital. Na dúvida, veta.',
    tools=[search_news],
    llm=MODELO_IA,
    verbose=True
)

manager = Agent(
    role='Portfolio Manager',
    goal='Validar entrada técnica com base no risco.',
    backstory='Você recebe o sinal técnico e as notícias. Decide o trade.',
    llm=MODELO_IA,
    verbose=True
)

# --- 4. TAREFAS ---
t_risco = Task(
    description='Busque notícias urgentes de {ticket}.',
    expected_output='Resumo de riscos.',
    agent=analista_risco
)

t_manager = Task(
    description='''O ativo {ticket} tem setup técnico de COMPRA.
    Dados Técnicos: Preço {price}, ATR {atr}.
    Analise o risco das notícias.
    Retorne JSON:
    {{
        "ticker": "{ticket}",
        "decisao": "COMPRA" ou "CANCELAR",
        "entrada": float,
        "stop": float,
        "alvo": float,
        "confianca": "ALTA" ou "MEDIA",
        "motivo": "string curta"
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

# --- 5. REGISTRO DE TRADES (DATA WAREHOUSE) ---
def registrar_trade(sinal):
    historico = []
    
    # Carrega histórico com segurança
    if os.path.exists(CAMINHO_TRADES):
        try:
            with open(CAMINHO_TRADES, "r") as f:
                historico = json.load(f)
        except:
            pass # Se arquivo estiver corrompido, cria novo
    
    # Evita duplicatas do dia
    hoje = datetime.now().strftime("%Y-%m-%d")
    for trade in historico:
        if trade['ticker'] == sinal['ticker'] and trade['data'].startswith(hoje):
            return 

    # Estrutura de Dados Enriquecida para ML Futuro
    novo_trade = {
        "data": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": sinal['ticker'],
        # Dados Operacionais
        "entrada": sinal['entrada'],
        "stop": sinal['stop'],
        "alvo": sinal['alvo'],
        "status": "ABERTO",
        "resultado_financeiro": 0.0,
        "resultado_pct": 0.0,
        
        # Metadados da Decisão
        "confianca": sinal['confianca'],
        "motivo_ia": sinal.get('motivo', 'N/A'),
        
        # A CAIXA PRETA (Dados Técnicos para Análise de Falha/Sucesso)
        "features_tecnicas": sinal.get('features_ml', {})
    }
    
    historico.append(novo_trade)
    
    with open(CAMINHO_TRADES, "w") as f:
        json.dump(historico, f, indent=4)
        
    print(f"📝 Trade Registrado (Com dados técnicos): {sinal['ticker']}")

# --- 6. TELEGRAM & EXECUÇÃO ---
def enviar_alerta(sinal):
    if not bot: return
    emoji = "🟢" if sinal.get('confianca') == "ALTA" else "🟡"
    ft = sinal.get('features_ml', {})
    
    # Adicionamos dados técnicos no alerta para você ver na hora
    msg = f"""
🚀 **SINAL: {sinal.get('ticker')}**
📊 **Decisão:** `COMPRA` {emoji}

💰 **Entrada:** `R$ {sinal.get('entrada')}`
🛑 **Stop:** `R$ {sinal.get('stop')}`
🏁 **Alvo:** `R$ {sinal.get('alvo')}`

📉 **Dados da Caixa Preta:**
• RSI: {ft.get('rsi', 0):.1f} (Ideal: 35-60)
• Vol Ratio: {ft.get('volume_ratio', 0):.2f}x (Ideal: >1.0)
• Dist. MM200: {ft.get('distancia_sma200_pct', 0):.1f}%

📝 **Motivo IA:** {sinal.get('motivo')}
    """
    try:
        bot.send_message(TELEGRAM_CHAT_ID, msg, parse_mode="Markdown")
    except Exception as e:
        print(f"Erro Telegram: {e}")

def rodar_robo():
    print("--- INICIANDO ROBÔ V7.1 (PRODUÇÃO & COLETA DE DADOS) ---")
    
    if not os.path.exists(CAMINHO_CARTEIRA):
        # Fallback de segurança: cria carteira padrão se não existir
        with open(CAMINHO_CARTEIRA, "w") as f:
            json.dump(["WEGE3.SA", "VALE3.SA", "PETR4.SA", "ITUB4.SA", "PRIO3.SA"], f)
            
    with open(CAMINHO_CARTEIRA, "r") as f:
        carteira = json.load(f)
        
    for ticker in carteira:
        print(f"\n🔎 Analisando {ticker}...")
        aprovado, df, features_tecnicas = validar_setup_v2(ticker)
        
        if aprovado:
            print(f"✅ {ticker} Aprovado no Filtro Quantitativo.")
            
            inputs = {
                'ticket': ticker, 
                'atr': f"{features_tecnicas['atr_absoluto']:.2f}",
                'price': f"{features_tecnicas['preco_entrada']:.2f}"
            }
            
            try:
                print("⏳ Aguardando 20s (API Rate Limit)...")
                time.sleep(20)
                
                resultado = equipe.kickoff(inputs=inputs)
                
                # Tratamento robusto de saída da IA
                raw_out = getattr(resultado, 'raw', str(resultado))
                texto_limpo = raw_out.replace('```json', '').replace('```', '').strip()
                sinal = json.loads(texto_limpo)
                
                if sinal['decisao'] == "COMPRA":
                    # INJETA OS DADOS TÉCNICOS NO SINAL PARA GRAVAÇÃO
                    sinal['features_ml'] = features_tecnicas
                    
                    print(f"🚀 COMPRA CONFIRMADA: {ticker}")
                    enviar_alerta(sinal)
                    registrar_trade(sinal)
                else:
                    print(f"❌ {ticker} vetado pelo Risk Manager.")
                    
            except Exception as e:
                print(f"Erro Crítico na IA ou JSON: {e}")
        else:
            print(f"⏹️ {ticker} Reprovado no filtro técnico.")
            
    print("--- FIM DA ROTINA ---")

if __name__ == "__main__":
    rodar_robo()