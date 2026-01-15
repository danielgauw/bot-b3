import yfinance as yf
import pandas as pd
import json
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, ADXIndicator

# --- CONFIGURAÇÃO ---
CAPITAL_INICIAL = 10000.0
RISCO_POR_TRADE = 0.02 # 2%
DATA_INICIO = "2023-01-01"

def executar_backtest_otimizado():
    print(f"--- BACKTEST V2: OTIMIZADO ({DATA_INICIO}) ---")
    
    try:
        with open("carteira_alvo.json", "r") as f:
            ativos = json.load(f)
    except:
        print("Erro: Gere a carteira_alvo.json primeiro.")
        return

    trades_log = []
    
    for ticker in ativos:
        # print(f"Analisando {ticker}...") # Comentei para limpar o terminal
        try:
            df = yf.download(ticker, start=DATA_INICIO, progress=False)
            if df.empty: continue
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # INDICADORES OTIMIZADOS
            df['SMA200'] = SMAIndicator(df['Close'], window=200).sma_indicator()
            df['SMA50'] = SMAIndicator(df['Close'], window=50).sma_indicator() # Tendência Média
            df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
            
            # ADX - Mede a força da tendência (Evita mercado lateral)
            adx = ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
            df['ADX'] = adx.adx()

            posicionado = False
            preco_entrada = 0.0
            stop_loss = 0.0
            take_profit = 0.0
            dias = 0
            
            for i in range(200, len(df)):
                hoje = df.iloc[i]
                
                if not posicionado:
                    # REGRA 1: Tendência Sólida (Preço > SMA200 E SMA50)
                    tendencia = (hoje['Close'] > hoje['SMA200']) and (hoje['Close'] > hoje['SMA50'])
                    
                    # REGRA 2: Tendência Forte (ADX > 20) - O segredo para evitar falsos rompimentos
                    forca = hoje['ADX'] > 20
                    
                    # REGRA 3: Pullback Controlado (RSI não pode estar morto < 30, tem que estar reagindo)
                    # Entramos quando RSI está entre 30 e 55
                    pullback = (hoje['RSI'] < 55) and (hoje['RSI'] > 35)
                    
                    if tendencia and forca and pullback:
                        preco_entrada = hoje['Close']
                        
                        # Stop Loss Otimizado: 2.5% fixo ou mínima recente (o que for menor)
                        # Isso evita stops muito curtos que violam por ruído
                        stop_loss = preco_entrada * 0.96 # 4% de stop (Dá espaço pro preço respirar)
                        
                        risk = preco_entrada - stop_loss
                        take_profit = preco_entrada + (risk * 2.0) # Alvo 2x1 (diminuí um pouco para acertar mais)
                        
                        posicionado = True
                        dias = 0
                
                else:
                    dias += 1
                    low = hoje['Low']
                    high = hoje['High']
                    
                    sair = False
                    res = 0
                    
                    if low <= stop_loss:
                        res = -1
                        sair = True
                    elif high >= take_profit:
                        res = 2
                        sair = True
                    elif dias > 15: # Time Stop mais curto
                        sair = True
                        risk = preco_entrada - stop_loss
                        res = (hoje['Close'] - preco_entrada) / risk
                    
                    if sair:
                        trades_log.append({"res": res})
                        posicionado = False

        except: continue
            
    # RESULTADOS
    df_res = pd.DataFrame(trades_log)
    if df_res.empty: return

    wins = len(df_res[df_res['res'] > 0])
    total = len(df_res)
    win_rate = (wins/total)*100
    
    # Simulação Financeira
    saldo = CAPITAL_INICIAL
    risco_reais = CAPITAL_INICIAL * RISCO_POR_TRADE
    for r in df_res['res']:
        saldo += (r * risco_reais)
        
    lucro = saldo - CAPITAL_INICIAL
    rent = (lucro / CAPITAL_INICIAL) * 100
    
    print("\n" + "="*40)
    print("RESULTADO V2 (COM FILTRO DE TENDÊNCIA ADX)")
    print("="*40)
    print(f"Total Trades: {total}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Rentabilidade: {rent:.2f}%")
    print(f"Capital Final: R$ {saldo:.2f}")
    
    if win_rate > 45:
        print("✅ SINAL VERDE: Acurácia aceitável. A IA agora fará o resto.")
    else:
        print("⚠️ AINDA ARRISCADO: Precisamos de stops mais longos.")

if __name__ == "__main__":
    executar_backtest_otimizado()