import yfinance as yf
import pandas as pd
import json
import time

# CONFIGURAÇÃO
MINIMO_VOLUME = 20_000_000 # R$ 20 Milhões/dia

CANDIDATOS = [
    "VALE3.SA", "PETR4.SA", "PRIO3.SA", "WEGE3.SA", "ITUB4.SA", "BBDC4.SA", 
    "BBAS3.SA", "RENT3.SA", "LREN3.SA", "BPAC11.SA", "GGBR4.SA", 
    "CSNA3.SA", "JBSS3.SA", "SUZB3.SA", "RAIL3.SA", "RADL3.SA", "EQTL3.SA",
    "VBBR3.SA", "UGPA3.SA", "CMIG4.SA", "CPLE6.SA", "CSAN3.SA", "TOTS3.SA"
]

def gerar():
    print("--- FILTRANDO LIQUIDEZ ---")
    aprovados = []
    
    for ticker in CANDIDATOS:
        try:
            df = yf.download(ticker, period="60d", progress=False)
            if df.empty: continue
            
            # Tratamento seguro para fechar e volume
            try:
                vol = df['Volume'].iloc[:, 0] if isinstance(df['Volume'], pd.DataFrame) else df['Volume']
                close = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
            except:
                vol = df['Volume']
                close = df['Close']

            media_fin = (vol * close).mean()
            
            if media_fin > MINIMO_VOLUME:
                print(f"✅ {ticker}: Aprovado (R$ {media_fin/1_000_000:.1f}M)")
                aprovados.append(ticker)
            else:
                print(f"❌ {ticker}: Reprovado")
                
        except Exception as e:
            print(f"Erro {ticker}: {e}")
    
    with open("carteira_alvo.json", "w") as f:
        json.dump(aprovados, f)
    print("Arquivo 'carteira_alvo.json' gerado!")

if __name__ == "__main__":
    gerar()