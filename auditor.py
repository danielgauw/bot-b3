import json
import yfinance as yf
import pandas as pd
import os
import requests
import telebot
from datetime import datetime, timedelta
from dotenv import load_dotenv

# --- INFRAESTRUTURA BLINDADA ---
DIRETORIO_BASE = os.path.dirname(os.path.abspath(__file__))
CAMINHO_ENV = os.path.join(DIRETORIO_BASE, '.env')
CAMINHO_TRADES = os.path.join(DIRETORIO_BASE, 'trades_simulados.json')
CAMINHO_HTML = os.path.join(DIRETORIO_BASE, 'dashboard.html')

load_dotenv(CAMINHO_ENV)

# --- CONFIGURAÇÕES ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
CAPITAL_INICIAL = 10000.00
APOSTA_POR_TRADE = 2000.00

# --- CHOQUE DE REALIDADE (TAXAS) ---
# 0.03% B3 + 0.07% Slippage estimado = 0.1% por ponta (0.001)
# Total Ida e Volta = 0.2% aprox.
TAXA_OPERACIONAL = 0.001 

bot = telebot.TeleBot(TELEGRAM_TOKEN)

# --- FUNÇÕES AUXILIARES ---
def get_cdi_acumulado(data_inicio):
    try:
        data_fmt = datetime.strptime(data_inicio, "%Y-%m-%d").strftime("%d/%m/%Y")
        hoje_fmt = datetime.now().strftime("%d/%m/%Y")
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.11/dados?formato=json&dataInicial={data_fmt}&dataFinal={hoje_fmt}"
        response = requests.get(url)
        dados = response.json()
        fator = 1.0
        for dia in dados:
            taxa = float(dia['valor']) / 100
            fator *= (1 + taxa/100)
        return (fator - 1) * 100
    except:
        return 0.5

def get_ibov_acumulado(data_inicio):
    try:
        ibov = yf.download("^BVSP", start=data_inicio, progress=False)
        if ibov.empty: return 0.0
        inicio = ibov['Close'].iloc[0].item()
        fim = ibov['Close'].iloc[-1].item()
        return ((fim - inicio) / inicio) * 100
    except:
        return 0.0

# --- GERADOR DE DASHBOARD ---
def gerar_html(stats, trades, benchmarks):
    cor_saldo = "#00ff88" if stats['lucro_liquido'] >= 0 else "#ff4d4d"
    
    chart_labels = json.dumps(["Início"] + [t['data'].split(' ')[0] for t in trades])
    chart_data = json.dumps([CAPITAL_INICIAL] + [CAPITAL_INICIAL + t['acumulado'] for t in trades])
    
    html = f"""
    <!DOCTYPE html>
    <html lang="pt-br">
    <head>
        <meta charset="UTF-8">
        <title>Auditoria V7.2 (Realista)</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
        <style>
            body {{ background-color: #0d1117; color: #c9d1d9; font-family: 'Inter', sans-serif; margin: 0; padding: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #30363d; padding-bottom: 20px; margin-bottom: 30px; }}
            .card {{ background: #161b22; padding: 20px; border-radius: 6px; border: 1px solid #30363d; }}
            .grid-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 30px; }}
            .value {{ font-size: 24px; font-weight: 600; margin-top: 10px; }}
            
            table {{ width: 100%; border-collapse: collapse; background: #161b22; border-radius: 6px; overflow: hidden; font-size: 13px; }}
            th {{ background: #21262d; color: #8b949e; text-align: left; padding: 12px; }}
            td {{ padding: 12px; border-bottom: 1px solid #30363d; vertical-align: middle; }}
            .tag {{ padding: 3px 8px; border-radius: 12px; font-size: 11px; font-weight: bold; }}
            .gain {{ background: rgba(0,255,136,0.15); color: #00ff88; }}
            .loss {{ background: rgba(255,77,77,0.15); color: #ff4d4d; }}
            .tech-data {{ font-family: 'Courier New', monospace; font-size: 11px; color: #8b949e; }}
            .obs {{ font-size: 10px; color: #ff4d4d; margin-top: 5px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div>
                    <h1>🦅 Auditoria Realista V7.2</h1>
                    <div class="obs">Considerando Custos B3 + Slippage (0.2% total)</div>
                </div>
                <span style="color: #8b949e">{datetime.now().strftime('%d/%m/%Y %H:%M')}</span>
            </div>

            <div class="grid-cards">
                <div class="card"><h3>Saldo Líquido</h3><div class="value" style="color: {cor_saldo}">R$ {stats['lucro_liquido']:.2f}</div></div>
                <div class="card"><h3>Win Rate</h3><div class="value">{stats['win_rate']:.0f}%</div></div>
                <div class="card"><h3>Rentabilidade</h3><div class="value">{stats['rentabilidade_pct']:.2f}%</div></div>
                <div class="card"><h3>CDI Ref.</h3><div class="value" style="color: #58a6ff">{benchmarks['cdi']:.2f}%</div></div>
            </div>

            <div class="card" style="height: 300px; margin-bottom: 30px;">
                <canvas id="equityCurve"></canvas>
            </div>

            <table>
                <thead>
                    <tr>
                        <th>Data</th>
                        <th>Ativo</th>
                        <th>Raio-X Técnico</th>
                        <th>Entrada / Saída</th>
                        <th>Res Liq %</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for t in reversed(trades):
        # Mostra o resultado LÍQUIDO (já descontado taxas)
        res_val = t.get('resultado_liquido_pct', 0)
        
        tag_class = "tag"
        if t['status'] == "GAIN": tag_class += " gain"
        elif t['status'] == "LOSS": tag_class += " loss"

        ft = t.get('features_tecnicas', {})
        raio_x = ""
        if ft:
            raio_x = f"""
            <div class="tech-data">
                RSI: <b>{ft.get('rsi', 0):.1f}</b> | Vol: <b>{ft.get('volume_ratio', 0):.2f}x</b><br>
                MM200: <b>{ft.get('distancia_sma200_pct', 0):.1f}%</b>
            </div>
            """
        else:
            raio_x = "<span class='tech-data'>--</span>"

        html += f"""
            <tr>
                <td>{t['data'].split(' ')[0]}</td>
                <td><b style="color: #58a6ff">{t['ticker']}</b></td>
                <td>{raio_x}</td>
                <td>Ent: {t['entrada']}<br>Sai: {float(t.get('preco_atual', 0)):.2f}</td>
                <td style="color: {'#00ff88' if res_val>=0 else '#ff4d4d'}">{res_val:.2f}%</td>
                <td><span class="{tag_class}">{t['status']}</span></td>
            </tr>
        """

    html += f"""
                </tbody>
            </table>
        </div>
        <script>
            const ctx = document.getElementById('equityCurve').getContext('2d');
            new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: {chart_labels},
                    datasets: [{{
                        label: 'Patrimônio Líquido',
                        data: {chart_data},
                        borderColor: '#58a6ff',
                        backgroundColor: 'rgba(88, 166, 255, 0.1)',
                        tension: 0.3, fill: true
                    }}]
                }},
                options: {{ maintainAspectRatio: false, plugins: {{ legend: {{ display: false }} }}, scales: {{ x: {{ display: false }}, y: {{ grid: {{ color: '#30363d' }} }} }} }}
            }});
        </script>
    </body>
    </html>
    """
    
    with open(CAMINHO_HTML, "w", encoding='utf-8') as f:
        f.write(html)
    return CAMINHO_HTML

# --- LÓGICA DE AUDITORIA ---
def auditar():
    print("--- AUDITORIA REALISTA V7.2 (COM CUSTOS) ---")
    
    if not os.path.exists(CAMINHO_TRADES):
        print("Arquivo de trades não encontrado.")
        return

    try:
        with open(CAMINHO_TRADES, "r") as f:
            trades = json.load(f)
    except:
        return

    if not trades: return

    data_inicio = trades[0]['data'].split(' ')[0]
    saldo_acumulado = 0
    vitorias = 0
    derrotas = 0
    trades_processados = []

    for trade in trades:
        ticker = trade['ticker']
        entrada = float(trade['entrada'])
        alvo = float(trade['alvo'])
        stop = float(trade['stop'])
        
        # Se já fechou, usa o dado salvo
        if trade['status'] != "ABERTO":
            # Aqui assumimos que trades antigos já calcularam custos ou deixamos como está
            # Para o futuro, o ideal é recalcular se tiver o preço de saída
            res_liquido = trade.get('resultado_liquido_financeiro', 0)
            # Se não tiver o campo novo (legado), usa o antigo
            if res_liquido == 0 and trade.get('resultado_pct', 0) != 0:
                 res_liquido = (trade.get('resultado_pct')/100) * APOSTA_POR_TRADE
            
            saldo_acumulado += res_liquido
            
            if trade['status'] == "GAIN": vitorias += 1
            elif trade['status'] == "LOSS": derrotas += 1
            
            trade['acumulado'] = saldo_acumulado
            trades_processados.append(trade)
            continue

        # Se está ABERTO, atualiza
        try:
            df = yf.download(ticker, period="5d", progress=False)
            if df.empty:
                trade['preco_atual'] = entrada
                trade['acumulado'] = saldo_acumulado
                trades_processados.append(trade)
                continue
                
            ultimo = df.iloc[-1]
            high = float(ultimo['High'].iloc[0]) if isinstance(ultimo['High'], pd.Series) else float(ultimo['High'])
            low = float(ultimo['Low'].iloc[0]) if isinstance(ultimo['Low'], pd.Series) else float(ultimo['Low'])
            close = float(ultimo['Close'].iloc[0]) if isinstance(ultimo['Close'], pd.Series) else float(ultimo['Close'])

            novo_status = "ABERTO"
            preco_saida = close

            if high >= alvo:
                novo_status = "GAIN"
                preco_saida = alvo
                vitorias += 1
            elif low <= stop:
                novo_status = "LOSS"
                preco_saida = stop
                derrotas += 1
            
            # --- CÁLCULO FINANCEIRO REALISTA ---
            # Resultado Bruto
            res_bruto_pct = ((preco_saida - entrada) / entrada)
            
            # Custos: Taxa na entrada + Taxa na saída
            # Simplificação: Subtraímos a taxa do percentual bruto
            # Se a taxa é 0.1% (0.001) por trade completo
            res_liquido_pct = res_bruto_pct - TAXA_OPERACIONAL
            
            # Resultado Financeiro
            res_financeiro_liquido = res_liquido_pct * APOSTA_POR_TRADE
            
            # Atualiza o trade
            trade['status'] = novo_status
            trade['preco_atual'] = preco_saida
            trade['resultado_pct'] = res_bruto_pct * 100 # Mantemos o bruto para referência
            trade['resultado_liquido_pct'] = res_liquido_pct * 100 # O que importa pro bolso
            trade['resultado_liquido_financeiro'] = res_financeiro_liquido
            
            if novo_status != "ABERTO": 
                saldo_acumulado += res_financeiro_liquido
            
            trade['acumulado'] = saldo_acumulado
            trades_processados.append(trade)

        except Exception as e:
            print(f"Erro {ticker}: {e}")
            trade['acumulado'] = saldo_acumulado
            trades_processados.append(trade)

    # Salva e Envia
    with open(CAMINHO_TRADES, "w") as f:
        json.dump(trades_processados, f, indent=4)

    total = vitorias + derrotas
    win_rate = (vitorias / total * 100) if total > 0 else 0
    patrimonio = CAPITAL_INICIAL + saldo_acumulado
    rentabilidade = ((patrimonio - CAPITAL_INICIAL) / CAPITAL_INICIAL) * 100
    
    stats = {
        "lucro_liquido": saldo_acumulado, 
        "win_rate": win_rate, 
        "rentabilidade_pct": rentabilidade,
        "patrimonio_final": patrimonio
    }
    benchmarks = {"cdi": get_cdi_acumulado(data_inicio), "ibov": 0.0}
    
    arquivo_final = gerar_html(stats, trades_processados, benchmarks)
    
    print("📤 Enviando Relatório Realista...")
    with open(arquivo_final, 'rb') as doc:
        caption = f"🦅 **Auditoria Realista**\n(Descontando custos B3/Slippage)\n\n💰 Líquido: R$ {saldo_acumulado:.2f}\n📊 Rentab.: {rentabilidade:.2f}%"
        bot.send_document(TELEGRAM_CHAT_ID, doc, caption=caption)

if __name__ == "__main__":
    auditar()
