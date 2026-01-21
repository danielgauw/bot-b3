import json
import yfinance as yf
import pandas as pd
import os
import requests
import telebot
from datetime import datetime, timedelta
from dotenv import load_dotenv

# --- BLINDAGEM DE CAMINHO ABSOLUTO (CRÍTICO) ---
# Garante que o robô ache os arquivos independente de quem o execute (Cron, Usuário, etc)
DIRETORIO_BASE = os.path.dirname(os.path.abspath(__file__))
CAMINHO_ENV = os.path.join(DIRETORIO_BASE, '.env')
CAMINHO_TRADES = os.path.join(DIRETORIO_BASE, 'trades_simulados.json')
CAMINHO_HTML = os.path.join(DIRETORIO_BASE, 'dashboard.html')

# --- CONFIGURAÇÕES ---
load_dotenv(CAMINHO_ENV) # Carrega do caminho certo

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
CAPITAL_INICIAL = 10000.00  # Começamos com 10k fictícios
APOSTA_POR_TRADE = 2000.00  # Risco de 2k por operação

bot = telebot.TeleBot(TELEGRAM_TOKEN)

# --- FUNÇÕES DE DADOS DE MERCADO ---

def get_cdi_acumulado(data_inicio):
    """Busca CDI acumulado via API do Banco Central"""
    try:
        data_fmt = datetime.strptime(data_inicio, "%Y-%m-%d").strftime("%d/%m/%Y")
        hoje_fmt = datetime.now().strftime("%d/%m/%Y")
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.11/dados?formato=json&dataInicial={data_fmt}&dataFinal={hoje_fmt}"
        response = requests.get(url)
        dados = response.json()
        fator = 1.0
        for dia in dados:
            taxa = float(dia['valor']) / 100
            fator *= (1 + taxa/100) # CDI diário aproximado
        return (fator - 1) * 100
    except:
        return 0.5 # Fallback (0.5% se der erro)

def get_ibov_acumulado(data_inicio):
    """Busca retorno do IBOV no período"""
    try:
        ibov = yf.download("^BVSP", start=data_inicio, progress=False)
        if ibov.empty: return 0.0
        inicio = ibov['Close'].iloc[0].item()
        fim = ibov['Close'].iloc[-1].item()
        return ((fim - inicio) / inicio) * 100
    except:
        return 0.0

# --- GERADOR DE DASHBOARD HTML (LAYOUT PRO) ---

def gerar_html(stats, trades, benchmarks):
    cor_saldo = "#00ff88" if stats['lucro_liquido'] >= 0 else "#ff4d4d"
    
    # Dados para o Gráfico (JSON para o JS ler)
    chart_labels = json.dumps(["Início"] + [t['data'].split(' ')[0] for t in trades])
    chart_data = json.dumps([CAPITAL_INICIAL] + [CAPITAL_INICIAL + t['acumulado'] for t in trades])
    
    html = f"""
    <!DOCTYPE html>
    <html lang="pt-br">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Dashboard Robô Swing Trade</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
        <style>
            body {{ background-color: #121214; color: #e1e1e6; font-family: 'Inter', sans-serif; margin: 0; padding: 20px; }}
            .container {{ max-width: 1100px; margin: 0 auto; }}
            
            /* HEADER */
            .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; border-bottom: 1px solid #29292e; padding-bottom: 20px; }}
            .header h1 {{ font-size: 24px; margin: 0; background: linear-gradient(90deg, #00ff88, #00b3ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
            .date {{ color: #7c7c8a; font-size: 14px; }}

            /* CARDS */
            .grid-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
            .card {{ background: #202024; padding: 20px; border-radius: 8px; border: 1px solid #323238; }}
            .card h3 {{ font-size: 14px; color: #a8a8b3; margin: 0 0 10px 0; }}
            .card .value {{ font-size: 24px; font-weight: 600; }}
            
            /* BENCHMARKS BAR */
            .bench-bar {{ background: #202024; padding: 15px; border-radius: 8px; display: flex; justify-content: space-around; margin-bottom: 30px; font-size: 14px; }}
            .bench-item span {{ font-weight: bold; }}
            .positive {{ color: #00ff88; }}
            .negative {{ color: #ff4d4d; }}

            /* CHART AREA */
            .chart-container {{ background: #202024; padding: 20px; border-radius: 8px; margin-bottom: 30px; border: 1px solid #323238; height: 300px; }}

            /* TABLE */
            table {{ width: 100%; border-collapse: collapse; background: #202024; border-radius: 8px; overflow: hidden; }}
            th {{ background: #29292e; color: #a8a8b3; font-weight: 600; text-align: left; padding: 16px; font-size: 14px; }}
            td {{ padding: 16px; border-bottom: 1px solid #323238; font-size: 14px; }}
            tr:last-child td {{ border-bottom: none; }}
            .status-tag {{ padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
            .tag-gain {{ background: rgba(0, 255, 136, 0.1); color: #00ff88; }}
            .tag-loss {{ background: rgba(255, 77, 77, 0.1); color: #ff4d4d; }}
            .tag-open {{ background: rgba(255, 166, 0, 0.1); color: orange; }}

        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>⚡ Dashboard Performance</h1>
                <span class="date">Atualizado: {datetime.now().strftime('%d/%m/%Y %H:%M')}</span>
            </div>

            <div class="grid-cards">
                <div class="card">
                    <h3>Patrimônio Simulado</h3>
                    <div class="value" style="color: {cor_saldo}">R$ {stats['patrimonio_final']:.2f}</div>
                </div>
                <div class="card">
                    <h3>Lucro Líquido</h3>
                    <div class="value" style="color: {cor_saldo}">R$ {stats['lucro_liquido']:.2f}</div>
                </div>
                <div class="card">
                    <h3>Win Rate</h3>
                    <div class="value">{stats['win_rate']:.0f}%</div>
                </div>
                <div class="card">
                    <h3>Trades Totais</h3>
                    <div class="value">{len(trades)}</div>
                </div>
            </div>

            <div class="bench-bar">
                <div class="bench-item">🤖 Robô: <span class="{ 'positive' if stats['rentabilidade_pct'] > 0 else 'negative' }">{stats['rentabilidade_pct']:.2f}%</span></div>
                <div class="bench-item">📉 IBOV: <span class="{ 'positive' if benchmarks['ibov'] > 0 else 'negative' }">{benchmarks['ibov']:.2f}%</span></div>
                <div class="bench-item">🏦 CDI (Renda Fixa): <span class="positive">{benchmarks['cdi']:.2f}%</span></div>
            </div>

            <div class="chart-container">
                <canvas id="equityCurve"></canvas>
            </div>

            <table>
                <thead>
                    <tr>
                        <th>Data</th>
                        <th>Ativo</th>
                        <th>Entrada</th>
                        <th>Preço Atual / Saída</th>
                        <th>Resultado %</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for t in reversed(trades):
        res_val = t['resultado_pct']
        cor_res = "positive" if res_val >= 0 else "negative"
        tag_class = "tag-open"
        if t['status'] == "GAIN": tag_class = "tag-gain"
        if t['status'] == "LOSS": tag_class = "tag-loss"

        html += f"""
            <tr>
                <td>{t['data'].split(' ')[0]}</td>
                <td><strong>{t['ticker']}</strong></td>
                <td>R$ {float(t['entrada']):.2f}</td>
                <td>R$ {float(t['preco_atual']):.2f}</td>
                <td class="{cor_res}">{res_val:.2f}%</td>
                <td><span class="status-tag {tag_class}">{t['status']}</span></td>
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
                        label: 'Patrimônio (R$)',
                        data: {chart_data},
                        borderColor: '#00ff88',
                        backgroundColor: 'rgba(0, 255, 136, 0.1)',
                        tension: 0.4,
                        fill: true
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{ legend: {{ display: false }} }},
                    scales: {{
                        y: {{ grid: {{ color: '#323238' }}, ticks: {{ color: '#a8a8b3' }} }},
                        x: {{ grid: {{ display: false }}, ticks: {{ color: '#a8a8b3' }} }}
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    # SALVA USANDO O CAMINHO ABSOLUTO
    with open(CAMINHO_HTML, "w", encoding='utf-8') as f:
        f.write(html)
    return CAMINHO_HTML

# --- LÓGICA PRINCIPAL DE AUDITORIA ---

def auditar():
    print("--- INICIANDO AUDITORIA V7 ---")
    
    # 1. Carregar Trades com Caminho Absoluto
    if not os.path.exists(CAMINHO_TRADES):
        print("Sem trades para auditar.")
        return

    with open(CAMINHO_TRADES, "r") as f:
        trades = json.load(f)

    if not trades: return

    # 1. Definir data de início para Benchmarks
    data_inicio_simulacao = trades[0]['data'].split(' ')[0]
    
    # 2. Atualizar Trades Abertos
    saldo_acumulado = 0
    vitorias = 0
    derrotas = 0
    
    trades_processados = []
    
    for trade in trades:
        ticker = trade['ticker']
        entrada = float(trade['entrada'])
        alvo = float(trade['alvo'])
        stop = float(trade['stop'])
        
        # Se já fechou, mantém o resultado
        if trade['status'] != "ABERTO":
            res_pct = trade.get('resultado_pct', 0)
            res_financeiro = (res_pct / 100) * APOSTA_POR_TRADE
            trade['acumulado'] = saldo_acumulado + res_financeiro
            saldo_acumulado += res_financeiro
            
            if trade['status'] == "GAIN": vitorias += 1
            if trade['status'] == "LOSS": derrotas += 1
            
            trades_processados.append(trade)
            continue

        # Se está aberto, atualiza preço
        try:
            df = yf.download(ticker, period="5d", progress=False)
            if df.empty:
                trade['preco_atual'] = entrada
                trade['acumulado'] = saldo_acumulado
                trades_processados.append(trade)
                continue

            # Dados do candle
            ultimo = df.iloc[-1]
            high = float(ultimo['High'].iloc[0]) if isinstance(ultimo['High'], pd.Series) else float(ultimo['High'])
            low = float(ultimo['Low'].iloc[0]) if isinstance(ultimo['Low'], pd.Series) else float(ultimo['Low'])
            close = float(ultimo['Close'].iloc[0]) if isinstance(ultimo['Close'], pd.Series) else float(ultimo['Close'])

            novo_status = "ABERTO"
            preco_saida = close

            # Verifica Gain/Loss
            if high >= alvo:
                novo_status = "GAIN"
                preco_saida = alvo
                vitorias += 1
            elif low <= stop:
                novo_status = "LOSS"
                preco_saida = stop
                derrotas += 1
            
            # Calcula resultado
            res_pct = ((preco_saida - entrada) / entrada) * 100
            res_financeiro = (res_pct / 100) * APOSTA_POR_TRADE
            
            trade['status'] = novo_status
            trade['preco_atual'] = preco_saida
            trade['resultado_pct'] = res_pct
            
            if novo_status != "ABERTO":
                saldo_acumulado += res_financeiro
            
            trade['acumulado'] = saldo_acumulado # Saldo flutuante
            trades_processados.append(trade)

        except Exception as e:
            print(f"Erro em {ticker}: {e}")
            trade['acumulado'] = saldo_acumulado
            trades_processados.append(trade)

    # 3. Salvar JSON Atualizado (Caminho Absoluto)
    with open(CAMINHO_TRADES, "w") as f:
        json.dump(trades_processados, f, indent=4)

    # 4. Calcular Estatísticas
    total_fechados = vitorias + derrotas
    win_rate = (vitorias / total_fechados * 100) if total_fechados > 0 else 0
    patrimonio_final = CAPITAL_INICIAL + saldo_acumulado
    rentabilidade_robo = ((patrimonio_final - CAPITAL_INICIAL) / CAPITAL_INICIAL) * 100

    stats = {
        "lucro_liquido": saldo_acumulado,
        "patrimonio_final": patrimonio_final,
        "win_rate": win_rate,
        "rentabilidade_pct": rentabilidade_robo,
        "em_aberto": len(trades) - total_fechados
    }

    # 5. Obter Benchmarks
    benchmarks = {
        "cdi": get_cdi_acumulado(data_inicio_simulacao),
        "ibov": get_ibov_acumulado(data_inicio_simulacao)
    }

    # 6. Gerar Dashboard e Enviar
    arquivo_gerado = gerar_html(stats, trades_processados, benchmarks)
    
    print("📤 Enviando para Telegram...")
    with open(arquivo_gerado, 'rb') as doc:
        bot.send_document(TELEGRAM_CHAT_ID, doc, caption=f"📊 **Fechamento Diário**\n\n💰 Saldo: R$ {saldo_acumulado:.2f}\n📈 Rentabilidade: {rentabilidade_robo:.2f}% (CDI: {benchmarks['cdi']:.2f}%)")
    
    print("Sucesso!")

if __name__ == "__main__":
    auditar()
