import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

NOME_DO_VIDEO = r'C:\Users\jozef\Documents\Programacao\pendulo\video.mp4'
NOME_CSV_SAIDA = 'dados_pendulo.csv'

cap = cv2.VideoCapture(NOME_DO_VIDEO)

if not cap.isOpened():
    print(f"Erro ao abrir o vídeo: {NOME_DO_VIDEO}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or np.isnan(fps):
    fps = 30.0 

print(f"Vídeo carregado com sucesso. FPS: {fps:.2f}")

dados_trajetoria = []
frame_idx = 0

fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    t = frame_idx / fps

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)

    fgmask = fgbg.apply(gray_blur)
    
    kernel = np.ones((5, 5), np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x_centro, y_centro = None, None

    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        if area > 100:
            M = cv2.moments(c)
            if M["m00"] != 0:
                x_centro = float(M["m10"] / M["m00"])
                y_centro = float(M["m01"] / M["m00"])

    if x_centro is not None and y_centro is not None:
        dados_trajetoria.append({
            'frame': frame_idx,
            't': round(t, 4),
            'x': round(x_centro, 2),
            'y': round(y_centro, 2)
        })

    frame_idx += 1

cap.release()

df = pd.DataFrame(dados_trajetoria)
df.to_csv(NOME_CSV_SAIDA, index=False)
print(f"Processamento concluído! {len(df)} frames rastreados.")
print(f"Tabela salva em: {NOME_CSV_SAIDA}\n")

from scipy.signal import find_peaks

x_med = df['x'].mean()
df['x_cent'] = df['x'] - x_med

t_data = df['t'].values
x_data = df['x_cent'].values

picos_idx, _ = find_peaks(x_data, distance=int(fps*1.0), prominence=10)

t_picos = t_data[picos_idx]
x_picos = x_data[picos_idx]

#A(t) = A0 * exp(-gamma * t)
def envelope_exp(t, A0, gamma):
    return A0 * np.exp(-gamma * t)

A0_chute = x_picos[0] if len(x_picos) > 0 else x_data.max()
gamma_chute = 0.005

try:
    popt_env, _ = curve_fit(envelope_exp, t_picos, x_picos, p0=[A0_chute, gamma_chute])
    A0_fit, gamma_fit = popt_env

    periodos = np.diff(t_picos)
    T_medio = np.mean(periodos)
    omega_fit = 2 * np.pi / T_medio

    #Q = omega / (2 * gamma)
    Q = abs(omega_fit) / (2 * abs(gamma_fit))

    print("=== RESULTADOS CORRIGIDOS DO AJUSTE DE FÍSICA ===")
    print(f"Amplitude Inicial (A0): {abs(A0_fit):.2f} pixels")
    print(f"Coeficiente de Amortecimento (gamma): {abs(gamma_fit):.5f} s^-1")
    print(f"Frequência Angular (omega): {abs(omega_fit):.4f} rad/s")
    print(f"Período de Oscilação (T): {T_medio:.3f} segundos")
    print(f"\n")
    print(f"FATOR DE QUALIDADE (Q): {Q:.2f}\n")

    plt.figure(figsize=(10, 5))
    plt.plot(t_data, x_data, 'b.', alpha=0.3, markersize=2, label='Dados Rastreados (Pixels)')
    plt.plot(t_picos, x_picos, 'ro', markersize=4, label='Picos Registados')
    
    t_smooth = np.linspace(t_data.min(), t_data.max(), 1000)
    envelope = abs(A0_fit) * np.exp(-abs(gamma_fit) * t_smooth)

    plt.plot(t_smooth, envelope, 'k--', linewidth=2, label=f'Envelope Exp. [Q = {Q:.1f}]')
    plt.plot(t_smooth, -envelope, 'k--', linewidth=2)

    plt.title('Movimento do Pêndulo Simples - Decaimento da Amplitude')
    plt.xlabel('Tempo t (s)')
    plt.ylabel('Posição Horizontal x (pixels)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig('grafico_pendulo_corrigido.png', dpi=300)
    print("Novo gráfico gerado e salvo como 'grafico_pendulo_corrigido.png'.")
    plt.show()

except Exception as e:
    print(f"Erro ao calcular o envelope dos picos: {e}")