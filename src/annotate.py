import os
import cv2
import numpy as np
import sys
from ultralytics import SAM

# --- Configurações ---
VIDEO_PATH = "data/raw/video.mp4"
DATASET_DIR = "data/processed"
# Usando SAM 2 Tiny (nativo da Ultralytics) - Otimizado para CPU/Edge
MODEL_TYPE = "sam2_t.pt" 

# Variável global para armazenar o clique
click_coords = None

def mouse_callback(event, x, y, flags, param):
    global click_coords
    if event == cv2.EVENT_LBUTTONDOWN:
        click_coords = (x, y)
        print(f"📍 Ponto capturado: x={x}, y={y}")

def ensure_dirs():
    dirs = [
        f"{DATASET_DIR}/images/train", f"{DATASET_DIR}/labels/train",
        f"{DATASET_DIR}/images/val", f"{DATASET_DIR}/labels/val"
    ]
    for d in dirs: os.makedirs(d, exist_ok=True)

def convert_ultralytics_mask_to_yolo(results, w_img, h_img):
    """Extrai a bbox do resultado da Ultralytics e converte para YOLO."""
    # O resultado da Ultralytics já traz as máscaras e bboxes
    if not results or not results[0].masks:
        return None
    
    # Pega o primeiro objeto detectado (supondo 1 clique = 1 objeto)
    x, y, w, h = results[0].boxes.xywh[0].tolist() # xywh retorna centro x,y e largura,altura
    
    # Normalizar para 0-1
    x_n = x / w_img
    y_n = y / h_img
    w_n = w / w_img
    h_n = h / h_img
    
    return f"0 {x_n:.6f} {y_n:.6f} {w_n:.6f} {h_n:.6f}"

def get_target_point(frame):
    global click_coords
    h, w = frame.shape[:2]
    
    print(f"\n--- Interface de Seleção ({w}x{h}) ---")
    print("🖱️  Clique no centro do objeto e pressione 'Q' para confirmar.")
    
    cv2.namedWindow("Selecione o Objeto")
    cv2.setMouseCallback("Selecione o Objeto", mouse_callback)
    
    while True:
        disp = frame.copy()
        if click_coords:
            # Desenha um alvo onde clicou
            cv2.circle(disp, click_coords, 5, (0, 0, 255), -1)
            cv2.drawMarker(disp, click_coords, (0,0,255), cv2.MARKER_CROSS, 20, 2)
        
        cv2.imshow("Selecione o Objeto", disp)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    if not click_coords:
        print("⚠️ Nenhum clique detectado. Usando centro da imagem.")
        return (w//2, h//2)
    return click_coords

def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Vídeo não encontrado: {VIDEO_PATH}")
        sys.exit(1)

    ensure_dirs()
    
    # 1. Carregar Vídeo
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    cap.release()
    if not ret: sys.exit("Erro ao ler frame.")

    # 2. Obter Clique
    tx, ty = get_target_point(frame)
    
    # 3. Inferência SAM 2 (Via Ultralytics)
    print(f"🤖 Carregando {MODEL_TYPE}...")
    model = SAM(MODEL_TYPE) # Baixa sozinho se não existir
    
    print("⏳ Processando máscara (pode levar alguns segundos na CPU)...")
    # A API da Ultralytics aceita 'bboxes' ou 'points'. Points deve ser [[x,y]]
    # Labels: 1 para foreground
    results = model.predict(frame, points=[[tx, ty]], labels=[1])
    
    # 4. Salvar Dados
    yolo_line = convert_ultralytics_mask_to_yolo(results, frame.shape[1], frame.shape[0])
    
    if yolo_line:
        # Salvar Frame
        cv2.imwrite(f"{DATASET_DIR}/images/train/img0.jpg", frame)
        cv2.imwrite(f"{DATASET_DIR}/images/val/img0.jpg", frame)
        
        # Salvar Label
        with open(f"{DATASET_DIR}/labels/train/img0.txt", "w") as f: f.write(yolo_line)
        with open(f"{DATASET_DIR}/labels/val/img0.txt", "w") as f: f.write(yolo_line)
        
        print(f"\n✅ Dataset gerado em {DATASET_DIR}")
        
        # Debug Visual
        res_plotted = results[0].plot()
        cv2.imwrite("debug_annotation.jpg", res_plotted)
        print("🖼️  Debug salvo: debug_annotation.jpg")
    else:
        print("❌ Falha ao gerar máscara.")

if __name__ == "__main__":
    main()