"""
Auto-Label Profissional para Vision Counter
- Detecta até laranjas borradas/em movimento
- Filtra labels ruins automaticamente
- Salva em data_manual com split train/val
- Interface de revisão manual
"""

import cv2
import os
import sys
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# ==================== CONFIGURAÇÕES ====================
VIDEO_PATH = "data/raw/video.mp4"
OUTPUT_DIR = "data_manual"  # Agora salva no lugar certo!
CLASS_NAME = "orange"

# Parâmetros de Detecção
CONFIDENCE = 0.25  # Mais alto que 0.13, mais baixo que 0.35 (meio termo)
IOU_THRESHOLD = 0.5  # Remove sobreposições

# Parâmetros de Qualidade (FILTROS AUTOMÁTICOS)
MIN_DETECTIONS = 2   # Mínimo de laranjas por frame
MAX_DETECTIONS = 12  # Máximo (se passar disso, é ruído)
MIN_BOX_AREA = 0.005  # 0.5% da imagem (remove caixas minúsculas)
MAX_BOX_AREA = 0.15   # 15% da imagem (remove caixas gigantes)
MAX_OVERLAP_ALLOWED = 0.4  # Sobreposição máxima tolerada

# Amostragem
TOTAL_FRAMES_TO_EXTRACT = 80  # Quantos frames processar
VAL_SPLIT = 0.15  # 15% para validação (12 imgs), 85% train (68 imgs)

# ==================== FUNÇÕES AUXILIARES ====================

def ensure_dirs():
    """Cria estrutura de pastas em data_manual"""
    for sub in ["train/images", "train/labels", "val/images", "val/labels", "rejected"]:
        Path(OUTPUT_DIR, sub).mkdir(parents=True, exist_ok=True)
    print(f"✅ Estrutura criada em: {OUTPUT_DIR}")

def calculate_iou(box1, box2):
    """Calcula IoU entre duas caixas (formato: cx, cy, w, h normalizado)"""
    # Converte para x1, y1, x2, y2
    b1_x1, b1_y1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    b1_x2, b1_y2 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    
    b2_x1, b2_y1 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    b2_x2, b2_y2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2
    
    # Interseção
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    
    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def validate_detections(boxes_xywh_normalized):
    """
    Valida se as detecções são de boa qualidade
    Retorna: (is_valid, reason, filtered_boxes)
    """
    if len(boxes_xywh_normalized) < MIN_DETECTIONS:
        return False, f"Poucas detecções ({len(boxes_xywh_normalized)})", []
    
    if len(boxes_xywh_normalized) > MAX_DETECTIONS:
        return False, f"Muitas detecções ({len(boxes_xywh_normalized)})", []
    
    # Filtra por tamanho
    good_boxes = []
    for box in boxes_xywh_normalized:
        cx, cy, w, h = box
        area = w * h
        
        if area < MIN_BOX_AREA:
            continue  # Muito pequena
        if area > MAX_BOX_AREA:
            continue  # Muito grande
        
        good_boxes.append(box)
    
    if len(good_boxes) < MIN_DETECTIONS:
        return False, "Após filtro de tamanho, restaram poucas", []
    
    # Verifica sobreposições excessivas
    overlaps = 0
    total_pairs = 0
    
    for i in range(len(good_boxes)):
        for j in range(i+1, len(good_boxes)):
            iou = calculate_iou(good_boxes[i], good_boxes[j])
            total_pairs += 1
            if iou > 0.3:
                overlaps += 1
    
    if total_pairs > 0:
        overlap_ratio = overlaps / total_pairs
        if overlap_ratio > MAX_OVERLAP_ALLOWED:
            return False, f"Muita sobreposição ({overlap_ratio:.1%})", []
    
    return True, f"OK - {len(good_boxes)} laranjas válidas", good_boxes

def save_frame_with_labels(frame, boxes_normalized, filename, subset="train"):
    """
    Salva imagem e arquivo .txt com labels
    boxes_normalized: lista de [cx, cy, w, h] em coordenadas normalizadas (0-1)
    """
    # Caminhos
    img_path = Path(OUTPUT_DIR, subset, "images", f"{filename}.jpg")
    lbl_path = Path(OUTPUT_DIR, subset, "labels", f"{filename}.txt")
    
    # Salva imagem
    cv2.imwrite(str(img_path), frame)
    
    # Salva labels no formato YOLO
    with open(lbl_path, 'w') as f:
        for box in boxes_normalized:
            cx, cy, w, h = box
            f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    
    return True

def draw_detections_on_frame(frame, boxes_normalized, color=(0, 255, 0)):
    """Desenha as caixas no frame para preview"""
    h, w = frame.shape[:2]
    frame_copy = frame.copy()
    
    for box in boxes_normalized:
        cx, cy, bw, bh = box
        
        # Desnormaliza
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame_copy, (int(cx*w), int(cy*h)), 3, (255, 0, 255), -1)
    
    return frame_copy

# ==================== MAIN ====================

def main():
    print("=" * 70)
    print("🍊 AUTO-LABEL PROFISSIONAL - VISION COUNTER")
    print("=" * 70)
    
    # Verificações iniciais
    if not Path(VIDEO_PATH).exists():
        sys.exit(f"❌ Vídeo não encontrado: {VIDEO_PATH}")
    
    ensure_dirs()
    
    # Carrega modelo
    print("\n🧠 Carregando YOLO-World...")
    try:
        model = YOLO('yolov8s-world.pt')
        model.set_classes([CLASS_NAME])
        print(f"✅ Modelo carregado! Buscando: '{CLASS_NAME}'")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        print("💡 Baixando yolov8s-world.pt pela primeira vez...")
        return
    
    # Abre vídeo
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\n📹 Vídeo: {total_frames} frames @ {fps:.1f}fps")
    print(f"🎯 Extraindo {TOTAL_FRAMES_TO_EXTRACT} frames espaçados uniformemente")
    
    # Calcula step (pula frames para distribuir uniformemente)
    frame_step = max(1, total_frames // TOTAL_FRAMES_TO_EXTRACT)
    
    saved_count = 0
    rejected_count = 0
    frame_indices = range(0, total_frames, frame_step)
    
    print(f"⏭️  Processando 1 frame a cada {frame_step} frames\n")
    print("=" * 70)
    
    for idx, frame_num in enumerate(frame_indices):
        if saved_count >= TOTAL_FRAMES_TO_EXTRACT:
            break
        
        # Pula para o frame específico
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        success, frame = cap.read()
        
        if not success:
            continue
        
        # Redimensiona para 640p (padrão YOLO)
        h, w = frame.shape[:2]
        target_w = 640
        scale = target_w / w
        target_h = int(h * scale)
        frame_resized = cv2.resize(frame, (target_w, target_h))
        
        # Detecção com YOLO-World
        results = model.predict(
            frame_resized, 
            conf=CONFIDENCE,
            iou=IOU_THRESHOLD,
            verbose=False
        )
        
        # Extrai boxes normalizadas
        if len(results[0].boxes) == 0:
            continue
        
        boxes_xywh = results[0].boxes.xywh.cpu().numpy()  # Pixels
        h_img, w_img = frame_resized.shape[:2]
        
        # Normaliza (0-1)
        boxes_normalized = []
        for box in boxes_xywh:
            cx, cy, bw, bh = box
            boxes_normalized.append([
                cx / w_img,
                cy / h_img,
                bw / w_img,
                bh / h_img
            ])
        
        # Valida qualidade
        is_valid, reason, filtered_boxes = validate_detections(boxes_normalized)
        
        if not is_valid:
            rejected_count += 1
            print(f"❌ Frame {frame_num:05d}: {reason}")
            
            # Salva na pasta rejected para debug
            rejected_img = draw_detections_on_frame(frame_resized, boxes_normalized, color=(0,0,255))
            cv2.imwrite(str(Path(OUTPUT_DIR, "rejected", f"rejected_{idx:03d}.jpg")), rejected_img)
            continue
        
        # Determina se vai para train ou val
        subset = "val" if saved_count < int(TOTAL_FRAMES_TO_EXTRACT * VAL_SPLIT) else "train"
        
        # Salva
        filename = f"orange_{saved_count:03d}"
        save_frame_with_labels(frame_resized, filtered_boxes, filename, subset)
        
        saved_count += 1
        print(f"✅ {subset.upper()}: {filename} - {reason}")
        
        # Preview (opcional - comentar se quiser mais rápido)
        preview = draw_detections_on_frame(frame_resized, filtered_boxes)
        cv2.putText(preview, f"Salvos: {saved_count}/{TOTAL_FRAMES_TO_EXTRACT}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Auto-Label (Q para sair)", preview)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n⚠️  Interrompido pelo usuário")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Relatório final
    print("\n" + "=" * 70)
    print("📊 RELATÓRIO FINAL")
    print("=" * 70)
    
    train_count = len(list(Path(OUTPUT_DIR, "train/images").glob("*.jpg")))
    val_count = len(list(Path(OUTPUT_DIR, "val/images").glob("*.jpg")))
    
    print(f"✅ Frames salvos: {saved_count}")
    print(f"   └─ Train: {train_count} imagens")
    print(f"   └─ Val: {val_count} imagens")
    print(f"❌ Frames rejeitados: {rejected_count}")
    print(f"📂 Dataset salvo em: {OUTPUT_DIR}/")
    
    if saved_count < 30:
        print("\n⚠️  AVISO: Menos de 30 imagens!")
        print("   Recomendação:")
        print("   1. Abaixe CONFIDENCE para 0.20")
        print("   2. Ou aumente TOTAL_FRAMES_TO_EXTRACT para 120")
        print("   3. Ou complemente com rotulagem manual")
    else:
        print("\n🎉 Dataset pronto para treino!")
        print("\nPRÓXIMOS PASSOS:")
        print("   1. Revise algumas imagens em data_manual/rejected/")
        print("      (Se tiver laranjas boas rejeitadas, ajuste os parâmetros)")
        print("   2. Atualize config/data.yaml para apontar para data_manual")
        print("   3. Treine: python src/train.py")
        print("   4. Teste: python src/counter.py")

if __name__ == "__main__":
    main()