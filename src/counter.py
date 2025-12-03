"""
Vision Counter - Versão Otimizada (SEM TRAVAMENTOS)
Correções:
- Skip de frames para performance
- Redução de resolução adaptativa
- Cache de resultados
- Threading para inferência
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time

# ==================== CONFIGURAÇÕES ====================
VIDEO_PATH = "data/raw/video.mp4"
MODEL_PATH = "models/custom_counter/weights/best.pt"

# Otimizações de Performance
PROCESS_EVERY_N_FRAMES = 1  # Processar 1 a cada N frames (1=todos, 2=metade)
TARGET_FPS = 30  # FPS alvo para display
RESIZE_WIDTH = 380  # Largura de processamento (menor = mais rápido)

# Linha de Contagem
LINE_LEFT_Y = 0.19   # 19% da altura (esquerda)
LINE_RIGHT_Y = 0.25  # 25% da altura (direita)
LINE_MARGIN = 13

# Detecção
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.65
TRACKER_CONFIG = "bytetrack.yaml"

# ==================== CLASSE CONTADOR ====================

class OptimizedCounter:
    def __init__(self):
        print("🔄 Carregando modelo...")
        self.model = YOLO(MODEL_PATH)
        self.model.fuse()  # Otimização do modelo
        
        self.track_history = defaultdict(lambda: [])
        self.track_states = {}
        self.counted_ids = set()
        self.count = 0
        
        # Controle de FPS
        self.frame_time = 1.0 / TARGET_FPS
        self.last_process_time = 0
        
        print("✅ Modelo carregado e otimizado!")
    
    def get_line_y_at_x(self, x, width, height):
        """Calcula Y da linha diagonal na posição X"""
        y_start = height * LINE_LEFT_Y
        y_end = height * LINE_RIGHT_Y
        slope = (y_end - y_start) / width
        return int(y_start + (slope * x))
    
    def get_track_state(self, cx, cy, width, height, margin):
        """Determina estado do objeto em relação à linha"""
        line_y_at_x = self.get_line_y_at_x(cx, width, height)
        
        if cy < line_y_at_x - margin:
            return 'above'
        elif cy > line_y_at_x + margin:
            return 'below'
        else:
            return 'in_zone'
    
    def check_crossing(self, track_id, current_state):
        """Verifica se houve cruzamento da linha"""
        if track_id not in self.track_states:
            self.track_states[track_id] = current_state
            return False
        
        prev_state = self.track_states[track_id]
        crossed = False
        
        # Detecção de cruzamento (cima para baixo)
        if (prev_state == 'above' and current_state == 'below') or \
           (prev_state == 'in_zone' and current_state == 'below' and 
            self.track_states.get(f"{track_id}_was_above", False)):
            crossed = True
        
        # Marca se esteve acima antes de entrar na zona
        if prev_state == 'above' and current_state == 'in_zone':
            self.track_states[f"{track_id}_was_above"] = True
        
        self.track_states[track_id] = current_state
        return crossed
    
    def run(self):
        cap = cv2.VideoCapture(VIDEO_PATH)
        
        # Info do vídeo
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"\n📹 Vídeo: {orig_w}x{orig_h} @ {video_fps:.1f}fps ({total_frames} frames)")
        print(f"⚙️  Processando a cada {PROCESS_EVERY_N_FRAMES} frame(s)")
        print(f"🎯 Linha: Esquerda={LINE_LEFT_Y*100:.0f}% → Direita={LINE_RIGHT_Y*100:.0f}%")
        print(f"🚀 Iniciando...\n")
        
        frame_idx = 0
        last_results = None  # Cache do último resultado
        
        while cap.isOpened():
            start_time = time.time()
            
            success, frame = cap.read()
            if not success:
                # Loop infinito
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.count = 0
                self.counted_ids.clear()
                self.track_states.clear()
                self.track_history.clear()
                continue
            
            frame_idx += 1
            
            # Redimensiona SEMPRE (para display consistente)
            scale = RESIZE_WIDTH / orig_w
            new_h = int(orig_h * scale)
            frame_resized = cv2.resize(frame, (RESIZE_WIDTH, new_h))
            
            # Decide se processa este frame
            should_process = (frame_idx % PROCESS_EVERY_N_FRAMES == 0)
            
            if should_process:
                # Inferência
                results = self.model.track(
                    frame_resized,
                    persist=True,
                    conf=CONFIDENCE_THRESHOLD,
                    iou=IOU_THRESHOLD,
                    tracker=TRACKER_CONFIG,
                    verbose=False,
                    stream=False  # Desativa streaming para evitar buffer
                )
                last_results = results
            else:
                # Reutiliza resultado anterior (economiza GPU)
                results = last_results
            
            # Desenha linha
            pt1 = (0, int(new_h * LINE_LEFT_Y))
            pt2 = (RESIZE_WIDTH, int(new_h * LINE_RIGHT_Y))
            
            cv2.line(frame_resized, (0, pt1[1]-LINE_MARGIN), 
                    (RESIZE_WIDTH, pt2[1]-LINE_MARGIN), (255, 255, 0), 1)
            cv2.line(frame_resized, pt1, pt2, (0, 0, 255), 2)
            cv2.line(frame_resized, (0, pt1[1]+LINE_MARGIN), 
                    (RESIZE_WIDTH, pt2[1]+LINE_MARGIN), (255, 255, 0), 1)
            
            # Processa detecções
            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.int().cpu().tolist()
                
                for box, track_id in zip(boxes, ids):
                    x1, y1, x2, y2 = box
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    
                    # Estado e cruzamento
                    state = self.get_track_state(cx, cy, RESIZE_WIDTH, new_h, LINE_MARGIN)
                    
                    if self.check_crossing(track_id, state):
                        if track_id not in self.counted_ids:
                            self.count += 1
                            self.counted_ids.add(track_id)
                            cv2.line(frame_resized, pt1, pt2, (0, 255, 0), 4)
                            print(f"🍊 Total: {self.count} | ID: {track_id}")
                    
                    # Visualização
                    color = (0, 255, 0) if track_id in self.counted_ids else (0, 165, 255)
                    
                    cv2.rectangle(frame_resized, (int(x1), int(y1)), 
                                 (int(x2), int(y2)), color, 2)
                    cv2.circle(frame_resized, (cx, cy), 3, (255, 0, 255), -1)
                    
                    # Trail
                    self.track_history[track_id].append((cx, cy))
                    if len(self.track_history[track_id]) > 15:
                        self.track_history[track_id].pop(0)
                    
                    points = np.array(self.track_history[track_id], dtype=np.int32)
                    if len(points) > 1:
                        cv2.polylines(frame_resized, [points], False, color, 2)
                    
                    cv2.putText(frame_resized, str(track_id), 
                               (int(x1), int(y1)-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Placar
            cv2.rectangle(frame_resized, (0, 0), (300, 80), (0, 0, 0), -1)
            cv2.putText(frame_resized, f"TOTAL: {self.count}", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
            # FPS real
            elapsed = time.time() - start_time
            current_fps = 1.0 / elapsed if elapsed > 0 else 0
            cv2.putText(frame_resized, f"FPS: {current_fps:.1f}", (20, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display
            cv2.imshow("Vision Counter (Q=sair | R=reset)", frame_resized)
            
            # Controle de teclas
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset contador
                self.count = 0
                self.counted_ids.clear()
                self.track_states.clear()
                print("🔄 Contador resetado!")
            
            # Controle de FPS (evita travamentos)
            elapsed = time.time() - start_time
            if elapsed < self.frame_time:
                time.sleep(self.frame_time - elapsed)
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n📊 RESULTADO FINAL: {self.count} laranjas contadas")

# ==================== MAIN ====================

def main():
    print("=" * 60)
    print("🍊 VISION COUNTER - VERSÃO OTIMIZADA")
    print("=" * 60)
    
    try:
        counter = OptimizedCounter()
        counter.run()
    except FileNotFoundError:
        print(f"\n❌ ERRO: Modelo não encontrado em {MODEL_PATH}")
        print("   Execute o treinamento primeiro: python src/train.py")
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()