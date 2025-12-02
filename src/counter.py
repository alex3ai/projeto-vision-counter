"""
Vision Counter - Versão V5 (Linha Diagonal)
Suporte para linha inclinada perpendicular ao fluxo
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# --- CONFIGURAÇÕES DE PRODUÇÃO ---
VIDEO_PATH = "data/raw/video.mp4"
MODEL_PATH = "models/custom_counter/weights/best.pt"

# 📏 GEOMETRIA DA LINHA (Ajuste a inclinação aqui)
# 0.0 = Topo, 1.0 = Fundo
LINE_LEFT_Y  = 0.19  # Esquerda mais ALTA (0.40 = 40% da tela)
LINE_RIGHT_Y = 0.25  # Direita mais BAIXA (0.60 = 60% da tela)

LINE_MARGIN = 13     # Tamanho da zona de cruzamento (em pixels)

# Parâmetros de IA
CONFIDENCE_THRESHOLD = 0.15 #0.35 padrao
IOU_THRESHOLD = 0.65
TRACKER_CONFIG = "bytetrack.yaml"  # ou botsort.yaml # padrão: bytetrack.yaml

class DiagonalCounter:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        self.track_history = defaultdict(lambda: [])
        self.track_states = {}  # 'above', 'in_zone', 'below'
        self.counted_ids = set()
        self.count = 0
        
    def get_line_y_at_x(self, x, width, height):
        """
        Calcula a altura Y da linha diagonal em uma posição X específica.
        Matemática: y = mx + b (Equação da reta)
        """
        y_start = height * LINE_LEFT_Y
        y_end = height * LINE_RIGHT_Y
        
        # Inclinação (slope)
        slope = (y_end - y_start) / width
        
        # Y na posição X atual
        return int(y_start + (slope * x))

    def get_track_state(self, cx, cy, width, height, margin):
        """Determina se o objeto está acima, abaixo ou na linha (considerando a inclinação)"""
        
        # Descobre qual a altura da linha EXATAMENTE onde a laranja está (no eixo X)
        line_y_at_x = self.get_line_y_at_x(cx, width, height)
        
        if cy < line_y_at_x - margin:
            return 'above'
        elif cy > line_y_at_x + margin:
            return 'below'
        else:
            return 'in_zone'
    
    def check_crossing(self, track_id, current_state):
        """Verifica a transição de estado para contar"""
        if track_id not in self.track_states:
            self.track_states[track_id] = current_state
            return False
        
        prev_state = self.track_states[track_id]
        crossed = False
        
        # Lógica de Cruzamento: De cima para baixo
        # (Se a esteira fosse de baixo pra cima, inverteria aqui)
        if prev_state == 'above' and current_state == 'below':
            crossed = True
        elif prev_state == 'above' and current_state == 'in_zone':
            pass # Entrou na zona, aguardando sair
        elif prev_state == 'in_zone' and current_state == 'below':
            crossed = True # Saiu da zona por baixo -> Conta
            
        # Atualiza estado
        self.track_states[track_id] = current_state
        return crossed
    
    def run(self):
        cap = cv2.VideoCapture(VIDEO_PATH)
        
        # Info inicial
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"🚀 Iniciando Diagonal Counter: {orig_w}x{orig_h} @ {fps:.1f}fps")
        print(f"📐 Linha: Esquerda={LINE_LEFT_Y*100}% -> Direita={LINE_RIGHT_Y*100}%")
        
        frame_idx = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                # Loop infinito para demonstração
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            frame_idx += 1
            
            # Redimensionamento (Otimização)
            target_h = 640
            scale = target_h / orig_h
            new_w = int(orig_w * scale)
            frame = cv2.resize(frame, (new_w, target_h))
            
            # --- DESENHO DA LINHA DIAGONAL ---
            # Coordenadas das pontas
            pt1 = (0, int(target_h * LINE_LEFT_Y))       # Esquerda
            pt2 = (new_w, int(target_h * LINE_RIGHT_Y))  # Direita
            
            # Desenha a "Zona de Margem" (Linhas amarelas)
            # Offset vertical simples para visualização
            cv2.line(frame, (0, pt1[1]-LINE_MARGIN), (new_w, pt2[1]-LINE_MARGIN), (0, 255, 255), 1)
            cv2.line(frame, (0, pt1[1]+LINE_MARGIN), (new_w, pt2[1]+LINE_MARGIN), (0, 255, 255), 1)
            # Linha Principal (Vermelha)
            cv2.line(frame, pt1, pt2, (0, 0, 255), 2)
            
            # --- INFERÊNCIA ---
            results = self.model.track(
                frame, 
                persist=True, 
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                tracker=TRACKER_CONFIG,
                verbose=False
            )
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.int().cpu().tolist()
                
                for box, track_id in zip(boxes, ids):
                    x1, y1, x2, y2 = box
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    
                    # 1. Obter Estado (com base na linha diagonal)
                    state = self.get_track_state(cx, cy, new_w, target_h, LINE_MARGIN)
                    
                    # 2. Verificar Cruzamento
                    if self.check_crossing(track_id, state):
                        if track_id not in self.counted_ids:
                            self.count += 1
                            self.counted_ids.add(track_id)
                            # Efeito visual de cruzamento
                            cv2.line(frame, pt1, pt2, (0, 255, 0), 4)
                            print(f"🍊 Count: {self.count} | ID: {track_id}")

                    # --- VISUALIZAÇÃO ---
                    color = (0, 255, 0) if track_id in self.counted_ids else (0, 165, 255)
                    
                    # BBox e Centro
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)
                    
                    # Rastro
                    self.track_history[track_id].append((cx, cy))
                    if len(self.track_history[track_id]) > 20:
                        self.track_history[track_id].pop(0)
                        
                    points = np.hstack(self.track_history[track_id]).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)

                    # Label
                    cv2.putText(frame, str(track_id), (int(x1), int(y1)-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Placar
            cv2.rectangle(frame, (0, 0), (250, 80), (0,0,0), -1)
            cv2.putText(frame, f"TOTAL: {self.count}", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            cv2.imshow("Vision Counter V5 (Diagonal)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    DiagonalCounter().run()