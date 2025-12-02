import os
import sys
from ultralytics import YOLO

def train_model():
    print("🚀 Iniciando Protocolo de Treinamento...")
    
    # --- LÓGICA DE CAMINHOS BLINDADA ---
    # 1. Descobre onde este script (train.py) está
    current_script_path = os.path.abspath(__file__) # .../vision-counter/src/train.py
    src_dir = os.path.dirname(current_script_path)  # .../vision-counter/src
    project_root = os.path.dirname(src_dir)         # .../vision-counter
    
    # 2. Monta o caminho exato do data.yaml
    yaml_path = os.path.join(project_root, "config", "data.yaml")
    
    print(f"📂 Diretório Raiz identificado: {project_root}")
    print(f"📄 Tentando carregar config em: {yaml_path}")
    
    # Verificação de segurança antes de chamar o YOLO
    if not os.path.exists(yaml_path):
        print(f"❌ ERRO CRÍTICO: O Python jura que o arquivo não está lá: {yaml_path}")
        return

    # 3. Carrega o modelo
    model = YOLO("yolo11n.pt")
    
    # 4. Inicia o Treino
    results = model.train(
    data=yaml_path,
    epochs=70,  # Aumentar
    batch=4,
    augment=True,  # ATIVE ISSO
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=15.0,
    translate=0.2,
    scale=0.5,
    fliplr=0.5,
    mosaic=0.0,
    mixup=0.0,
    lr0=0.001,        # Learning rate menor
    warmup_epochs=5,  # Mais warmup
    patience=20,      # Mais paciência
    box=7.5,
    cls=0.5
)
    
    print("\n✅ Treinamento Finalizado!")
    print(f"💾 Modelo salvo em: {os.path.join(project_root, 'models', 'custom_counter', 'weights', 'best.pt')}")

if __name__ == "__main__":
    train_model()