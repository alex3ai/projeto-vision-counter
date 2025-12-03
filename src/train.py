import os
import sys
from ultralytics import YOLO

def train_model():
    print("🚀 Iniciando Protocolo de Treinamento...")
    
    # --- LÓGICA DE CAMINHOS BLINDADA ---
    current_script_path = os.path.abspath(__file__)
    src_dir = os.path.dirname(current_script_path)
    project_root = os.path.dirname(src_dir)
    
    # 2. Caminho correto do config (Você acertou aqui!)
    yaml_path = os.path.join(project_root, "config", "data.yaml")
    
    print(f"📂 Diretório Raiz identificado: {project_root}")
    print(f"📄 Tentando carregar config em: {yaml_path}")
    
    if not os.path.exists(yaml_path):
        print(f"❌ ERRO CRÍTICO: O Python jura que o arquivo não está lá: {yaml_path}")
        return

    # 3. Carrega o modelo
    model = YOLO("yolo11n.pt")
    
    # 4. Inicia o Treino
    results = model.train(
        data=yaml_path,
        epochs=80,
        batch=4,
        
        # --- ONDE SALVAR (CRUCIAL - FALTAVA ISSO) ---
        project=os.path.join(project_root, "models"), 
        name="custom_counter",
        exist_ok=True,  # Sobrescreve para não criar custom_counter2, 3...
        
        # --- AUGMENTATION ---
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.2,
        scale=0.5,
        fliplr=0.5,
        mosaic=0.0,
        mixup=0.0,
        lr0=0.001,
        warmup_epochs=5,
        patience=20,
        box=7.5,
        cls=0.5
    )
    
    # Agora sim o print fala a verdade
    print("\n✅ Treinamento Finalizado!")
    print(f"💾 Modelo salvo em: {os.path.join(project_root, 'models', 'custom_counter', 'weights', 'best.pt')}")

if __name__ == "__main__":
    train_model()