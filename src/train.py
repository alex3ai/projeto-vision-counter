import os
import yaml
from ultralytics import YOLO

# --- Configurações de Caminho (SRE Approach: Absolute Paths) ---
# Pega o diretório raiz onde o script está rodando
BASE_DIR = os.path.abspath(os.getcwd())

# Define onde estão os dados EXATAMENTE
DATASET_DIR = os.path.join(BASE_DIR, "data", "processed")
TEMP_YAML_PATH = os.path.join(BASE_DIR, "config", "data_run.yaml")

MODEL_NAME = "yolo11n.pt"

def create_dynamic_config():
    """Gera um YAML com caminhos absolutos para evitar erros de FileNotfound."""
    config = {
        'path': DATASET_DIR,        # Caminho absoluto da raiz do dataset
        'train': 'images/train',    # Relativo ao 'path' acima
        'val': 'images/val',        # Relativo ao 'path' acima
        'nc': 1,                    # Número de classes
        'names': {0: 'objeto'}      # Nome das classes
    }
    
    print(f"⚙️  Gerando configuração dinâmica em: {TEMP_YAML_PATH}")
    print(f"📂 Apontando dados para: {DATASET_DIR}")
    
    with open(TEMP_YAML_PATH, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return TEMP_YAML_PATH

def main():
    print(f"🚀 Iniciando pipeline de treinamento com {MODEL_NAME}...")
    
    # 1. Resolver conflito de caminhos criando config sob medida
    data_config_path = create_dynamic_config()

    # 2. Carregar Modelo
    model = YOLO(MODEL_NAME)

    # 3. Treinar
    try:
        results = model.train(
            data=data_config_path,
            epochs=50,
            imgsz=640,
            batch=4,
            device="cpu",
            project="models",
            name="custom_counter",
            exist_ok=True
        )
        print("\n✅ Treinamento concluído com sucesso!")
        print(f"💾 Modelo salvo em: {results.save_dir}/weights/best.pt")
        
    except Exception as e:
        print(f"\n❌ Erro durante o treinamento: {e}")
    
    finally:
        # Limpeza (opcional): remove o yaml temporário
        # if os.path.exists(data_config_path): os.remove(data_config_path)
        pass

if __name__ == "__main__":
    main()