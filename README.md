# Vision Counter One-Shot

Pipeline de contagem de objetos usando One-Shot Learning com SAM 2 e inferência via YOLO11n.

## Estrutura
- **src/annotate.py**: Gera labels automáticos usando SAM 2.
- **src/train.py**: Fine-tuning do YOLO11n.
- **src/counter.py**: Inferência e contagem de linha.

## Como rodar
1. Coloque o vídeo em `data/raw/video.mp4`.
2. Execute `python src/annotate.py` para gerar o dataset.
3. Execute `python src/train.py` para treinar o modelo.
4. Execute `python src/counter.py` para ver a mágica.