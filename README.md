# 🍊 Vision Counter

> Sistema de contagem de objetos em esteiras industriais usando YOLO11 + ByteTrack

![Demo](data/raw/resultado_final.gif)

---

## 💡 O Problema

Contar laranjas em uma esteira industrial em alta velocidade parece simples, mas não é:

- **Motion blur:** Objetos ficam borrados a 2m/s
- **Frames pulados:** CPU lenta não processa tudo
- **Contagem duplicada:** Mesmo objeto contado 2-3 vezes

Este projeto resolve esses problemas com uma abordagem focada em **performance e simplicidade**.

---

## 🎯 Como Funciona

```
Vídeo → YOLO11 (detecção) → ByteTrack (rastreamento) → Linha Diagonal → Contador
```

### Principais Técnicas

**1. Linha Diagonal Inteligente**
- Não uso linha horizontal (conta objetos parados)
- Linha diagonal segue a perspectiva da câmera
- Verifica transição de estado: `Above → In_Zone → Below`

**2. Otimizações de Performance**
- Resolução adaptativa (1080p → 480p para inferência)
- Frame skipping inteligente (processa 1 a cada 2-3 frames)

**3. Dataset automatizado**
- 80 frames rotulados com auto_label
- Foco em laranjas borradas (motion blur)
- Heavy data augmentation (rotação, HSV, scale)

---

## 📊 Performance

### Resultados do Modelo

Treinado por **80 épocas** com YOLO11-Nano:

| Métrica | Valor | Observação |
|---------|-------|------------|
| **Precision** | 75% | 3 em 4 detecções são corretas |
| **Recall** | 95% | Detecta 95% das laranjas |
| **mAP50** | 83% | Ótimo para produção |

---

## 🚀 Como Usar

### Instalação

```bash
git clone https://github.com/alex3ai/projeto-vision-counter.git
cd projeto-vision-counter

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
```

### Executar

```bash
# Contador otimizado para CPU (recomendado)
python src/counter.py
```

**Saída esperada:**
```
📹 Vídeo: 1920x1080 @ 30fps
⚡ Resolução de processamento: 480px
🎯 Linha de contagem: 19% → 25%

🍊 #1 | ID:42 | Frame:0089
🍊 #2 | ID:51 | Frame:0142
...
📊 Total: 247 laranjas
⏱️  Tempo: 3min 12s
```

---

## 📂 Estrutura do Projeto

```
vision-counter/
├── data/
│   ├── raw/                    # Vídeos originais
│   └── data_manual/            # Dataset anotado (80 frames)
├── models/
│   └── custom_counter/
│       └── weights/
│           ├── best.pt         # Modelo treinado
│           └── results.csv     # Métricas de treino
├── src/
│   ├── auto_label_yolo_world.py   # Gerador de dataset
│   ├── counter.py                 # Contador
│   └── train.py                   # Treinamento
└── requirements.txt
```

---

## 🔧 Reproduzir o Treino

Se quiser treinar do zero:

```bash
# 1. Gerar dataset (80 frames com auto-labeling)
python src/auto_label_yolo_world.py

# 2. Treinar modelo (80 épocas, ~30min em Colab GPU)
python src/train.py

# 3. Testar
python src/counter.py
```

**Configurações de treino:**
- Batch size: 4 (limitado por VRAM)
- Augmentation: Rotation ±15°, HSV variation
- Early stopping: Patience de 20 épocas

---

## 🧠 O Que Aprendi

### Desafios Técnicos

1. **CPU vs GPU é MUITO diferente**
   - PyTorch em CPU é lento demais (4 FPS)
   - ONNX + resolução baixa melhora a perfomance de execução do projeto em (20 FPS)

2. **Data augmentation importa mais que modelo grande**
   - YOLO11-Nano (pequeno) funciona bem com bom dataset
   - 80 frames bem rotulados (auto-label) > 500 frames com labels ruins

3. **Tracking é essencial**
   - Sem ByteTrack: conta a mesma laranja 5x
   - Com ByteTrack: zero duplicatas

### Otimizações que Funcionaram

✅ Redimensionar para 480p (-75% de pixels)  
✅ Frame skipping (processa 1 a cada 2-3)  
✅ ONNX export (2.5x speedup em CPU)  
✅ Linha diagonal (elimina falsos positivos)

### Otimizações que NÃO Funcionaram
  
❌ Tentar rodar 1080p em CPU (FPS <5)  
❌ Usar YOLO11x (grande demais, nenhum ganho prático)

---

## 🐛 Problemas Conhecidos

- **Lag em CPU fraca:** Use `googleColab` ao invés do real-time
- **Contagem duplicada:** Aumente `CONFIDENCE_THRESHOLD` para 0.40+
- **Perde objetos muito rápidos:** Reduza `PROCESS_EVERY_N_FRAMES` para 1

---

## 📝 Próximos Passos

- [ ] Exportar para TensorRT (testar em Jetson Nano)
- [ ] Adicionar API REST para integração
- [ ] Testar em outros tipos de objetos
- [ ] Dashboard web com contagens em tempo real

---

## 🛠️ Tecnologias

- **YOLO11-Nano** (Ultralytics) - Detecção de objetos
- **ByteTrack** - Rastreamento multi-objeto
- **OpenCV** - Processamento de vídeo

---

## 📄 Licença

MIT License - use como quiser!

---

## 👤 Alex Oliveira Mendes

Projeto desenvolvido como estudo de Computer Vision e otimização de performance.

Se tiver dúvidas ou sugestões, abra uma issue!

---

**⭐ Se este projeto te ajudou, deixa uma estrela no GitHub!**