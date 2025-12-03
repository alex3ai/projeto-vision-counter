import cv2
import os
import glob

# Configurações
VIDEO_PATH = "data/raw/video.mp4"
DATASET_DIR = "data_manual"
NUM_FRAMES_TO_LABEL = 80  # Vamos rotular apenas 10 frames

# Variáveis de Mouse
drawing = False
ix, iy = -1, -1
bboxes = []

def ensure_dirs():
    for d in ["train/images", "train/labels", "val/images", "val/labels"]:
        os.makedirs(os.path.join(DATASET_DIR, d), exist_ok=True)

def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, bboxes
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = param.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Anotador Manual", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(param, (ix, iy), (x, y), (0, 255, 0), 2)
        # Salva coordenadas x1, y1, x2, y2
        bboxes.append([min(ix, x), min(iy, y), max(ix, x), max(iy, y)])
        cv2.imshow("Anotador Manual", param)

def save_annotation(filename, img_shape, boxes, subset="train"):
    # Salva txt YOLO
    h_img, w_img = img_shape[:2]
    txt_path = os.path.join(DATASET_DIR, subset, "labels", filename.replace(".jpg", ".txt"))
    
    with open(txt_path, "w") as f:
        for box in boxes:
            x1, y1, x2, y2 = box
            # Converter para YOLO (center_x, center_y, w, h)
            w = (x2 - x1) / w_img
            h = (y2 - y1) / h_img
            cx = (x1 + x2) / 2 / w_img
            cy = (y1 + y2) / 2 / h_img
            f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

def main():
    ensure_dirs()
    
    # 1. Extração de Frames (Espalhados pelo vídeo)
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = total_frames // NUM_FRAMES_TO_LABEL
    
    frames_data = []
    print(f"📸 Extraindo {NUM_FRAMES_TO_LABEL} frames...")
    
    for i in range(NUM_FRAMES_TO_LABEL):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if ret:
            # Resize mantendo proporção (altura 640)
            h, w = frame.shape[:2]
            new_h = 640
            new_w = int(w * (640/h))
            frame = cv2.resize(frame, (new_w, new_h))
            frames_data.append(frame)
    cap.release()

    # 2. Loop de Anotação
    print("\n--- INSTRUÇÕES ---")
    print("1. Desenhe caixas nas laranjas (clique e arraste).")
    print("2. Pegue laranjas BORRADAS também!")
    print("3. Pressione 'SPACE' para salvar e ir para próxima imagem.")
    print("4. Pressione 'R' para limpar a imagem atual se errar.")
    print("5. Pressione 'Q' para sair.")

    cv2.namedWindow("Anotador Manual")
    
    for idx, frame in enumerate(frames_data):
        global bboxes
        bboxes = []
        clean_frame = frame.copy()
        cv2.setMouseCallback("Anotador Manual", mouse_callback, frame)
        
        while True:
            cv2.imshow("Anotador Manual", frame)
            k = cv2.waitKey(1) & 0xFF
            
            if k == 32: # SPACE
                # Salva imagem e label
                subset = "val" if idx >= 70 else "train"
                fname = f"orange_{idx:03d}.jpg"
                
                img_path = os.path.join(DATASET_DIR, subset, "images", fname)
                cv2.imwrite(img_path, clean_frame)
                save_annotation(fname, frame.shape, bboxes, subset)
                print(f"✅ Salvo {fname} ({len(bboxes)} laranjas)")
                break
            elif k == ord('r'): # Reset
                frame = clean_frame.copy()
                bboxes = []
                cv2.setMouseCallback("Anotador Manual", mouse_callback, frame)
                print("🧹 Limpo")
            elif k == ord('q'):
                return

    cv2.destroyAllWindows()
    print("🎉 Dataset Pronto!")

if __name__ == "__main__":
    main()