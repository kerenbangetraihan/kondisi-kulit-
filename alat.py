import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
import os
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime
import pandas as pd

class Config:
    MODEL_PATH = "E:/DATASET/skin_texture_analysis_80160/output/model/best_modeel.h5"
    CLASSES = ['lembut', 'normal', 'kasar']
    CLASS_DISPLAY = {'lembut': 'Lembut', 'normal': 'Normal', 'kasar': 'Kasar'}
    IMG_SIZE = (200, 200) 
    
    CANNY_THRESH1 = 50     
    CANNY_THRESH2 = 100    
   
    ROI_SIZE = 200        
    MIN_SIMILARITY = 0.65 

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   
    edges = cv2.Canny(gray, 
                     Config.CANNY_THRESH1, 
                     Config.CANNY_THRESH2)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    processed = cv2.resize(edges_rgb, Config.IMG_SIZE)
    processed = processed.astype('float32') / 255.0
    
    return processed, edges 

class SkinTextureApp(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Klasifikasi Tekstur Kulit dengan Canny Edge")
        self.geometry("1200x800")
        self.configure(bg="#f0f0f0")
        
        self.cap = None
        self.current_frame = None
        self.captured_img = None
        self.model = None
        self.results = []

        self.init_ui()
        self.init_camera()
        self.load_model()
        
    def init_ui(self):
        style = ttk.Style()
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabel', background='#f0f0f0', font=('Helvetica', 10))
        style.configure('TButton', font=('Helvetica', 10))
        
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.cam_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.cam_tab, text="Kamera")
        self.setup_camera_tab()
        
        self.preprocess_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.preprocess_tab, text="Preprocessing")
        self.setup_preprocess_tab()
        
        self.result_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.result_tab, text="Hasil Klasifikasi")
        self.setup_result_tab()
    
    def setup_camera_tab(self):
       
        main_frame = ttk.Frame(self.cam_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.cam_label = ttk.Label(left_panel)
        self.cam_label.pack(fill=tk.BOTH, expand=True)
        
        right_panel = ttk.Frame(main_frame, width=300)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.capture_btn = ttk.Button(right_panel, text="Ambil Gambar",
                                     command=self.capture_image)
        self.capture_btn.pack(fill=tk.X, pady=5)

        self.analyze_btn = ttk.Button(right_panel, text="Analisis Edge",
                                    command=self.analyze_image,
                                    state=tk.DISABLED)
        self.analyze_btn.pack(fill=tk.X, pady=5)
        
        self.classify_btn = ttk.Button(right_panel, text="Klasifikasi",
                                     command=self.classify_image,
                                     state=tk.DISABLED)
        self.classify_btn.pack(fill=tk.X, pady=5)
        
        result_frame = ttk.LabelFrame(right_panel, text="Hasil Prediksi")
        result_frame.pack(fill=tk.X, pady=10)
        
        self.prediction_label = ttk.Label(result_frame, text="Kelas: -",
                                        font=('Helvetica', 12, 'bold'))
        self.prediction_label.pack(pady=5)
        
        self.confidence_label = ttk.Label(result_frame, text="Keyakinan: -")
        self.confidence_label.pack(pady=5)
        
        self.edge_info_label = ttk.Label(result_frame, text="Edge Density: -")
        self.edge_info_label.pack(pady=5)
    
    def setup_preprocess_tab(self):
        main_frame = ttk.Frame(self.preprocess_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
        titles = ["Original", "Grayscale", "Canny Edge", "Final (Untuk Model)"]
        self.preprocess_labels = []
        
        for i, title in enumerate(titles):
            frame = ttk.LabelFrame(main_frame, text=title)
            frame.grid(row=0, column=i, padx=5, pady=5, sticky="nsew")
            label = ttk.Label(frame)
            label.pack()
            self.preprocess_labels.append(label)
            main_frame.columnconfigure(i, weight=1)
        
        info_frame = ttk.Frame(main_frame)

        info_frame.grid(row=1, column=0, columnspan=4, pady=10, sticky="ew")
        
        self.preprocess_info = tk.Text(info_frame, height=6, wrap=tk.WORD,
                                     font=('Helvetica', 9))
        self.preprocess_info.pack(fill=tk.BOTH, expand=True)
        
        control_frame = ttk.Frame(main_frame)

        control_frame.grid(row=2, column=0, columnspan=4, pady=5)
        
        ttk.Label(control_frame, text="Canny Threshold:").pack(side=tk.LEFT)
        
        self.thresh1_slider = tk.Scale(control_frame, from_=0, to=200, 
                                      orient=tk.HORIZONTAL, 
                                      command=self.update_threshold)
        self.thresh1_slider.set(Config.CANNY_THRESH1)
        self.thresh1_slider.pack(side=tk.LEFT, padx=5)
        
        self.thresh2_slider = tk.Scale(control_frame, from_=0, to=300, 
                                      orient=tk.HORIZONTAL,
                                      command=self.update_threshold)
        self.thresh2_slider.set(Config.CANNY_THRESH2)
        self.thresh2_slider.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Reset", 
                 command=self.reset_threshold).pack(side=tk.LEFT)
    
    def setup_result_tab(self):
       
        main_frame = ttk.Frame(self.result_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.result_tree = ttk.Treeview(main_frame, 
                                      columns=("timestamp", "class", "confidence", "edge_density"),
                                      selectmode="browse")
        
        self.result_tree.heading("#0", text="ID")
        self.result_tree.heading("timestamp", text="Waktu")
        self.result_tree.heading("class", text="Kelas")
        self.result_tree.heading("confidence", text="Keyakinan")
        self.result_tree.heading("edge_density", text="Edge Density")
        
        self.result_tree.column("#0", width=50)
        self.result_tree.column("timestamp", width=150)
        self.result_tree.column("class", width=100)
        self.result_tree.column("confidence", width=100)
        self.result_tree.column("edge_density", width=100)
        
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", 
                                 command=self.result_tree.yview)
        self.result_tree.configure(yscrollcommand=scrollbar.set)
        
        self.result_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        btn_frame = ttk.Frame(self.result_tab)
        btn_frame.pack(fill=tk.Y, side=tk.RIGHT, padx=5)
        
        ttk.Button(btn_frame, text="Simpan Hasil", 
                 command=self.save_results).pack(pady=5)
        
        ttk.Button(btn_frame, text="Hapus Pilihan", 
                 command=self.delete_selected).pack(pady=5)
    
    def init_camera(self):
        self.cap = cv2.VideoCapture(2)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Tidak dapat mengakses kamera")
            return
        
        self.update_camera()
    
    def load_model(self):
        try:
            self.model = load_model(Config.MODEL_PATH)
            messagebox.showinfo("Sukses", "Model berhasil dimuat")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal memuat model: {str(e)}")
            self.destroy()
    
    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame.copy()
            
            h, w = frame.shape[:2]
            x1, y1 = (w - Config.ROI_SIZE) // 2, (h - Config.ROI_SIZE) // 2
            cv2.rectangle(frame, (x1, y1), 
                         (x1 + Config.ROI_SIZE, y1 + Config.ROI_SIZE), 
                         (0, 255, 0), 2)
            
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(image=img)
            
            self.cam_label.configure(image=img)
            self.cam_label.image = img
        
        self.after(10, self.update_camera)
    
    def capture_image(self):
        if self.current_frame is None:
            return
            
        h, w = self.current_frame.shape[:2]
        x1, y1 = (w - Config.ROI_SIZE) // 2, (h - Config.ROI_SIZE) // 2
        
        self.captured_img = self.current_frame[y1:y1+Config.ROI_SIZE, x1:x1+Config.ROI_SIZE].copy()
        
        self.analyze_btn.config(state=tk.NORMAL)
        self.classify_btn.config(state=tk.DISABLED)
        
        self.prediction_label.config(text="Kelas: -")
        self.confidence_label.config(text="Keyakinan: -")
        self.edge_info_label.config(text="Edge Density: -")
        
        self.analyze_image()
    
    def analyze_image(self):
        if self.captured_img is None:
            return
        
        Config.CANNY_THRESH1 = self.thresh1_slider.get()
        Config.CANNY_THRESH2 = self.thresh2_slider.get()
        
        gray = cv2.cvtColor(self.captured_img, cv2.COLOR_BGR2GRAY)
        
        edges = cv2.Canny(gray, Config.CANNY_THRESH1, Config.CANNY_THRESH2)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        images = [
            cv2.cvtColor(self.captured_img, cv2.COLOR_BGR2RGB),
            gray,
            edges,
            edges_rgb
        ]
        
        for i, img_data in enumerate(images):
            img = cv2.resize(img_data, (200, 200))
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            img_pil = Image.fromarray(img)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            
            self.preprocess_labels[i].configure(image=img_tk)
            self.preprocess_labels[i].image = img_tk
        
        info_text = f"""Hasil Preprocessing:
        - Canny Threshold: {Config.CANNY_THRESH1}-{Config.CANNY_THRESH2}
        - Edge Density: {edge_density:.2%}
        """
        self.preprocess_info.delete(1.0, tk.END)
        self.preprocess_info.insert(tk.END, info_text)
        
        self.classify_btn.config(state=tk.NORMAL)
        self.edge_info_label.config(text=f"Edge Density: {edge_density:.2%}")
    
    def classify_image(self):
        if self.captured_img is None or self.model is None:
            return
            
        processed_img, edges = preprocess_image(self.captured_img)
        input_img = np.expand_dims(processed_img, axis=0)
        
        predictions = self.model.predict(input_img)[0]
        class_idx = np.argmax(predictions)
        confidence = predictions[class_idx]
        class_name = Config.CLASS_DISPLAY[Config.CLASSES[class_idx]]
        
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        self.prediction_label.config(text=f"Kelas: {class_name}")
        self.confidence_label.config(text=f"Keyakinan: {confidence*100:.2f}%")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result_id = f"result_{len(self.results)+1}"
        
        result_data = {
            "id": result_id,
            "timestamp": timestamp,
            "class": class_name,
            "confidence": f"{confidence*100:.2f}%",
            "edge_density": f"{edge_density:.2%}",
            "image": self.captured_img.copy(),
            "edges": edges.copy()
        }
        self.results.append(result_data)
        
        self.result_tree.insert("", tk.END, iid=result_id, 
                              values=(result_data["timestamp"], result_data["class"], 
                                     result_data["confidence"], 
                                     result_data["edge_density"]))
    
    def update_threshold(self, event=None):
        if self.captured_img is not None:
            self.analyze_image()
    
    def reset_threshold(self):
        self.thresh1_slider.set(Config.CANNY_THRESH1)
        self.thresh2_slider.set(Config.CANNY_THRESH2)
        self.update_threshold()
    
    def save_results(self):
        """Simpan hasil klasifikasi ke file"""
        if not self.results:
            messagebox.showwarning("Peringatan", "Tidak ada hasil yang bisa disimpan")
            return
            
        save_dir = filedialog.askdirectory(title="Pilih Folder Penyimpanan")
        if not save_dir:
            return
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = os.path.join(save_dir, f"hasil_klasifikasi_{timestamp}")
            os.makedirs(session_dir, exist_ok=True)
            
            data_for_csv = []
            for result in self.results:
                img_path = os.path.join(session_dir, f"{result['id']}_original.jpg")
                cv2.imwrite(img_path, result["image"])
                
                edge_path = os.path.join(session_dir, f"{result['id']}_edge.jpg")
                cv2.imwrite(edge_path, result["edges"])
                
                data_for_csv.append({
                    "ID": result["id"],
                    "Waktu": result["timestamp"],
                    "Kelas": result["class"],
                    "Keyakinan": result["confidence"],
                    "Edge Density": result["edge_density"],
                    "Path Gambar": os.path.abspath(img_path),
                    "Path Edge": os.path.abspath(edge_path)
                })
            
            csv_path = os.path.join(session_dir, "hasil_klasifikasi.csv")
            df = pd.DataFrame(data_for_csv)
            df.to_csv(csv_path, index=False)
            
            messagebox.showinfo("Sukses", f"Hasil berhasil disimpan di:\n{session_dir}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menyimpan hasil: {str(e)}")
    
    def delete_selected(self):
        selected_items = self.result_tree.selection()
        if not selected_items:
            return
            
        if messagebox.askyesno("Konfirmasi", "Hapus hasil yang dipilih?"):
            for item_id in selected_items:
                self.result_tree.delete(item_id)
                self.results = [r for r in self.results if r["id"] != item_id]
    
    def on_closing(self):
        if messagebox.askokcancel("Keluar", "Apakah Anda yakin ingin keluar?"):
            if self.cap is not None:
                self.cap.release()
            self.destroy()

if __name__ == "__main__":
    app = SkinTextureApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()