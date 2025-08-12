import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

SOURCE_DIR = "E:/DATASET/sayang"  
OUTPUT_DIR = "E:/DATASET/hasil_analisis_50100"  

LOW_THRESHOLD_PERCENT = 50    
HIGH_THRESHOLD_PERCENT = 100   
TARGET_SIZE = (200, 200)      

def resize_to_target(img):
    return cv2.resize(img, TARGET_SIZE)

def apply_canny_edge_detection(gray_img):
    return cv2.Canny(gray_img, LOW_THRESHOLD_PERCENT, HIGH_THRESHOLD_PERCENT)

def calculate_edge_stats(edges):
    total_pixels = TARGET_SIZE[0] * TARGET_SIZE[1]

    sum_pixels = cv2.sumElems(edges)[0] if hasattr(cv2, 'sumElems') else cv2.sum(edges)[0]
    
    edge_count = cv2.countNonZero(edges)
    
    edge_density = (edge_count / total_pixels) * 100
    
    return sum_pixels, edge_count, edge_density

def process_single_image(img_path, output_dir, category):
    img = cv2.imread(img_path)
    if img is None:
        return None

    img = resize_to_target(img)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    edges = apply_canny_edge_detection(gray)
    
    sum_pixels, edge_count, edge_density = calculate_edge_stats(edges)
    
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_canny.jpg"), edges)
    
    with open(os.path.join(output_dir, f"{base_name}_results.txt"), 'w') as f:
        f.write(f"Kategori: {category}\n")
        f.write(f"Sum of Pixels: {sum_pixels:.2f}\n")
        f.write(f"Jumlah Pixel Tepi: {edge_count}\n")
        f.write(f"Density Tepi: {edge_density:.2f}%\n")
        f.write(f"Ambang Canny: {LOW_THRESHOLD_PERCENT}%-{HIGH_THRESHOLD_PERCENT}%\n")
        f.write(f"Ukuran Frame: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}\n")
    
    return {
        'category': category,
        'sum_pixels': sum_pixels,
        'edge_count': edge_count,
        'edge_density': edge_density
    }

def process_all_images():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    categories = ['lembut', 'normal', 'kasar']
    all_results = []
    
    for category in categories:
        input_dir = os.path.join(SOURCE_DIR, category)
        output_dir = os.path.join(OUTPUT_DIR, category)
        os.makedirs(output_dir, exist_ok=True)
        
        if not os.path.exists(input_dir):
            print(f"Folder {input_dir} tidak ditemukan, dilewati...")
            continue
            
        print(f"\nMemproses kategori: {category}")
        img_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for filename in tqdm(img_files, desc=category):
            img_path = os.path.join(input_dir, filename)
            result = process_single_image(img_path, output_dir, category)
            if result:
                all_results.append(result)
    
    create_summary_report(all_results)
    
    create_visualizations(all_results)

def create_summary_report(all_results):
    with open(os.path.join(OUTPUT_DIR, "summary_report.txt"), 'w') as f:
        f.write("===== LAPORAN HASIL ANALISIS UNTUK FRAME 200x200 =====\n\n")
        f.write(f"Parameter Canny: Low={LOW_THRESHOLD_PERCENT}%, High={HIGH_THRESHOLD_PERCENT}%\n\n")
        
        categories = ['lembut', 'normal', 'kasar']
        for category in categories:
            cat_results = [r for r in all_results if r['category'] == category]
            if not cat_results:
                continue
                
            avg_sum = np.mean([r['sum_pixels'] for r in cat_results])
            avg_edges = np.mean([r['edge_count'] for r in cat_results])
            avg_density = np.mean([r['edge_density'] for r in cat_results])
            
            f.write(f"Kategori: {category}\n")
            f.write(f"Rata-rata Sum of Pixels: {avg_sum:.2f}\n")
            f.write(f"Rata-rata Pixel Tepi: {avg_edges:.2f}\n")
            f.write(f"Rata-rata Density Tepi: {avg_density:.2f}%\n")
            f.write(f"Jumlah Gambar: {len(cat_results)}\n\n")

def create_visualizations(all_results):
    """Membuat visualisasi hasil analisis"""
    categories = ['lembut', 'normal', 'kasar']
    
    avg_edges = []
    avg_density = []
    
    for category in categories:
        cat_results = [r for r in all_results if r['category'] == category]
        if cat_results:
            avg_edges.append(np.mean([r['edge_count'] for r in cat_results]))
            avg_density.append(np.mean([r['edge_density'] for r in cat_results]))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.bar(categories, avg_edges, color=['skyblue', 'lightgreen', 'salmon'])
    plt.title('Rata-rata Jumlah Pixel Tepi\n(Ukuran Frame 200x200)')
    plt.ylabel('Jumlah Pixel')
   
    plt.subplot(1, 2, 2)
    plt.bar(categories, avg_density, color=['skyblue', 'lightgreen', 'salmon'])
    plt.title('Rata-rata Density Tepi\n(Ukuran Frame 200x200)')
    plt.ylabel('Persentase Area (%)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'hasil_analisis.png'))
    plt.close()

if __name__ == "__main__":
    print("===== PROGRAM ANALISIS TEKSTUR UNTUK FRAME 200x200 =====")
    print(f"Folder Sumber: {SOURCE_DIR}")
    print(f"Folder Tujuan: {OUTPUT_DIR}")
    print(f"Parameter Canny: Low={LOW_THRESHOLD_PERCENT}%, High={HIGH_THRESHOLD_PERCENT}%")
    print(f"Ukuran Target Frame: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}\n")
    
    process_all_images()
    
    print("\n===== PROSES SELESAI =====")
    print("Hasil disimpan dalam struktur:")
    print(f"  - {OUTPUT_DIR}/[kategori]/[nama_file]_canny.jpg")
    print(f"  - {OUTPUT_DIR}/[kategori]/[nama_file]_results.txt")
    print(f"  - {OUTPUT_DIR}/summary_report.txt")
    print(f"  - {OUTPUT_DIR}/hasil_analisis.png (visualisasi)")