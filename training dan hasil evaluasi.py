import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FOLDER_CONFIG = {
    'lembut': 'lembut',
    'normal': 'normal',
    'kasar': 'kasar'
}
CLASSES = ['lembut', 'normal', 'kasar']
IMG_SIZE = (200, 200)
MODEL_CHECKPOINT_PATH = "best_model.h5"

CANNY_THRESH_1 = 50
CANNY_THRESH_2 = 100
BATCH_SIZE = 16
EPOCHS = 30 

EVALUATION_THRESHOLD_PAIRS = [
    (50, 100),
    (80, 160),
    (30, 90),
    (20, 40),
    (15, 45)
]


def load_and_process_data_for_training():
    images = []
    labels = []
    
    label_map = {label: i for i, label in enumerate(CLASSES)}

    print("Loading and processing images for training...")
    for label, path in FOLDER_CONFIG.items():
        if not os.path.isdir(path):
            print(f"Warning: Directory not found for class '{label}': {path}")
            continue
            
        class_num = label_map[label]
        file_list = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for image_name in tqdm(file_list, desc=f"Processing '{label}' images"):
            try:
                img_path = os.path.join(path, image_name)
                gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if gray_img is None:
                    continue
                
                edges = cv2.Canny(gray_img, CANNY_THRESH_1, CANNY_THRESH_2)
                edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                resized_img = cv2.resize(edges_rgb, IMG_SIZE)
                
                images.append(resized_img)
                labels.append(class_num)

            except Exception as e:
                print(f"Error processing image {image_name}: {e}")

    images = np.array(images, dtype="float32") / 255.0
    labels = np.array(labels)
    labels = to_categorical(labels, num_classes=len(CLASSES))
    
    return images, labels


def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Flatten(), 
        Dense(256, activation='relu'),
        Dropout(0.8 ),
        Dense(num_classes, activation='softmax')
    ])
    return model


def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_history.png")
    print("\nTraining history graph saved as training_history.png")

def visualize_confusion_matrix(y_true, y_pred, thresholds_str, accuracy):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES, annot_kws={"size": 16})
    title = f"Confusion Matrix (Akurasi: {accuracy:.2%})\nAmbang Batas: {thresholds_str}"
    plt.title(title, fontsize=14)
    plt.ylabel('Aktual', fontsize=12)
    plt.xlabel('Prediksi', fontsize=12)
    plt.tight_layout()
    filename = f"confusion_matrix_{thresholds_str.strip('()').replace(', ', '_')}.png"
    plt.savefig(filename)
    print(f"\nConfusion matrix saved to {filename}")

def plot_density_chart(df):
    print("\nGenerating average edge density chart...")
    
    df_plot = df.set_index('Thresholds')
    density_cols = ['Soft Density', 'Normal Density', 'Rough Density']
    
    df_plot.rename(columns={
        'Soft Density': 'Soft Density',
        'Normal Density': 'Normal Density',
        'Rough Density': 'Rough Density'
    }, inplace=True)
    
    ax = df_plot[density_cols].plot(kind='bar', figsize=(14, 8), width=0.8, colormap='viridis')
    
    ax.set_title('Rata-rata Kerapatan Tepi Berdasarkan Ambang Batas Canny dan Kelembutan Kulit', fontsize=16)
    ax.set_ylabel('Rata-rata Kerapatan Tepi', fontsize=12)
    ax.set_xlabel('Ambang Batas Canny (t1, t2)', fontsize=12)
    
    ax.tick_params(axis='x', rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(title='Skin Category')
    
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1%}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points')

    plt.tight_layout()
    plt.savefig("average_edge_density.png")
    print("Average edge density chart saved to average_edge_density.png")

def plot_accuracy_chart(df):
    print("\nGenerating per-class accuracy chart...")

    df_plot = df.set_index('Thresholds')
    accuracy_cols = ['Soft Acc.', 'Normal Acc.', 'Rough Acc.']
    
    df_plot.rename(columns={
        'Soft Acc.': 'Soft Acc.',
        'Normal Acc.': 'Normal Acc.',
        'Rough Acc.': 'Rough Acc.'
    }, inplace=True)

    ax = df_plot[accuracy_cols].plot(kind='bar', figsize=(14, 8), width=0.8)

    ax.set_title('Per-Class Model Accuracy by Canny Thresholds', fontsize=16)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel('Canny Thresholds (Thresh1, Thresh2)', fontsize=12)
    
    ax.tick_params(axis='x', rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(title='Skin Category')
    ax.set_ylim(0, 1.05)

    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1%}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    color='white',
                    xytext=(0, -12),
                    textcoords='offset points')

    plt.tight_layout()
    plt.savefig("per_class_accuracy.png")
    print("Per-class accuracy chart saved to per_class_accuracy.png")


def evaluate_model_performance():
    print("\n\n" + "="*80)
    print("STARTING EVALUATION PHASE")
    print("="*80)
    
    if not os.path.exists(MODEL_CHECKPOINT_PATH):
        print(f"Error: Model file not found at '{MODEL_CHECKPOINT_PATH}'.")
        return

    print(f"Loading best model from: {MODEL_CHECKPOINT_PATH}")
    model = load_model(MODEL_CHECKPOINT_PATH)
    
    image_paths = {label: [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                   for label, path in FOLDER_CONFIG.items()}
    
    labels_map = {label: i for i, label in enumerate(CLASSES)}
    results_data = []

    print("\nEvaluating model accuracy and edge density for different Canny threshold pairs...")
    for t1, t2 in tqdm(EVALUATION_THRESHOLD_PAIRS, desc="Testing Pairs"):
        all_true_labels = []
        all_pred_labels = []
        class_densities = {label: [] for label in CLASSES}

        for label, paths in image_paths.items():
            true_label_index = labels_map[label]
            for img_path in paths:
                gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if gray_img is None: continue

                edges = cv2.Canny(gray_img, t1, t2)
                
                density = np.sum(edges > 0) / edges.size
                class_densities[label].append(density)
                
                edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                processed = cv2.resize(edges_rgb, IMG_SIZE)
                processed = processed.astype('float32') / 255.0
                processed_batch = np.expand_dims(processed, axis=0)

                prediction = model.predict(processed_batch, verbose=0)[0]
                predicted_label_index = np.argmax(prediction)

                all_true_labels.append(true_label_index)
                all_pred_labels.append(predicted_label_index)
        
        report = classification_report(all_true_labels, all_pred_labels, target_names=CLASSES, output_dict=True, zero_division=0)
        
        avg_densities = {label: np.mean(densities) if densities else 0 for label, densities in class_densities.items()}
        
        results_data.append({
            'Thresholds': f"({t1}, {t2})",
            'Overall Acc.': report['accuracy'],
            'Soft Acc.': report['lembut']['precision'],
            'Normal Acc.': report['normal']['precision'],
            'Rough Acc.': report['kasar']['precision'],
            'Soft Density': avg_densities.get('lembut', 0),
            'Normal Density': avg_densities.get('normal', 0),
            'Rough Density': avg_densities.get('kasar', 0)
        })

    df = pd.DataFrame(results_data)
    df['Thresholds'] = pd.Categorical(df['Thresholds'], categories=[f"({t1}, {t2})" for t1, t2 in EVALUATION_THRESHOLD_PAIRS], ordered=True)
    df = df.sort_values('Thresholds').reset_index(drop=True)
    
    print("\n" + "="*80)
    print("Evaluation Results: Model Accuracy and Average Edge Density")
    print("="*80)
    df_display = df.copy()
    for col in df_display.columns:
        if 'Acc.' in col or 'Density' in col:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.2%}")
    print(df_display.to_string(index=False))
    print("="*80)

    plot_density_chart(df)
    plot_accuracy_chart(df)
    
    df_sorted_by_acc = df.sort_values(by='Overall Acc.', ascending=False).reset_index(drop=True)
    if not df_sorted_by_acc.empty:
        best_row = df_sorted_by_acc.iloc[0]
        best_pair_str = best_row['Thresholds']
        best_overall_acc = best_row['Overall Acc.']
        t1_best, t2_best = [int(x) for x in best_pair_str.strip('()').split(',')]

        print(f"\nGenerating final report for the best parameter set: {best_pair_str}...")
        final_y_true, final_y_pred = [], []
        for label, paths in image_paths.items():
            for img_path in tqdm(paths, desc=f"Final Prediction - {label}"):
                gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if gray_img is None: continue
                
                edges = cv2.Canny(gray_img, t1_best, t2_best)
                edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                processed = cv2.resize(edges_rgb, IMG_SIZE)
                processed = processed.astype('float32') / 255.0
                processed_batch = np.expand_dims(processed, axis=0)
                
                prediction = model.predict(processed_batch, verbose=0)[0]
                final_y_true.append(labels_map[label])
                final_y_pred.append(np.argmax(prediction))

        visualize_confusion_matrix(final_y_true, final_y_pred, best_pair_str, best_overall_acc)

        print("\n" + "-"*60)
        print(f"Detailed Classification Report for Best Thresholds {best_pair_str}")
        print("-"*60)
        print(classification_report(final_y_true, final_y_pred, target_names=CLASSES))
        print("-"*60)


if __name__ == "__main__":
    print("="*80)
    print("STARTING MODEL TRAINING PHASE")
    print("="*80)

    images, labels = load_and_process_data_for_training()
    
    if len(images) == 0:
        print("\nNo images were loaded. Please check FOLDER_CONFIG paths. Exiting.")
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            images, labels, test_size=0.2, random_state=42, stratify=labels
        )
        print(f"\nData loaded: {len(X_train)} training images, {len(X_val)} validation images.")
        
        input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)
        num_classes = len(CLASSES)
        model = build_model(input_shape, num_classes)
        
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        model.summary()
        
        checkpoint = ModelCheckpoint(
            MODEL_CHECKPOINT_PATH, 
            monitor='val_accuracy', 
            verbose=1, 
            save_best_only=True, 
            mode='max'
        )
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=2,
            verbose=1, 
            mode='min'
        )
        
        print("\nStarting model training...")
        history = model.fit(
            X_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=[checkpoint, early_stopping]
        )
        
        plot_training_history(history)
        print("\nTraining complete! The best model was saved as '{}'.".format(MODEL_CHECKPOINT_PATH))

        evaluate_model_performance()