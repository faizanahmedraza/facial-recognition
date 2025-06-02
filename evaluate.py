import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import seaborn as sns
from main import EliteFaceRecognition

def validate_image_path(image_path):
    """Validate that an image path exists and is readable"""
    if not isinstance(image_path, str) or not os.path.exists(image_path):
        return False
    try:
        img = cv2.imread(image_path)
        return img is not None
    except:
        return False

def prepare_test_data(test_folder="dataset/unknown_faces"):
    """
    Prepare test data with known labels for evaluation
    Returns DataFrame with valid image paths only
    """
    test_data = []
    
    if not os.path.exists(test_folder):
        print(f"⚠️ Test folder not found: {test_folder}")
        return pd.DataFrame()
    
    for person_name in os.listdir(test_folder):
        person_folder = os.path.join(test_folder, person_name)
        
        if not os.path.isdir(person_folder):
            continue
            
        valid_images = []
        for image_file in os.listdir(person_folder):
            if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(person_folder, image_file)
                if validate_image_path(image_path):
                    valid_images.append(image_path)
                else:
                    print(f"⚠️ Invalid image skipped: {image_path}")
        
        if valid_images:
            test_data.extend([{
                'true_label': person_name if person_name else "Unknown",
                'image_path': img_path
            } for img_path in valid_images])
    
    if not test_data:
        print("⚠️ No valid test images found!")
        return pd.DataFrame()
    
    return pd.DataFrame(test_data)

def evaluate_system(face_system, test_df, results_dir="results"):
    """
    Robust evaluation with better error handling
    """
    os.makedirs(results_dir, exist_ok=True)
    detections_dir = os.path.join(results_dir, "detections")
    os.makedirs(detections_dir, exist_ok=True)
    
    results = []
    processed_count = 0
    
    for idx, row in test_df.iterrows():
        if not validate_image_path(row['image_path']):
            print(f"⚠️ Skipping invalid image: {row['image_path']}")
            continue
            
        try:
            image = cv2.imread(row['image_path'])
            if image is None:
                print(f"⚠️ Could not read image (None returned): {row['image_path']}")
                continue
                
            result = {
                'true_label': row['true_label'],
                'image_path': row['image_path'],
                'predicted_label': "Unknown",
                'confidence': 0,
                'detection_image': None
            }
            
            try:
                face_locations, face_names, face_confidences = face_system.process_frame_optimized(image)
                
                if face_names:
                    result.update({
                        'predicted_label': face_names[0],
                        'confidence': face_confidences[0] if face_confidences else 0,
                    })
                    
                    try:
                        result_image = face_system.draw_results(
                            image.copy(), 
                            face_locations, 
                            face_names, 
                            face_confidences
                        )
                        filename = f"det_{processed_count}_{row['true_label']}_{result['predicted_label']}.jpg"
                        save_path = os.path.join(detections_dir, filename)
                        if cv2.imwrite(save_path, result_image):
                            result['detection_image'] = save_path
                        else:
                            print(f"⚠️ Failed to save detection image: {save_path}")
                    except Exception as e:
                        print(f"⚠️ Error drawing results for {row['image_path']}: {str(e)}")
                
                results.append(result)
                processed_count += 1
                
            except Exception as e:
                print(f"⚠️ Processing failed for {row['image_path']}: {str(e)}")
                results.append(result)
                
        except Exception as e:
            print(f"⚠️ Unexpected error with {row['image_path']}: {str(e)}")
            continue
    
    if not results:
        print("⚠️ No images processed successfully!")
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    results_df['correct'] = results_df['true_label'] == results_df['predicted_label']
    
    print(f"\n✅ Processed {processed_count}/{len(test_df)} images successfully")
    return results_df

def generate_metrics(results_df, results_dir="results"):
    """Generate metrics with better handling of edge cases"""
    if results_df.empty:
        print("⚠️ No results to generate metrics from!")
        return None
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Get all unique labels (including Unknown)
    all_labels = sorted(set(results_df['true_label'].unique()) | 
                  set(results_df['predicted_label'].unique()))
    
    # Classification Report
    try:
        report = classification_report(
            results_df['true_label'],
            results_df['predicted_label'],
            labels=all_labels,
            output_dict=True,
            zero_division=0
        )
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(os.path.join(results_dir, "classification_report.csv"), float_format="%.4f")
    except Exception as e:
        print(f"⚠️ Error generating classification report: {str(e)}")
        report_df = None
    
    # Confusion Matrix
    try:
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(
            results_df['true_label'],
            results_df['predicted_label'],
            labels=all_labels
        )
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=all_labels, yticklabels=all_labels)
        plt.title('Confusion Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"⚠️ Error generating confusion matrix: {str(e)}")
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(results_df['true_label'], results_df['predicted_label']),
        'precision_macro': precision_score(results_df['true_label'], results_df['predicted_label'], 
                              average='macro', zero_division=0),
        'recall_macro': recall_score(results_df['true_label'], results_df['predicted_label'], 
                            average='macro', zero_division=0),
        'f1_macro': f1_score(results_df['true_label'], results_df['predicted_label'], 
                      average='macro', zero_division=0),
        'total_samples': len(results_df),
        'known_predictions': (results_df['predicted_label'] != 'Unknown').sum(),
        'unknown_predictions': (results_df['predicted_label'] == 'Unknown').sum(),
        'average_confidence': results_df.loc[results_df['predicted_label'] != 'Unknown', 'confidence'].mean(),
        'correct_known': results_df[results_df['predicted_label'] != 'Unknown']['correct'].mean(),
        'correct_unknown': results_df[results_df['predicted_label'] == 'Unknown']['correct'].mean()
    }
    
    # Save metrics
    try:
        with open(os.path.join(results_dir, "metrics.txt"), 'w') as f:
            for k, v in metrics.items():
                if isinstance(v, float):
                    f.write(f"{k}: {v:.4f}\n")
                else:
                    f.write(f"{k}: {v}\n")
    except Exception as e:
        print(f"⚠️ Error saving metrics: {str(e)}")
    
    print("\n📊 Evaluation Metrics:")
    print(f"- Accuracy: {metrics['accuracy']:.4f}")
    print(f"- Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"- Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"- F1 Score (macro): {metrics['f1_macro']:.4f}")
    print(f"- Known predictions: {metrics['known_predictions']} (avg conf: {metrics['average_confidence']:.1f}%)")
    print(f"- Unknown predictions: {metrics['unknown_predictions']}")
    print(f"- Correct known: {metrics['correct_known']:.1%}")
    print(f"- Correct unknown: {metrics['correct_unknown']:.1%}")
    
    return metrics

def display_sample_results(results_df, n=9, cols=3):
    """Display sample results in a grid layout with better error handling"""
    if results_df.empty:
        print("⚠️ No results to display")
        return
    
    # Filter out samples without valid detection images
    # Properly formatted filtering code
    valid_results = results_df[
        results_df['detection_image'].notna() & 
        results_df['detection_image'].apply(lambda x: isinstance(x, str)) & 
        results_df['detection_image'].apply(lambda x: os.path.exists(x) if isinstance(x, str) else False)
    ]
    
    if valid_results.empty:
        print("⚠️ No valid detection images to display")
        return
    
    # Split into correct and incorrect predictions
    correct_samples = valid_results[valid_results['correct']]
    incorrect_samples = valid_results[~valid_results['correct']]
    
    # Determine how many samples to show from each category
    n_correct = min(n // 2, len(correct_samples))
    n_incorrect = min(n - n_correct, len(incorrect_samples))
    
    # Combine the samples to display
    display_samples = pd.concat([
        correct_samples.sample(n_correct) if n_correct > 0 else pd.DataFrame(),
        incorrect_samples.sample(n_incorrect) if n_incorrect > 0 else pd.DataFrame()
    ])
    
    if display_samples.empty:
        print("⚠️ No valid samples to display after filtering")
        return
    
    # Calculate grid dimensions
    rows = int(np.ceil(len(display_samples) / cols))
    figsize = (5 * cols, 4 * rows)
    
    # Create the figure
    plt.figure(figsize=figsize)
    
    for i, (_, row) in enumerate(display_samples.iterrows(), 1):
        try:
            img = cv2.imread(row['detection_image'])
            if img is None:
                print(f"⚠️ Could not load image: {row['detection_image']}")
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(rows, cols, i)
            plt.imshow(img)
            
            # Color code title based on correctness
            if row['correct']:
                title_color = 'green'
                status = "✓"
            else:
                title_color = 'red'
                status = "✗"
            
            plt.title(
                f"{status} True: {row['true_label']}\nPred: {row['predicted_label']}\nConf: {row['confidence']:.1f}%",
                color=title_color
            )
            plt.axis('off')
            
        except Exception as e:
            print(f"⚠️ Error displaying {row['detection_image']}: {str(e)}")
    
    plt.tight_layout()
    save_path = os.path.join("results", "sample_results.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved sample results to {save_path}")
    plt.show()

def main():
    print("\n🔍 Starting Face Recognition Evaluation")
    
    # Initialize system
    face_system = EliteFaceRecognition(tolerance=0.5, model='hog')
    if not face_system.load_encodings('face_encodings.pkl'):
        print("⚠️ Failed to load face encodings!")
        return
    
    # Prepare test data
    print("\n📂 Preparing test data...")
    test_df = prepare_test_data()
    if test_df.empty:
        print("⚠️ No valid test data found!")
        return
    
    # Run evaluation
    print("\n🔎 Running evaluation...")
    results_df = evaluate_system(face_system, test_df)
    if results_df.empty:
        print("⚠️ No results generated!")
        return
    
    # Generate metrics
    print("\n📊 Calculating metrics...")
    metrics = generate_metrics(results_df)
    
    # Save results
    results_df.to_csv(os.path.join("results", "detailed_results.csv"), index=False)
    
    # Display samples
    print("\n🖼️ Displaying sample results...")
    display_sample_results(results_df)
    
    print("\n✅ Evaluation complete!")

if __name__ == "__main__":
    main()