import pandas as pd
from REPD_Impl import REPD
from autoencoder import AutoEncoder
import warnings
import tensorflow.compat.v1 as tf
import numpy as np
import json
import sys
import os
import scipy.stats as st

# Suppress warnings
tf.disable_v2_behavior()
warnings.simplefilter("ignore")

def format_predictions(predictions):
    """Format PDF predictions for display"""
    results = []
    
    print(f"Debug: Predictions shape: {predictions.shape}", file=sys.stderr)
    print(f"Debug: Predictions content: {predictions}", file=sys.stderr)
    
    # Now predictions should be (n_samples, 2) - no more 3D arrays!
    for i in range(predictions.shape[0]):
        pred = predictions[i]
        print(f"Debug: Processing prediction {i}: {pred}", file=sys.stderr)
        
        # Extract probabilities
        if isinstance(pred, np.ndarray) and len(pred) >= 2:
            p_defective = float(pred[0])
            p_non_defective = float(pred[1])
        else:
            print(f"Warning: Unexpected prediction format for {i}: {pred}", file=sys.stderr)
            p_defective = 0.0
            p_non_defective = 0.0
        
        results.append({
            'p_defective': p_defective,
            'p_non_defective': p_non_defective
        })
        
        print(f"Debug: File {i} - P(Defective): {p_defective}, P(Non-Defective): {p_non_defective}", file=sys.stderr)
    
    return results

def format_results_for_comparison(file_names, base_data, head_data):
    """Format results as tables comparing BEFORE and AFTER for each file"""
    
    # First, calculate percentage changes for all files to determine sorting order
    file_changes = []
    
    for i, file_name in enumerate(file_names):
        if i < len(base_data) and i < len(head_data):
            base_defective = base_data[i]['p_defective']
            base_non_defective = base_data[i]['p_non_defective']
            head_defective = head_data[i]['p_defective']
            head_non_defective = head_data[i]['p_non_defective']
            
            # Calculate percentage changes
            if base_defective != 0:
                defective_change = ((head_defective - base_defective) / abs(base_defective)) * 100
            else:
                defective_change = 0 if head_defective == 0 else float('inf')
            
            if base_non_defective != 0:
                non_defective_change = ((head_non_defective - base_non_defective) / abs(base_non_defective)) * 100
            else:
                non_defective_change = 0 if head_non_defective == 0 else float('inf')
            
            # Use the maximum absolute change for sorting
            max_change = max(abs(defective_change), abs(non_defective_change))
            
            file_changes.append({
                'index': i,
                'file_name': file_name,
                'max_change': max_change,
                'defective_change': defective_change,
                'non_defective_change': non_defective_change,
                'base_defective': base_defective,
                'base_non_defective': base_non_defective,
                'head_defective': head_defective,
                'head_non_defective': head_non_defective
            })
        else:
            file_changes.append({
                'index': i,
                'file_name': file_name,
                'max_change': 0,
                'error': True
            })
    
    # Sort files by maximum percentage change in descending order
    file_changes.sort(key=lambda x: x['max_change'], reverse=True)
    
    # Generate output with sorted files
    output = ["## 📊 Bug Prediction Analysis\n"]
    
    for file_data in file_changes:
        file_name = file_data['file_name']
        output.append(f"#### File: `{file_name}`\n")
        
        if 'error' in file_data:
            output.append("| Status |")
            output.append("|--------|")
            output.append("| Error: Prediction data not available |")
            output.append("")
        else:
            base_defective = file_data['base_defective']
            base_non_defective = file_data['base_non_defective']
            head_defective = file_data['head_defective']
            head_non_defective = file_data['head_non_defective']
            defective_change = file_data['defective_change']
            non_defective_change = file_data['non_defective_change']
            
            # Format percentage change values
            def format_change(change_val):
                if change_val == float('inf'):
                    return "∞%"
                elif change_val == float('-inf'):
                    return "-∞%"
                else:
                    return f"{change_val:+.2f}%"

            before = "Defective" if base_defective > base_non_defective else "Non-Defective"
            after = "Defective" if head_defective > head_non_defective else "Non-Defective"

            output.append("Outcome: " + before + " -> " + after)
            
            # Create table with 4 columns
            output.append("| Metric | BEFORE PR | AFTER PR | % Change |")
            output.append("|--------|-----------|----------|----------|")
            output.append(f"| PDF(Defective \\| Reconstruction Error) | {base_defective} | {head_defective} | {format_change(defective_change)} |")
            output.append(f"| PDF(Non-Defective \\| Reconstruction Error) | {base_non_defective} | {head_non_defective} | {format_change(non_defective_change)} |")
            output.append("")
    
    return "\n".join(output)

def format_results(file_names, prediction_data):
    """Format results with probability values (for individual calls)"""
    results = []
    
    for i, file_name in enumerate(file_names):
        if i < len(prediction_data):
            p_defective = prediction_data[i]['p_defective']
            p_non_defective = prediction_data[i]['p_non_defective']
            
            results.append({
                'file': file_name,
                'p_defective': p_defective,
                'p_non_defective': p_non_defective
            })
        else:
            results.append({
                'file': file_name,
                'error': 'No prediction available'
            })
    
    return results

def get_distribution_class(dist_name):
    """Get the distribution class (not frozen) from scipy.stats"""
    if dist_name is None:
        return None
    
    try:
        dist_class = getattr(st, dist_name)
        return dist_class
    except Exception as e:
        return None

def load_trained_model(model_dir="trained_model"):
    """Load the pre-trained model"""
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Trained model not found at {model_dir}. Please ensure the model is trained and saved.")
    
    # Load metadata from JSON
    metadata_path = os.path.join(model_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Model metadata not found at {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load REPD classifier parameters from JSON
    classifier_params_path = os.path.join(model_dir, "classifier_params.json")
    if not os.path.exists(classifier_params_path):
        raise FileNotFoundError(f"Classifier parameters not found at {classifier_params_path}")
    
    with open(classifier_params_path, 'r') as f:
        classifier_params = json.load(f)
        
    # Recreate the autoencoder with saved architecture
    autoencoder = AutoEncoder(
        metadata['architecture'], 
        metadata['learning_rate'], 
        metadata['epochs'], 
        metadata['batch_size']
    )
    
    # Load the saved autoencoder weights
    autoencoder_path = os.path.join(model_dir, "autoencoder")
    autoencoder.load(autoencoder_path)
    
    # Recreate REPD classifier
    classifier = REPD(autoencoder)
        
    # Non-defective distribution
    classifier.dnd = get_distribution_class(classifier_params.get('dnd_name'))
    classifier.dnd_pa = tuple(classifier_params.get('dnd_params', []))
    
    # Defective distribution  
    classifier.dd = get_distribution_class(classifier_params.get('dd_name'))
    classifier.dd_pa = tuple(classifier_params.get('dd_params', []))
    
    # Check if distributions were created successfully
    if classifier.dnd is None:
        raise ValueError("Failed to get non-defective distribution class")
    if classifier.dd is None:
        raise ValueError("Failed to get defective distribution class")
    
    return classifier

def predict(features_file, model_dir="trained_model", output_file=None):
    """Make predictions using pre-trained model"""
    
    classifier = load_trained_model(model_dir)
    
    # Load test data
    df_test = pd.read_csv(features_file)
    
    # Check if CSV has data rows (more than just header)
    if len(df_test) == 0:
        print("No files to analyze.")
        return []
    
    file_names = df_test["File"].values
    X_test = df_test.drop(columns=["File"]).values
    
    print(f"Debug: Processing {len(file_names)} files", file=sys.stderr)
    print(f"Debug: File names: {file_names}", file=sys.stderr)
    print(f"Debug: X_test shape: {X_test.shape}", file=sys.stderr)
                
    # Make predictions (PDF values)
    pdf_predictions = classifier.predict(X_test)
    print(f"Debug: Predictions shape: {pdf_predictions.shape}", file=sys.stderr)
    print(f"Debug: Predictions type: {type(pdf_predictions)}", file=sys.stderr)
    
    # Format predictions for display
    prediction_data = format_predictions(pdf_predictions)

    # Format and return results
    results = format_results(file_names, prediction_data)
    
    # Save to output file if specified
    if output_file:
        # Create submission CSV format
        submission_data = []
        for i, file_name in enumerate(file_names):
            if i < len(prediction_data):
                # Use p_defective as the predicted v(g) value
                predicted_vg = prediction_data[i]['p_defective']
                submission_data.append({
                    'File': file_name,
                    'Predicted_v(g)': predicted_vg
                })
            else:
                submission_data.append({
                    'File': file_name,
                    'Predicted_v(g)': 0.0
                })
        
        submission_df = pd.DataFrame(submission_data)
        submission_df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
    
    # Close the session
    classifier.dim_reduction_model.close()
    
    return results

def print_usage():
    """Print usage information"""
    print("Usage:")
    print("  python predict.py <features_csv> [model_dir] [output_csv]")
    print("  python predict.py --input <features_csv> --model-dir <model_dir> --output <output_csv>")
    print("")
    print("Arguments:")
    print("  features_csv    Path to the CSV file containing extracted features")
    print("  model_dir       Path to the directory containing trained model files (default: 'trained_model')")
    print("  output_csv      Path to save predictions as CSV (optional)")
    print("")
    print("Examples:")
    print("  python predict.py features.csv")
    print("  python predict.py features.csv ./my_model")
    print("  python predict.py features.csv ./my_model predictions.csv")
    print("  python predict.py --input features.csv --model-dir ./my_model --output predictions.csv")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    # Parse command line arguments
    features_csv_path = None
    model_dir = "trained_model"
    output_file = None
    
    # Support both positional and named arguments
    if sys.argv[1] == "--input":
        # Named argument format
        i = 1
        while i < len(sys.argv):
            if sys.argv[i] == "--input" and i + 1 < len(sys.argv):
                features_csv_path = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--model-dir" and i + 1 < len(sys.argv):
                model_dir = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--output" and i + 1 < len(sys.argv):
                output_file = sys.argv[i + 1]
                i += 2
            else:
                i += 1
    else:
        # Positional argument format
        features_csv_path = sys.argv[1]
        if len(sys.argv) > 2:
            model_dir = sys.argv[2]
        if len(sys.argv) > 3:
            output_file = sys.argv[3]
    
    if not features_csv_path:
        print("Error: Features CSV file path is required")
        print_usage()
        sys.exit(1)
    
    if not os.path.exists(features_csv_path):
        print(f"Error: Features CSV file not found: {features_csv_path}")
        sys.exit(1)
    
    if not os.path.exists(model_dir):
        print(f"Error: Model directory not found: {model_dir}")
        sys.exit(1)
    
    try:
        results = predict(features_csv_path, model_dir, output_file)
        
        # For command line usage, print individual results
        for result in results:
            if 'error' in result:
                print(f"File: {result['file']}")
                print(f"Error: {result['error']}")
            else:
                print(f"File: {result['file']}")
                print(f"PDF(Defective | Reconstruction Error): {result['p_defective']}")
                print(f"PDF(Non-Defective | Reconstruction Error): {result['p_non_defective']}")
            print()
            
    except Exception as e:
        print(f"Error during prediction: {str(e)}", file=sys.stderr)
        sys.exit(1)