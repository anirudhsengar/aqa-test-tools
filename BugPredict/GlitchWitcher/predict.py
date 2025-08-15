import pandas as pd
from REPD_Impl import REPD
from autoencoder import AutoEncoder
import warnings
import tensorflow.compat.v1 as tf
import numpy as np
import json
import sys
import os
import argparse
import scipy.stats as st

# Suppress warnings
tf.disable_v2_behavior()
warnings.simplefilter("ignore")

def format_predictions(predictions):
    """Format PDF predictions for display"""
    results = []
    for i in range(predictions.shape[0]):
        pred = predictions[i]
        if isinstance(pred, np.ndarray) and len(pred) >= 2:
            p_defective = float(pred[0])
            p_non_defective = float(pred[1])
        else:
            p_defective = 0.0
            p_non_defective = 0.0
        results.append({
            'p_defective': p_defective,
            'p_non_defective': p_non_defective
        })
    return results

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
    except Exception:
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

    if classifier.dnd is None:
        raise ValueError("Failed to get non-defective distribution class")
    if classifier.dd is None:
        raise ValueError("Failed to get defective distribution class")

    return classifier

def predict(features_file, model_dir="trained_model"):
    """Make predictions using pre-trained model"""
    classifier = load_trained_model(model_dir)
    # Load test data
    df_test = pd.read_csv(features_file)
    # Check if CSV has data rows (more than just header)
    if len(df_test) == 0:
        return []

    file_names = df_test["File"].values if "File" in df_test.columns else np.arange(len(df_test))
    X_test = df_test.drop(columns=["File"]).values if "File" in df_test.columns else df_test.values

    # Make predictions (PDF values)
    pdf_predictions = classifier.predict(X_test)
    prediction_data = format_predictions(pdf_predictions)

    # Close the session
    classifier.dim_reduction_model.close()
    return format_results(file_names, prediction_data)

def write_submission(results, output_path):
    """Write predictions to CSV"""
    if not results:
        # create an empty but valid CSV with header
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("File,PDF_Defective,PDF_NonDefective\n")
        return
    df = pd.DataFrame(results)
    # Normalize column names
    if {'file', 'p_defective', 'p_non_defective'}.issubset(df.columns):
        df = df.rename(columns={'file': 'File', 'p_defective': 'PDF_Defective', 'p_non_defective': 'PDF_NonDefective'})
    df.to_csv(output_path, index=False)

def parse_args(argv):
    """Accept both flag and positional styles for backward compatibility"""
    parser = argparse.ArgumentParser(description="Run REPD predictions.")
    # Flag style
    parser.add_argument("--input", dest="input_opt", help="Path to features CSV")
    parser.add_argument("--model-dir", dest="model_dir_opt", help="Path to trained_model directory")
    parser.add_argument("--model_dir", dest="model_dir_opt2", help="Path to trained_model directory (alt spelling)")
    parser.add_argument("--output", dest="output_opt", help="Output CSV path (default: submission.csv)")
    # Positional style: [input] [model_dir] [output]
    parser.add_argument("input_pos", nargs="?", help="Path to features CSV (positional)")
    parser.add_argument("model_dir_pos", nargs="?", help="Model directory (positional)")
    parser.add_argument("output_pos", nargs="?", help="Output CSV path (positional)")

    args = parser.parse_args(argv)

    features = args.input_opt or args.input_pos
    model_dir = (args.model_dir_opt or args.model_dir_opt2 or
                 args.model_dir_pos or "trained_model")
    output = args.output_opt or args.output_pos or "submission.csv"

    if not features:
        parser.print_help()
        sys.exit(2)

    return features, model_dir, output

if __name__ == "__main__":
    features_csv_path, model_dir, output_csv = parse_args(sys.argv[1:])
    results = predict(features_csv_path, model_dir=model_dir)
    write_submission(results, output_csv)

    # Also print a brief human-readable summary to stdout
    print(f"Predictions written to {output_csv}. Files: {len(results)}")