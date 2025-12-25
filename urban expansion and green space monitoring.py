#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install rasterio')


# In[2]:


# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')


# In[3]:


import os
base_dir = '/content/drive/My Drive/prj3'
os.chdir(base_dir)


# In[4]:


# Required imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from rasterio.plot import show
from matplotlib.colors import ListedColormap

# Set working directory - update this to your project folder path
# base_dir = r"C:\Users\divya\Desktop\urbanization in gurgaon"
# os.chdir(base_dir)

# LULC class definitions
class_names = {
    0: 'Urban',
    1: 'Vegetation',
    2: 'Water',
    3: 'Barren'
}

# Define color map for visualization
colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00']  # Red, Green, Blue, Yellow
cmap = ListedColormap(colors)

def get_lat_lon_extent(raster_path):
    """
    Get the latitude and longitude extent from a raster file

    Parameters:
    raster_path (str): Path to the raster file

    Returns:
    tuple: (min_lon, max_lon, min_lat, max_lat)
    """
    with rasterio.open(raster_path) as src:
        # Get the bounds in the CRS of the raster
        bounds = src.bounds

        # Extract coordinates
        min_lon, min_lat, max_lon, max_lat = bounds.left, bounds.bottom, bounds.right, bounds.top

        return min_lon, max_lon, min_lat, max_lat

def calculate_aspect_ratio(min_lon, max_lon, min_lat, max_lat):
    """
    Calculate proper aspect ratio for geographic coordinates
    """
    lat_range = max_lat - min_lat
    lon_range = max_lon - min_lon

    # Adjust for latitude (approximately cos(latitude) correction)
    lat_center = (max_lat + min_lat) / 2
    aspect_ratio = lon_range / lat_range * np.cos(np.radians(lat_center))

    return aspect_ratio

# Function to stack raster bands
def stack_raster_bands(year_folder, year_full=None):
    """
    Stack individual bands into a multi-band raster

    Parameters:
    year_folder (str): Folder name (e.g., '15')
    year_full (int, optional): Full year (e.g., 2015)

    Returns:
    str: Path to the stacked raster
    """
    # Define folder based on year
    folder = str(year_folder)

    # Define the full year for file naming
    if year_full is None:
        if year_folder == '15':
            year_full = 2015
        elif year_folder == '20':
            year_full = 2020
        elif year_folder == '23':
            year_full = 2023

    # Define file paths for the bands and indices
    file_prefix = f"Gurgaon_{year_full}_"
    bands = [
        f"{file_prefix}Blue.tif",
        f"{file_prefix}Green.tif",
        f"{file_prefix}Red.tif",
        f"{file_prefix}NIR.tif",
        f"{file_prefix}SWIR1.tif",
        f"{file_prefix}NDVI.tif",
        f"{file_prefix}NDBI.tif",
        f"{file_prefix}NDWI.tif"
    ]

    # Full paths
    band_paths = [os.path.join(folder, file) for file in bands]

    # Check if all files exist
    for path in band_paths:
        if not os.path.exists(path):
            print(f"Warning: {path} does not exist!")

    # Output path for stacked raster
    output_path = os.path.join(folder, f"Gurgaon_{year_full}_stack.tif")

    # Get metadata from first band
    with rasterio.open(band_paths[0]) as src:
        meta = src.meta

    # Update metadata for the stacked raster
    meta.update(count=len(band_paths))

    # Create the stacked raster
    with rasterio.open(output_path, 'w', **meta) as dst:
        for i, path in enumerate(band_paths, start=1):
            try:
                with rasterio.open(path) as src:
                    dst.write(src.read(1), i)
                    print(f"Added band {i}: {os.path.basename(path)}")
            except Exception as e:
                print(f"Error reading {path}: {e}")

    print(f"Stacked raster saved to: {output_path}")
    return output_path

# Function to display a multi-band raster with lat/lon coordinates
def display_multiband_raster(raster_path):
    """
    Display a multi-band raster with information about each band and proper coordinates

    Parameters:
    raster_path (str): Path to the raster file
    """
    with rasterio.open(raster_path) as src:
        # Print basic information
        print(f"Raster: {os.path.basename(raster_path)}")
        print(f"Dimensions: {src.width} x {src.height} pixels")
        print(f"Number of bands: {src.count}")
        print(f"Coordinate Reference System: {src.crs}")

        # Get lat/lon extent
        min_lon, max_lon, min_lat, max_lat = get_lat_lon_extent(raster_path)
        print(f"Geographic Extent:")
        print(f"  Longitude: {min_lon:.4f}° to {max_lon:.4f}°")
        print(f"  Latitude: {min_lat:.4f}° to {max_lat:.4f}°")

        # Calculate proper aspect ratio
        aspect_ratio = calculate_aspect_ratio(min_lon, max_lon, min_lat, max_lat)
        base_height = 4
        fig_width = base_height * aspect_ratio * 4  # 4 columns

        # Read all bands
        data = src.read()

        # Calculate statistics for each band
        for b in range(src.count):
            band_data = data[b]
            # Handle NaN values for statistics
            valid_data = band_data[~np.isnan(band_data)]
            if len(valid_data) > 0:
                min_val = np.min(valid_data)
                max_val = np.max(valid_data)
                mean_val = np.mean(valid_data)
                std_val = np.std(valid_data)
                nan_count = np.isnan(band_data).sum()
                nan_percent = (nan_count / band_data.size) * 100

                print(f"Band {b+1} Statistics:")
                print(f"  Min: {min_val:.6f}, Max: {max_val:.6f}")
                print(f"  Mean: {mean_val:.6f}, Std Dev: {std_val:.6f}")
                print(f"  NaN values: {nan_count} ({nan_percent:.2f}%)")
            else:
                print(f"Band {b+1}: No valid data")

        # Create a figure with subplots for each band
        n_bands = src.count
        fig, axes = plt.subplots(2, 4, figsize=(fig_width, base_height * 2))
        axes = axes.flatten()

        # Band names for labels
        band_names = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'NDVI', 'NDBI', 'NDWI']

        # Plot each band with proper coordinates
        for i in range(n_bands):
            band_data = data[i]
            # Create a masked array to handle NaN values
            masked_data = np.ma.masked_array(band_data, np.isnan(band_data))

            # Get percentiles for color scaling (avoid extreme values)
            valid_data = band_data[~np.isnan(band_data)]
            if len(valid_data) > 0:
                vmin = np.percentile(valid_data, 2)
                vmax = np.percentile(valid_data, 98)

                # Plot the band with geographic extent
                im = axes[i].imshow(masked_data, cmap='viridis', vmin=vmin, vmax=vmax,
                                   extent=[min_lon, max_lon, min_lat, max_lat], aspect='equal')

                # Add colorbar
                cbar = plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
                cbar.set_label('Value')

                # Add title
                if i < len(band_names):
                    axes[i].set_title(band_names[i])
                else:
                    axes[i].set_title(f"Band {i+1}")

                # Add coordinate labels
                axes[i].set_xlabel('Longitude (°)')
                axes[i].set_ylabel('Latitude (°)')
            else:
                axes[i].text(0.5, 0.5, 'No valid data', ha='center', va='center')
                axes[i].set_title(f"Band {i+1}")

        plt.tight_layout()
        plt.savefig(f"multiband_display_{os.path.splitext(os.path.basename(raster_path))[0]}.png",
                   dpi=300, bbox_inches='tight', format='png')
        plt.show()

        # Create an RGB composite with proper coordinates
        if src.count >= 3:
            # Calculate figure size for RGB composite
            rgb_fig_width = base_height * aspect_ratio

            # Create a figure for RGB composite
            plt.figure(figsize=(rgb_fig_width, base_height))

            # Extract Red, Green, Blue bands (actual indices 2,1,0)
            rgb = np.zeros((src.height, src.width, 3), dtype=np.float32)

            # Scale bands for display
            for idx, band_idx in enumerate([2, 1, 0]):  # Red, Green, Blue
                if band_idx < src.count:
                    band_data = data[band_idx]
                    valid_data = band_data[~np.isnan(band_data)]
                    if len(valid_data) > 0:
                        vmin = np.percentile(valid_data, 2)
                        vmax = np.percentile(valid_data, 98)
                        rgb[:,:,idx] = np.clip((band_data - vmin) / (vmax - vmin), 0, 1)

            # Create a mask for NaN values
            mask = np.isnan(data[0])
            rgb[mask, :] = 0

            # Display RGB composite with proper coordinates
            plt.imshow(rgb, extent=[min_lon, max_lon, min_lat, max_lat], aspect='equal')
            plt.title("RGB Composite (Red, Green, Blue)")
            plt.xlabel('Longitude (°)')
            plt.ylabel('Latitude (°)')
            plt.savefig(f"RGB_composite_{os.path.splitext(os.path.basename(raster_path))[0]}.png",
                       dpi=300, bbox_inches='tight', format='png')
            plt.show()

# Function to extract training data from stacked raster
def extract_training_data(year_folder, year_full=None):
    """
    Extract training data from stacked raster using the CSV points

    Parameters:
    year_folder (str): Folder name (e.g., '15')
    year_full (int, optional): Full year (e.g., 2015)

    Returns:
    tuple: X_train (features), y_train (labels), features_df (DataFrame)
    """
    # Define folder based on year
    folder = str(year_folder)

    # Define the full year for file naming
    if year_full is None:
        if year_folder == '15':
            year_full = 2015
        elif year_folder == '20':
            year_full = 2020
        elif year_folder == '23':
            year_full = 2023

    # Define paths
    stack_path = os.path.join(folder, f"Gurgaon_{year_full}_stack.tif")

    # Try different possible CSV file names
    possible_csv_paths = [
        os.path.join(folder, f"Training_Points_{year_full}.csv"),
        os.path.join(folder, f"Training_Points_{year_folder}.csv"),
        os.path.join(folder, f"training_points_{year_full}.csv"),
        os.path.join(folder, f"training_points_{year_folder}.csv"),
        os.path.join(folder, "Training_Points.csv"),
        os.path.join(folder, "training_points.csv")
    ]

    csv_path = None
    for path in possible_csv_paths:
        if os.path.exists(path):
            csv_path = path
            break

    if csv_path is None:
        raise FileNotFoundError(f"Training points CSV not found in folder {folder}")

    # Check if stacked raster exists, create it if not
    if not os.path.exists(stack_path):
        print(f"Stacked raster not found, creating it now...")
        stack_raster_bands(year_folder, year_full)

    # Read training points CSV
    points_df = pd.read_csv(csv_path)
    print(f"Read {len(points_df)} training points from {csv_path}")
    print(f"Columns in CSV: {list(points_df.columns)}")
    print(f"First few rows:\n{points_df.head()}")

    # Check if required columns exist, normalize column names
    required_columns = ['class', 'longitude', 'latitude']
    for col in required_columns:
        if col not in points_df.columns and col.capitalize() in points_df.columns:
            points_df[col] = points_df[col.capitalize()]

    # Verify all required columns exist
    for col in required_columns:
        if col not in points_df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV")

    # Open the stacked raster
    with rasterio.open(stack_path) as src:
        # Get number of bands
        num_bands = src.count
        print(f"Stacked raster has {num_bands} bands")

        # Initialize arrays for extracted features and labels
        X_train = []
        y_train = []

        # Extract values at each point location
        for idx, point in points_df.iterrows():
            try:
                # Get coordinates
                x, y = point['longitude'], point['latitude']

                # Sample values at these coordinates
                sample = list(src.sample([(x, y)]))[0]

                # Check if sample has any NaN values
                if not np.isnan(sample).any():
                    X_train.append(sample)
                    y_train.append(point['class'])
            except Exception as e:
                print(f"Error sampling point {idx}: {e}")

        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Print extraction results
        print(f"Extracted {X_train.shape[0]} valid samples with {X_train.shape[1]} features")

        # Create a DataFrame for better inspection
        band_names = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'NDVI', 'NDBI', 'NDWI']
        if len(band_names) != X_train.shape[1]:
            band_names = [f'Band_{i+1}' for i in range(X_train.shape[1])]

        features_df = pd.DataFrame(X_train, columns=band_names)
        features_df['class'] = y_train
        features_df['class_name'] = features_df['class'].map(class_names)

        print("Sample of extracted features:")
        print(features_df.head())

        return X_train, y_train, features_df

# Function to verify training data
def verify_training_data(features_df):
    """
    Verify and visualize the extracted training data

    Parameters:
    features_df (DataFrame): DataFrame with extracted features and class labels
    """
    if features_df is None or len(features_df) == 0:
        print("No valid training data to verify")
        return

    # Print class distribution
    class_distribution = features_df['class_name'].value_counts()
    print("\nClass Distribution:")
    print(class_distribution)

    # Calculate statistics for each feature by class
    stats_by_class = features_df.groupby('class_name').describe()
    print("\nFeature Statistics by Class:")
    print(stats_by_class)

    # Visualize class distribution
    plt.figure(figsize=(10, 6))
    class_distribution.plot(kind='bar', color='skyblue')
    plt.title('Distribution of Training Samples by Class')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight', format='png')
    plt.show()

    # Visualize feature distributions by class
    feature_cols = features_df.columns[:-2]  # Exclude 'class' and 'class_name'

    # Box plots for each feature by class
    for feature in feature_cols:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='class_name', y=feature, data=features_df)
        plt.title(f'{feature} Distribution by Class')
        plt.xlabel('Class')
        plt.ylabel(feature)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{feature}_distribution.png', dpi=300, bbox_inches='tight', format='png')
        plt.show()

    # Pair plot for selected features (first 4)
    if len(feature_cols) > 3:
        selected_features = list(feature_cols[:4]) + ['class_name']
        plt.figure(figsize=(15, 15))
        sns.pairplot(features_df[selected_features], hue='class_name', height=2.5)
        plt.suptitle('Pairwise Feature Relationships by Class', y=1.02)
        plt.tight_layout()
        plt.savefig('pairwise_features.png', dpi=300, bbox_inches='tight', format='png')
        plt.show()

# Main function to execute Phase 1
def execute_phase1(year_folders):
    """
    Execute Phase 1: Data Preparation for specified years

    Parameters:
    year_folders (list): List of folder names (e.g., ['15', '20', '23'])
    """
    results = {}

    for folder in year_folders:
        year_full = None
        if folder == '15': year_full = 2015
        elif folder == '20': year_full = 2020
        elif folder == '23': year_full = 2023

        print(f"\n{'='*50}")
        print(f"Processing Year {year_full} (Folder: {folder})")
        print(f"{'='*50}")

        try:
            # Task 1: Stack raster bands
            print("\nTask 1: Stacking raster bands...")
            stack_path = stack_raster_bands(folder, year_full)

            # Display the stacked raster
            print("\nDisplaying stacked raster...")
            display_multiband_raster(stack_path)

            # Task 2: Extract training data
            print("\nTask 2: Extracting training data...")
            X_train, y_train, features_df = extract_training_data(folder, year_full)

            # Task 3: Verify training data
            print("\nTask 3: Verifying training data...")
            verify_training_data(features_df)

            # Store results
            results[year_full] = {
                'X_train': X_train,
                'y_train': y_train,
                'features_df': features_df,
                'stack_path': stack_path
            }

            print(f"\nPhase 1 completed successfully for year {year_full}")

        except Exception as e:
            print(f"Error processing year {year_full}: {str(e)}")
            import traceback
            traceback.print_exc()

    return results

# Execute Phase 1 for all years
if __name__ == "__main__":
    year_folders = ['15', '20', '23']
    phase1_results = execute_phase1(year_folders)
    print("\nPhase 1 completed with latitude/longitude coordinates!")
    print("All visualizations saved as PNG files with proper geographic referencing.")


# In[5]:


# Updated Phase 2 Code with Stratified Cross-Validation
# Land Cover Classification for Gurgaon Urban Expansion Study

# Required imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                           cohen_kappa_score, roc_curve, auc)
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from matplotlib.colors import ListedColormap

# Define class names and colors
class_names = {
    0: 'Urban',
    1: 'Vegetation',
    2: 'Water',
    3: 'Barren'
}

colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00']  # Red, Green, Blue, Yellow
cmap = ListedColormap(colors)

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

def train_and_evaluate_with_stratified_cv(X, y, year, n_splits=5):
    """
    Train Random Forest using stratified cross-validation

    Parameters:
    X: Feature matrix
    y: Labels
    year: Year for labeling
    n_splits: Number of CV folds

    Returns:
    Dictionary with trained model and evaluation metrics
    """
    print(f"\nTraining model for year {year} using {n_splits}-fold stratified cross-validation...")

    # Create stratified k-fold object
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Initialize metrics storage
    cv_scores = []
    cv_kappas = []
    cv_confusion_matrices = []
    feature_importances = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []

    # Perform cross-validation
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"Processing fold {fold_idx + 1}/{n_splits}...")
        print(f"  Train size: {len(train_idx)}, Test size: {len(test_idx)}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Print class distribution in fold
        train_classes, train_counts = np.unique(y_train, return_counts=True)
        test_classes, test_counts = np.unique(y_test, return_counts=True)
        print(f"  Train class distribution: {dict(zip(train_classes, train_counts))}")
        print(f"  Test class distribution: {dict(zip(test_classes, test_counts))}")

        # Initialize and train model
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            oob_score=True,
            class_weight='balanced'  # Handle class imbalance
        )

        # Train on actual data
        rf.fit(X_train, y_train)

        # Make predictions
        y_pred = rf.predict(X_test)
        y_proba = rf.predict_proba(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=range(len(class_names)))

        # Store results
        cv_scores.append(accuracy)
        cv_kappas.append(kappa)
        cv_confusion_matrices.append(cm)
        feature_importances.append(rf.feature_importances_)

        print(f"  Fold {fold_idx + 1} - Accuracy: {accuracy:.4f}, Kappa: {kappa:.4f}")

        # Store predictions for overall metrics
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        if len(all_y_proba) == 0:
            all_y_proba = y_proba
        else:
            all_y_proba = np.vstack([all_y_proba, y_proba])

    # Convert lists to arrays
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    # Calculate overall metrics
    mean_accuracy = np.mean(cv_scores)
    std_accuracy = np.std(cv_scores)
    mean_kappa = np.mean(cv_kappas)
    std_kappa = np.std(cv_kappas)

    # Average confusion matrix
    avg_cm = np.mean(cv_confusion_matrices, axis=0)

    # Average feature importances
    avg_feature_importances = np.mean(feature_importances, axis=0)

    # Train final model on all data
    print("\nTraining final model on all data...")
    final_rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        oob_score=True,
        class_weight='balanced'
    )

    final_rf.fit(X, y)

    # Calculate OOB score and training accuracy
    oob_error = 1 - final_rf.oob_score_
    y_train_pred = final_rf.predict(X)
    train_accuracy = accuracy_score(y, y_train_pred)

    # Print results
    print(f"\nCross-validation results for {year}:")
    print(f"Mean CV accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")
    print(f"Mean CV kappa: {mean_kappa:.4f} (+/- {std_kappa:.4f})")
    print(f"Training accuracy (full dataset): {train_accuracy:.4f}")
    print(f"OOB error: {oob_error:.4f}")

    # Print class distribution analysis
    print("\nClass distribution across all folds:")
    overall_cm = confusion_matrix(all_y_true, all_y_pred)
    class_accuracies = overall_cm.diagonal() / overall_cm.sum(axis=1)
    for i, class_name in class_names.items():
        print(f"  {class_name}: Accuracy = {class_accuracies[i]:.4f}")

    # Create results dictionary
    results = {
        'final_model': final_rf,
        'cv_scores': cv_scores,
        'cv_kappas': cv_kappas,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'mean_kappa': mean_kappa,
        'std_kappa': std_kappa,
        'oob_error': oob_error,
        'train_accuracy': train_accuracy,
        'avg_confusion_matrix': avg_cm,
        'avg_feature_importances': avg_feature_importances,
        'all_y_true': all_y_true,
        'all_y_pred': all_y_pred,
        'all_y_proba': all_y_proba,
        'class_accuracies': class_accuracies
    }

    return results

def visualize_cv_results(cv_results, year):
    """Visualize cross-validation results"""
    print(f"\nVisualizing results for year {year}...")

    # Plot CV scores
    plt.figure(figsize=(12, 5))
    folds = range(1, len(cv_results['cv_scores']) + 1)

    plt.subplot(1, 2, 1)
    plt.plot(folds, cv_results['cv_scores'], 'bo-', label='Accuracy', markersize=8)
    plt.axhline(y=cv_results['mean_accuracy'], color='r', linestyle='--',
                label=f'Mean ({cv_results["mean_accuracy"]:.3f})')
    plt.fill_between(folds,
                     cv_results['mean_accuracy'] - cv_results['std_accuracy'],
                     cv_results['mean_accuracy'] + cv_results['std_accuracy'],
                     alpha=0.2, color='r')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title(f'Stratified CV Accuracy - {year}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(folds, cv_results['cv_kappas'], 'go-', label='Kappa', markersize=8)
    plt.axhline(y=cv_results['mean_kappa'], color='r', linestyle='--',
                label=f'Mean ({cv_results["mean_kappa"]:.3f})')
    plt.fill_between(folds,
                     cv_results['mean_kappa'] - cv_results['std_kappa'],
                     cv_results['mean_kappa'] + cv_results['std_kappa'],
                     alpha=0.2, color='r')
    plt.xlabel('Fold')
    plt.ylabel('Kappa')
    plt.title(f'Stratified CV Kappa - {year}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Plot average confusion matrix
    plt.figure(figsize=(10, 8))
    cm = cv_results['avg_confusion_matrix'].astype(int)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(class_names.values()),
                yticklabels=list(class_names.values()))
    plt.title(f'Average Confusion Matrix - {year}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

    # Plot class-wise accuracy
    plt.figure(figsize=(10, 6))
    classes = list(class_names.values())
    accuracies = cv_results['class_accuracies']

    bars = plt.bar(classes, accuracies, color=['#FF0000', '#00FF00', '#0000FF', '#FFFF00'])
    plt.ylabel('Accuracy')
    plt.title(f'Class-wise Accuracy - {year}')
    plt.ylim(0, 1.1)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')

    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot feature importance
    feature_names = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'NDVI', 'NDBI', 'NDWI']
    importances = cv_results['avg_feature_importances']
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.title(f'Average Feature Importance - {year}')
    plt.ylabel('Importance')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def calculate_roc_curves(cv_results, year):
    """Calculate and plot ROC curves for each class"""
    y_true = cv_results['all_y_true']
    y_proba = cv_results['all_y_proba']

    plt.figure(figsize=(10, 8))
    auc_values = []

    for i, class_name in class_names.items():
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        auc_values.append(roc_auc)

        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {year}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    return np.mean(auc_values)

def execute_phase2_stratified(data_by_year):
    """
    Main function to run Phase 2 with stratified cross-validation
    """
    print("\n=== Starting Phase 2: Land Cover Classification with Stratified CV ===\n")

    # Initialize results storage
    results = {}
    summary_data = []

    # Process each year
    for year, data in data_by_year.items():
        print(f"\n{'='*50}")
        print(f"Processing year {year}")
        print(f"{'='*50}")

        # Get data
        X = data['X']
        y = data['y']
        feature_names = data['feature_names']

        print(f"Data shape: {X.shape}")
        print(f"Number of samples: {X.shape[0]}")
        print(f"Number of classes: {len(np.unique(y))}")

        # Print class distribution
        classes, counts = np.unique(y, return_counts=True)
        print("\nOverall class distribution:")
        for c, count in zip(classes, counts):
            print(f"  {class_names[c]}: {count} samples ({count/len(y)*100:.1f}%)")

        # Train and evaluate with stratified cross-validation
        cv_results = train_and_evaluate_with_stratified_cv(X, y, year)

        # Visualize results
        visualize_cv_results(cv_results, year)

        # Calculate ROC curves and AUC
        avg_auc = calculate_roc_curves(cv_results, year)
        cv_results['avg_auc'] = avg_auc

        # Save model
        model_path = os.path.join('models', f'rf_model_stratified_{year}.joblib')
        dump(cv_results['final_model'], model_path)
        print(f"Saved model to {model_path}")

        # Store results
        results[year] = cv_results

        # Add to summary data
        summary_data.append({
            'Year': year,
            'Training Accuracy': cv_results['train_accuracy'],
            'Testing Accuracy': cv_results['mean_accuracy'],
            'OOB Error': cv_results['oob_error'],
            'Kappa': cv_results['mean_kappa'],
            'Average AUC': avg_auc
        })

    # Create summary table
    summary_df = pd.DataFrame(summary_data)
    print("\n=== Summary of Results ===")
    print(summary_df)

    # Plot year-wise comparison
    plt.figure(figsize=(12, 6))

    years = [str(year) for year in summary_df['Year']]
    x = np.arange(len(years))
    width = 0.15

    # Plot bars
    plt.bar(x - width*2, summary_df['Training Accuracy'], width,
            label='Training Accuracy', color='skyblue')
    plt.bar(x - width, summary_df['Testing Accuracy'], width,
            label='Testing Accuracy', color='lightgreen')
    plt.bar(x, summary_df['OOB Error'], width,
            label='OOB Error', color='salmon')
    plt.bar(x + width, summary_df['Kappa'], width,
            label='Kappa', color='plum')
    plt.bar(x + width*2, summary_df['Average AUC'], width,
            label='Average AUC', color='gold')

    plt.xlabel('Year')
    plt.ylabel('Score')
    plt.title('Model Performance Across Years (Stratified CV)')
    plt.xticks(x, years)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Save results
    results_path = os.path.join('results', 'phase2_stratified_results.joblib')
    dump(results, results_path)
    print(f"Saved Phase 2 results to {results_path}")

    return results, summary_df

# Main execution
if __name__ == "__main__":
    # Check if Phase 1 results exist
    if 'phase1_results' in globals():
        print("Using existing Phase 1 results")

        # Convert Phase 1 results to format expected by Phase 2
        data_by_year = {}

        for year, data in phase1_results.items():
            data_by_year[year] = {
                'X': data['X_train'],
                'y': data['y_train'],
                'feature_names': ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'NDVI', 'NDBI', 'NDWI']
            }

        # Execute Phase 2 with stratified CV
        results, summary_df = execute_phase2_stratified(data_by_year)

        print("\n=== Phase 2 (Stratified CV) completed successfully ===")
        print("Models and results saved for use in Phase 3")
    else:
        print("Phase 1 results not found. Please run Phase 1 first.")


# In[19]:


# Updated Phase 3 Code for Stratified CV Models
# Image Classification for Gurgaon Urban Expansion Study
# WITH LATITUDE/LONGITUDE COORDINATES

# Required imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import rasterio
from rasterio.plot import show
from joblib import load
import time
from skimage.exposure import rescale_intensity
import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# Class definitions
class_names = {
    0: 'Urban',
    1: 'Vegetation',
    2: 'Water',
    3: 'Barren'
}

# Define color map for visualization
colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00']  # Red, Green, Blue, Yellow
cmap = ListedColormap(colors)

# Define satellite information by year
satellite_info = {
    2015: {'satellite': 'Landsat', 'resolution_m': 30.0},
    2020: {'satellite': 'Sentinel', 'resolution_m': 10.0},
    2023: {'satellite': 'Sentinel', 'resolution_m': 10.0}
}

def get_lat_lon_extent(raster_path):
    """
    Get the latitude and longitude extent from a raster file

    Parameters:
    raster_path (str): Path to the raster file

    Returns:
    tuple: (min_lon, max_lon, min_lat, max_lat)
    """
    with rasterio.open(raster_path) as src:
        # Get the bounds in the CRS of the raster
        bounds = src.bounds

        # Extract coordinates
        min_lon, min_lat, max_lon, max_lat = bounds.left, bounds.bottom, bounds.right, bounds.top

        return min_lon, max_lon, min_lat, max_lat

def calculate_aspect_ratio(min_lon, max_lon, min_lat, max_lat):
    """
    Calculate proper aspect ratio for geographic coordinates
    """
    lat_range = max_lat - min_lat
    lon_range = max_lon - min_lon

    # Adjust for latitude (approximately cos(latitude) correction)
    lat_center = (max_lat + min_lat) / 2
    aspect_ratio = lon_range / lat_range * np.cos(np.radians(lat_center))

    return aspect_ratio

def get_pixel_area_km2(year):
    """
    Calculate pixel area in square kilometers based on satellite resolution
    """
    if year in satellite_info:
        resolution_m = satellite_info[year]['resolution_m']
        area_km2 = (resolution_m * resolution_m) / 1000000.0
        return area_km2
    else:
        raise ValueError(f"Satellite information not available for year {year}")

# Function to classify the entire raster image
def classify_image(year_folder, model=None, year_full=None, output_path=None, batch_size=10000):
    """
    Classify the entire raster image using the trained Random Forest model from Stratified CV
    """
    # Define folder based on year
    folder = str(year_folder)

    # Define the full year for file naming
    if year_full is None:
        if year_folder == '15':
            year_full = 2015
        elif year_folder == '20':
            year_full = 2020
        elif year_folder == '23':
            year_full = 2023

    # Display satellite information
    if year_full in satellite_info:
        sat_info = satellite_info[year_full]
        print(f"Processing {sat_info['satellite']} data (Resolution: {sat_info['resolution_m']}m)")

    # Define paths
    stack_path = os.path.join(folder, f"Gurgaon_{year_full}_stack.tif")

    if output_path is None:
        output_path = os.path.join(folder, f"RFclassified_{year_full}.tif")

    # Load model if not provided - IMPORTANT CHANGE: Use stratified model
    if model is None:
        model_path = os.path.join("models", f"rf_model_stratified_{year_full}.joblib")
        if not os.path.exists(model_path):
            # If stratified model doesn't exist, try regular model
            model_path = os.path.join("models", f"rf_model_{year_full}.joblib")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            print(f"Warning: Using regular model instead of stratified model")

        print(f"Loading stratified CV model from {model_path}...")
        model = load(model_path)

    print(f"Classifying raster: {stack_path}")
    print(f"Output will be saved to: {output_path}")

    # Start timer
    start_time = time.time()

    # Open the stacked raster
    with rasterio.open(stack_path) as src:
        # Get metadata for output
        out_meta = src.meta.copy()
        out_meta.update(count=1, dtype='uint8', nodata=255)

        # Read all bands
        data = src.read()
        bands, rows, cols = data.shape

        print(f"Raster shape: {data.shape}")

        # Create a mask for valid data (not NaN in ANY band)
        valid_mask = np.ones((rows, cols), dtype=bool)

        # For each band, update mask to exclude NaN values
        for b in range(min(3, bands)):
            valid_mask = valid_mask & ~np.isnan(data[b])

        # Count valid pixels
        valid_count = np.sum(valid_mask)
        print(f"Valid pixels: {valid_count} out of {rows*cols} ({(valid_count/(rows*cols))*100:.2f}%)")

        # Create output array with no-data value
        result = np.ones((rows, cols), dtype=np.uint8) * 255

        # Process valid pixels in batches
        if valid_count > 0:
            # Get indices of valid pixels
            valid_indices = np.where(valid_mask)

            # Process in batches
            for i in range(0, valid_count, batch_size):
                # Get batch indices
                batch_end = min(i + batch_size, valid_count)
                batch_indices = (
                    valid_indices[0][i:batch_end],
                    valid_indices[1][i:batch_end]
                )

                # Extract feature values for this batch
                X_batch = np.zeros((batch_end - i, bands), dtype=np.float32)

                for b in range(bands):
                    X_batch[:, b] = data[b][batch_indices]

                # Replace any NaN values with 0
                X_batch = np.nan_to_num(X_batch, nan=0.0)

                # Predict classes
                try:
                    if (i % 50000) == 0:  # Print progress every 50k pixels
                        print(f"Classifying batch {i+1}-{batch_end} of {valid_count} pixels...")
                    pred_batch = model.predict(X_batch)

                    # Assign predictions to result array
                    result[batch_indices] = pred_batch
                except Exception as e:
                    print(f"Error classifying batch: {str(e)}")

            # Calculate class distribution
            for class_id, class_name in class_names.items():
                pixel_count = np.sum(result == class_id)
                percent = (pixel_count / valid_count) * 100
                print(f"Class {class_id} ({class_name}): {pixel_count} pixels ({percent:.2f}%)")

        # Save the result
        with rasterio.open(output_path, 'w', **out_meta) as dst:
            dst.write(result, 1)

        # End timer
        elapsed_time = time.time() - start_time
        print(f"Classification completed in {elapsed_time:.2f} seconds")
        print(f"Classified image saved to: {output_path}")

        return output_path

# Function to visualize the classified image with lat/lon coordinates
def visualize_classification(classified_path, year):
    """
    Visualize a classified image with proper legend and geographic coordinates
    """
    print(f"Visualizing classification for year {year}...")

    # Display satellite information
    if year in satellite_info:
        sat_info = satellite_info[year]
        print(f"Using {sat_info['satellite']} data (Resolution: {sat_info['resolution_m']}m)")

        # Get correct pixel area based on satellite resolution
        pixel_area_km2 = get_pixel_area_km2(year)
        print(f"Pixel area: {pixel_area_km2:.6f} km²")

    # Get coordinates for proper display
    min_lon, max_lon, min_lat, max_lat = get_lat_lon_extent(classified_path)
    aspect_ratio = calculate_aspect_ratio(min_lon, max_lon, min_lat, max_lat)

    # Calculate figure size for proper aspect ratio
    base_height = 10
    fig_width = base_height * aspect_ratio

    # Open the classified image
    with rasterio.open(classified_path) as src:
        # Read the data
        data = src.read(1)

        # Create a masked array to hide no-data values
        masked_data = np.ma.masked_where(data == 255, data)

        # Create figure with proper dimensions
        plt.figure(figsize=(fig_width, base_height))

        # Plot the classification with geographic coordinates
        img = plt.imshow(masked_data, cmap=cmap, vmin=0, vmax=3,
                        extent=[min_lon, max_lon, min_lat, max_lat], aspect='equal')

        # Add colorbar with class labels
        cbar = plt.colorbar(img, ticks=[0.4, 1.2, 2.0, 2.8], shrink=0.8)
        cbar.set_ticklabels(list(class_names.values()))

        # Add coordinate labels
        plt.xlabel('Longitude (°)', fontsize=12)
        plt.ylabel('Latitude (°)', fontsize=12)

        # Add title with satellite information
        if year in satellite_info:
            sat_name = satellite_info[year]['satellite']
            plt.title(f'Land Use Land Cover Classification {year} ({sat_name}) - Stratified CV', fontsize=14)
        else:
            plt.title(f'Land Use Land Cover Classification {year} - Stratified CV', fontsize=14)

        # Add coordinate information text
        coord_text = f'Extent: {min_lon:.3f}° to {max_lon:.3f}°E, {min_lat:.3f}° to {max_lat:.3f}°N'
        plt.annotate(coord_text, xy=(0.02, 0.95), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                     fontsize=10)

        plt.tight_layout()
        plt.savefig(f'LULC_stratified_{year}.png', dpi=300, bbox_inches='tight', format='png')
        plt.show()

        # Print class distribution
        print("\nClass Distribution:")
        for class_id, class_name in class_names.items():
            pixel_count = np.sum(data == class_id)
            percent = (pixel_count / np.sum(data != 255)) * 100
            print(f"Class {class_id} ({class_name}): {pixel_count} pixels ({percent:.2f}%)")

        # Calculate area in square kilometers based on satellite resolution
        pixel_area_km2 = get_pixel_area_km2(year)

        # Calculate area for each class
        print(f"\nEstimated Area (square kilometers) - {satellite_info[year]['satellite']} resolution:")
        for class_id, class_name in class_names.items():
            pixel_count = np.sum(data == class_id)
            area_km2 = pixel_count * pixel_area_km2
            print(f"Class {class_id} ({class_name}): {area_km2:.2f} km²")

        # Print geographic information
        print(f"\nGeographic Information:")
        print(f"Extent: {min_lon:.4f}° to {max_lon:.4f}°E, {min_lat:.4f}° to {max_lat:.4f}°N")
        print(f"Coverage: {(max_lon-min_lon):.4f}° longitude × {(max_lat-min_lat):.4f}° latitude")

# Function to create an RGB composite with classification overlay and coordinates
def create_composite_visualization(year_folder, year_full=None, classified_path=None, opacity=0.5):
    """
    Create a composite visualization with RGB bands and classification overlay
    WITH GEOGRAPHIC COORDINATES
    """
    # Define folder based on year
    folder = str(year_folder)

    # Define the full year for file naming
    if year_full is None:
        if year_folder == '15':
            year_full = 2015
        elif year_folder == '20':
            year_full = 2020
        elif year_folder == '23':
            year_full = 2023

    # Display satellite information
    if year_full in satellite_info:
        sat_info = satellite_info[year_full]
        print(f"Creating composite visualization for {sat_info['satellite']} data, {year_full} (Resolution: {sat_info['resolution_m']}m)")
    else:
        print(f"Creating composite visualization for year {year_full}")

    # Define paths
    stack_path = os.path.join(folder, f"Gurgaon_{year_full}_stack.tif")

    if classified_path is None:
        classified_path = os.path.join(folder, f"RFclassified_{year_full}.tif")

    # Get coordinates for proper display
    min_lon, max_lon, min_lat, max_lat = get_lat_lon_extent(classified_path)
    aspect_ratio = calculate_aspect_ratio(min_lon, max_lon, min_lat, max_lat)

    # Calculate figure size for proper aspect ratio
    base_height = 12
    fig_width = base_height * aspect_ratio

    # Open the stacked raster
    with rasterio.open(stack_path) as src:
        # Read RGB bands (assuming Red=2, Green=1, Blue=0)
        red = src.read(3)  # Index 3 is the 'Red' band (index 0-based)
        green = src.read(2)  # Index 2 is the 'Green' band
        blue = src.read(1)  # Index 1 is the 'Blue' band

        # Create an RGB array
        rgb = np.dstack((red, green, blue))

        # Handle NaN values
        rgb = np.nan_to_num(rgb, nan=0.0)

        # Rescale intensity values for better visualization
        for i in range(3):
            valid_data = rgb[:,:,i][rgb[:,:,i] > 0]
            if len(valid_data) > 0:
                p2, p98 = np.percentile(valid_data, (2, 98))
                rgb[:,:,i] = rescale_intensity(rgb[:,:,i], in_range=(p2, p98), out_range=(0, 1))

    # Open the classified image
    with rasterio.open(classified_path) as src:
        # Read the classification
        classification = src.read(1)

        # Create a masked array to hide no-data values
        mask = (classification == 255)
        masked_classification = np.ma.masked_where(mask, classification)

    # Create figure with proper dimensions
    fig, ax = plt.subplots(figsize=(fig_width, base_height))

    # Plot RGB image with geographic coordinates
    ax.imshow(rgb, extent=[min_lon, max_lon, min_lat, max_lat], aspect='equal')

    # Plot classification overlay with transparency
    class_overlay = ax.imshow(masked_classification, cmap=cmap, alpha=opacity, vmin=0, vmax=3,
                             extent=[min_lon, max_lon, min_lat, max_lat], aspect='equal')

    # Add colorbar with class labels
    cbar = plt.colorbar(class_overlay, ax=ax, ticks=[0.4, 1.2, 2.0, 2.8], shrink=0.8)
    cbar.set_ticklabels(list(class_names.values()))

    # Add coordinate labels
    ax.set_xlabel('Longitude (°)', fontsize=12)
    ax.set_ylabel('Latitude (°)', fontsize=12)

    # Add title with satellite information
    if year_full in satellite_info:
        sat_name = satellite_info[year_full]['satellite']
        plt.title(f'RGB Composite with LULC Classification Overlay {year_full} ({sat_name}) - Stratified CV', fontsize=14)
    else:
        plt.title(f'RGB Composite with LULC Classification Overlay {year_full} - Stratified CV', fontsize=14)

    # Add coordinate information text
    coord_text = f'Extent: {min_lon:.3f}° to {max_lon:.3f}°E, {min_lat:.3f}° to {max_lat:.3f}°N'
    ax.annotate(coord_text, xy=(0.02, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                fontsize=10)

    plt.tight_layout()
    plt.savefig(f'RGB_overlay_stratified_{year_full}.png', dpi=300, bbox_inches='tight', format='png')
    plt.show()

# Main function to execute Phase 3 with Stratified CV models and coordinates
def execute_phase3_stratified(year_folders, phase2_results=None):
    """
    Execute Phase 3: Image Classification using Stratified CV models
    WITH FULL LATITUDE/LONGITUDE COORDINATE SUPPORT
    """
    print("\n" + "="*80)
    print("EXECUTING PHASE 3 WITH LATITUDE/LONGITUDE COORDINATES")
    print("Image Classification using Stratified CV models")
    print("="*80 + "\n")

    classified_paths = {}

    for folder in year_folders:
        year_full = None
        if folder == '15': year_full = 2015
        elif folder == '20': year_full = 2020
        elif folder == '23': year_full = 2023

        print(f"\n{'='*50}")
        print(f"Processing Year {year_full} (Folder: {folder})")
        print(f"Using Stratified CV Model")
        print(f"{'='*50}")

        # Display satellite information
        if year_full in satellite_info:
            sat_info = satellite_info[year_full]
            print(f"Satellite: {sat_info['satellite']}, Resolution: {sat_info['resolution_m']}m")
            print(f"Pixel area: {get_pixel_area_km2(year_full):.6f} km²")

        try:
            # Get model from Phase 2 results if available
            model = None
            if phase2_results and year_full in phase2_results:
                model = phase2_results[year_full]['final_model']
                print("Using model from Phase 2 results")

            # Task 1: Classify the entire image
            print("\nTask 1: Classifying the entire image...")
            classified_path = classify_image(folder, model, year_full)

            # Task 2: Visualize the classification with coordinates
            print("\nTask 2: Visualizing the classification with lat/lon coordinates...")
            visualize_classification(classified_path, year_full)

            # Task 3: Create RGB composite with classification overlay and coordinates
            print("\nTask 3: Creating RGB composite with coordinates...")
            create_composite_visualization(folder, year_full, classified_path)

            # Store the path to the classified image
            classified_paths[year_full] = classified_path

            print(f"\nPhase 3 completed successfully for year {year_full}")
            print("✅ All maps include proper latitude/longitude coordinates")
            print("✅ Geographic extent information displayed")
            print("✅ Proper aspect ratios prevent map elongation")

        except Exception as e:
            print(f"Error processing year {year_full}: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("PHASE 3 WITH COORDINATES COMPLETED SUCCESSFULLY!")
    print("All classification maps now include proper geographic referencing")
    print("="*80)

    return classified_paths

# Main execution
def main():
    """
    Main function to execute Phase 3 with coordinate support
    """
    year_folders = ['15', '20', '23']

    # Check if we have stratified CV results from Phase 2
    if 'phase2_stratified_results' in globals():
        print("Executing Phase 3 using Stratified CV models from Phase 2...")
        # If you saved the results from stratified CV Phase 2
        try:
            phase2_results = load('results/phase2_stratified_results.joblib')
            phase3_results = execute_phase3_stratified(year_folders, phase2_results)
        except FileNotFoundError:
            print("Phase 2 results file not found. Loading models from disk...")
            phase3_results = execute_phase3_stratified(year_folders)
    else:
        print("Executing Phase 3 using Stratified CV models from disk...")
        # This will load the stratified models from the models directory
        phase3_results = execute_phase3_stratified(year_folders)

    return phase3_results

# Execute when script is run directly
if __name__ == "__main__":
    results = main()
    print("\n🎉 Phase 3 Classification with Coordinates Complete!")
    print("All maps are now properly georeferenced with lat/lon coordinates!")

# Alternative usage:
# phase3_results = execute_phase3_stratified(['15', '20', '23'])


# In[7]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from rasterio.warp import reproject, Resampling
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# LULC class definitions
class_names = {
    0: 'Urban',
    1: 'Vegetation',
    2: 'Water',
    3: 'Barren'
}

# Define satellite resolutions by year
satellite_info = {
    2015: {'satellite': 'Landsat', 'resolution_m': 30.0},
    2020: {'satellite': 'Sentinel', 'resolution_m': 10.0},
    2023: {'satellite': 'Sentinel', 'resolution_m': 10.0}
}

def calculate_pixel_area(year):
    """
    Calculate pixel area in square kilometers based on nominal resolution

    Parameters:
    year (int): Year for reference - used to look up the satellite info

    Returns:
    float: Pixel area in square kilometers
    """
    # Use the nominal resolution directly
    nominal_resolution = satellite_info[year]['resolution_m']
    pixel_area_m2 = nominal_resolution * nominal_resolution
    pixel_area_km2 = pixel_area_m2 / 1000000.0

    print(f"Using nominal pixel area for {year} ({satellite_info[year]['satellite']}): {pixel_area_km2:.8f} km²")
    return pixel_area_km2

def resample_raster(src_path, target_path, output_path=None):
    """
    Resample a raster to match the dimensions of a target raster
    """
    print(f"Resampling {os.path.basename(src_path)} to match {os.path.basename(target_path)}...")

    # If output path is not provided, create one
    if output_path is None:
        src_dir = os.path.dirname(src_path)
        src_name = os.path.basename(src_path)
        output_path = os.path.join(src_dir, f"resampled_{src_name}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with rasterio.open(src_path) as src:
        src_data = src.read(1)
        src_meta = src.meta.copy()

        with rasterio.open(target_path) as target:
            # Get target dimensions
            target_height = target.height
            target_width = target.width
            target_transform = target.transform
            target_crs = target.crs

            # Log the details for debugging
            print(f"Source shape: {src.shape}, resolution: {src.res}")
            print(f"Target shape: {target.shape}, resolution: {target.res}")

            # Update metadata for resampled output
            dst_meta = src_meta.copy()
            dst_meta.update({
                'height': target_height,
                'width': target_width,
                'transform': target_transform,
                'crs': target_crs
            })

            # Create destination array
            dst_data = np.ones((target_height, target_width), dtype=src_data.dtype) * 255

            # Perform reprojection/resampling
            try:
                reproject(
                    source=src.read(1),
                    destination=dst_data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest
                )

                # Write resampled raster to disk
                with rasterio.open(output_path, 'w', **dst_meta) as dst:
                    dst.write(dst_data, 1)

                print(f"Resampled raster saved to: {output_path}")
                return output_path

            except Exception as e:
                print(f"Error during resampling: {str(e)}")
                return None

def create_transition_matrix(from_year_path, to_year_path, from_year, to_year, resample=True):
    """
    Create a transition matrix comparing land cover changes between two years
    """
    print(f"Creating transition matrix from {from_year} to {to_year}...")

    # Check if rasters have the same dimensions
    with rasterio.open(from_year_path) as src_from, rasterio.open(to_year_path) as src_to:
        from_shape = (src_from.height, src_from.width)
        to_shape = (src_to.height, src_to.width)

        print(f"From shape: {from_shape}, To shape: {to_shape}")

        # If shapes don't match and resample=True, resample the from_year raster to match to_year
        if from_shape != to_shape and resample:
            print("Rasters have different dimensions. Resampling...")

            # Create a directory for resampled rasters if it doesn't exist
            resample_dir = "resampled_rasters"
            if not os.path.exists(resample_dir):
                os.makedirs(resample_dir)

            # Define output path for resampled raster
            resampled_path = os.path.join(resample_dir, f"resampled_{os.path.basename(from_year_path)}")

            # Resample from_year raster to match to_year dimensions
            resampled_from_path = resample_raster(from_year_path, to_year_path, resampled_path)

            if resampled_from_path is None:
                print("Resampling failed. Cannot create transition matrix.")
                return None, None, None

            # Update the from_year path to use the resampled raster
            from_year_path = resampled_from_path

    # Calculate the pixel area for the "to_year" raster - using nominal resolution
    pixel_area_km2 = calculate_pixel_area(to_year)

    # Now process with potentially resampled data
    with rasterio.open(from_year_path) as src_from, rasterio.open(to_year_path) as src_to:
        # Read the data
        from_data = src_from.read(1)
        to_data = src_to.read(1)

        # Create masks for valid data (not nodata)
        valid_mask_from = (from_data != 255)
        valid_mask_to = (to_data != 255)

        # Combined valid mask (pixels valid in both rasters)
        valid_mask = valid_mask_from & valid_mask_to

        # Count valid pixels
        valid_count = np.sum(valid_mask)
        print(f"Valid pixels for comparison: {valid_count} out of {from_data.size} ({(valid_count/from_data.size)*100:.2f}%)")

        # Create an empty transition matrix (from class to class)
        n_classes = len(class_names)
        transition_matrix = np.zeros((n_classes, n_classes), dtype=np.int32)

        # Fill the transition matrix
        for from_class in range(n_classes):
            for to_class in range(n_classes):
                # Count pixels that transitioned from from_class to to_class
                transition_count = np.sum((from_data == from_class) & (to_data == to_class) & valid_mask)
                transition_matrix[from_class, to_class] = transition_count

        # Calculate areas using the calculated pixel area
        transition_area_matrix = transition_matrix * pixel_area_km2

    # Create DataFrames for better visualization
    transition_df = pd.DataFrame(
        transition_matrix,
        index=[f"{class_names[i]}" for i in range(n_classes)],
        columns=[f"{class_names[i]}" for i in range(n_classes)]
    )

    print("\nTransition Matrix (pixel counts):")
    print(transition_df)

    transition_area_df = pd.DataFrame(
        transition_area_matrix,
        index=[f"{class_names[i]}" for i in range(n_classes)],
        columns=[f"{class_names[i]}" for i in range(n_classes)]
    )

    print("\nTransition Matrix (area in km²):")
    print(transition_area_df)

    return transition_matrix, transition_area_matrix, pixel_area_km2

def visualize_transition_heatmap(transition_matrix, transition_area_matrix, from_year, to_year, output_dir):
    """
    Create and save heatmap visualizations of transition matrices

    Parameters:
    transition_matrix (ndarray): Matrix of pixel count transitions
    transition_area_matrix (ndarray): Matrix of area transitions in km²
    from_year (int): Starting year
    to_year (int): Ending year
    output_dir (str): Directory to save output visualizations
    """
    n_classes = len(class_names)

    # Create DataFrames with proper labels
    transition_df = pd.DataFrame(
        transition_matrix,
        index=[f"{class_names[i]}" for i in range(n_classes)],
        columns=[f"{class_names[i]}" for i in range(n_classes)]
    )

    transition_area_df = pd.DataFrame(
        transition_area_matrix,
        index=[f"{class_names[i]}" for i in range(n_classes)],
        columns=[f"{class_names[i]}" for i in range(n_classes)]
    )

    # Calculate percentage matrix for visualization
    # Each cell shows what percentage of the "from" class went to the "to" class
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    percent_matrix = (transition_matrix / row_sums) * 100

    percent_df = pd.DataFrame(
        percent_matrix,
        index=[f"{class_names[i]}" for i in range(n_classes)],
        columns=[f"{class_names[i]}" for i in range(n_classes)]
    )

    # Create figure with 3 subplots for different visualizations
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # 1. Pixel count heatmap
    sns.heatmap(
        transition_df,
        annot=True,
        fmt=",d",  # Format as integer with commas
        cmap="YlGnBu",
        ax=axes[0],
        cbar_kws={'label': 'Pixel Count'}
    )
    axes[0].set_title(f'Transition Matrix (Pixel Count): {from_year} to {to_year}')
    axes[0].set_xlabel('To Class (End State)')
    axes[0].set_ylabel('From Class (Start State)')

    # 2. Area (km²) heatmap
    sns.heatmap(
        transition_area_df,
        annot=True,
        fmt=".2f",  # Format as float with 2 decimal places
        cmap="YlOrRd",
        ax=axes[1],
        cbar_kws={'label': 'Area (km²)'}
    )
    axes[1].set_title(f'Transition Matrix (Area in km²): {from_year} to {to_year}')
    axes[1].set_xlabel('To Class (End State)')
    axes[1].set_ylabel('From Class (Start State)')

    # 3. Percentage heatmap
    sns.heatmap(
        percent_df,
        annot=True,
        fmt=".1f",  # Format as float with 1 decimal place
        cmap="viridis",
        ax=axes[2],
        cbar_kws={'label': 'Percentage of Start Class (%)'}
    )
    axes[2].set_title(f'Transition Matrix (Percentage): {from_year} to {to_year}')
    axes[2].set_xlabel('To Class (End State)')
    axes[2].set_ylabel('From Class (Start State)')

    plt.tight_layout()

    # Save figure
    heatmap_path = os.path.join(output_dir, f"transition_heatmap_{from_year}_{to_year}.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"Transition heatmap saved to: {heatmap_path}")

    return heatmap_path

def generate_summary_table(transition_matrix, transition_area_matrix, pixel_area_km2, from_year, to_year, class_names):
    """
    Generate a summary table of key transitions
    """
    print(f"Generating summary table for {from_year}-{to_year}...")

    n_classes = len(class_names)

    # Calculate total area for each class in the from_year
    total_pixels_by_class = transition_matrix.sum(axis=1)
    total_area_by_class = total_pixels_by_class * pixel_area_km2

    # Create a list to store summary data
    summary_data = []

    # Focus on key transitions
    key_transitions = [
        (1, 0, 'Vegetation to Urban'),  # Vegetation to Urban
        (0, 1, 'Urban to Vegetation'),  # Urban to Vegetation
        (2, 0, 'Water to Urban'),       # Water to Urban
        (3, 0, 'Barren to Urban')       # Barren to Urban
    ]

    # Calculate metrics for each key transition
    for from_class, to_class, label in key_transitions:
        # Get pixel count
        pixel_count = transition_matrix[from_class, to_class]

        # Calculate area
        area_km2 = pixel_count * pixel_area_km2

        # Calculate percentage of initial class (safely)
        if total_pixels_by_class[from_class] > 0:
            percent_of_initial = (pixel_count / total_pixels_by_class[from_class]) * 100
        else:
            percent_of_initial = 0

        # Add to summary data
        summary_data.append({
            'Transition': label,
            'From Class': class_names[from_class],
            'To Class': class_names[to_class],
            'Pixels (N)': pixel_count,
            'Area (km²)': area_km2,
            '% of Initial Class': percent_of_initial
        })

    # Create a DataFrame
    summary_df = pd.DataFrame(summary_data)

    # Calculate total urban gain
    urban_gain_pixels = sum(transition_matrix[i, 0] for i in range(1, n_classes))
    urban_gain_area = urban_gain_pixels * pixel_area_km2

    # Add summary rows
    summary_df = pd.concat([
        summary_df,
        pd.DataFrame([{
            'Transition': 'Net Urban Gain',
            'From Class': 'Multiple',
            'To Class': 'Urban',
            'Pixels (N)': urban_gain_pixels,
            'Area (km²)': urban_gain_area,
            '% of Initial Class': 'N/A'
        }])
    ], ignore_index=True)

    print("\nSummary Table:")
    print(summary_df)

    return summary_df

# Main function to execute change detection analysis
def execute_change_detection():
    """
    Execute the Change Detection Analysis with transition matrix heatmaps
    """
    print("\n" + "="*80)
    print("EXECUTING STEP 5: CHANGE DETECTION ANALYSIS WITH HEATMAPS")
    print("="*80 + "\n")

    # Create output directory if it doesn't exist
    output_dir = 'change_detection'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define classified image paths
    classified_paths = {
        2015: os.path.join('15', 'RFclassified_2015.tif'),
        2020: os.path.join('20', 'RFclassified_2020.tif'),
        2023: os.path.join('23', 'RFclassified_2023.tif')
    }

    # Verify that all files exist
    missing_files = []
    for year, path in classified_paths.items():
        if not os.path.exists(path):
            missing_files.append(f"{year}: {path}")

    if missing_files:
        print("WARNING: The following classified images were not found:")
        for missing in missing_files:
            print(f"  - {missing}")
        if len(missing_files) >= 2:
            print("Not enough classified images available for comparison.")
            return None

    # Define year pairs for comparison
    year_pairs = [
        (2015, 2020),
        (2020, 2023),
        (2015, 2023)
    ]

    # Process each year pair
    results = {}
    for from_year, to_year in year_pairs:
        # Skip if any year is missing
        if from_year not in classified_paths or to_year not in classified_paths:
            print(f"Skipping {from_year}-{to_year}: Missing classified images")
            continue

        if not os.path.exists(classified_paths[from_year]) or not os.path.exists(classified_paths[to_year]):
            print(f"Skipping {from_year}-{to_year}: File not found")
            continue

        print(f"\n{'='*50}")
        print(f"Change Detection Analysis: {from_year} to {to_year}")
        print(f"{'='*50}")

        # Get paths to classified images
        from_year_path = classified_paths[from_year]
        to_year_path = classified_paths[to_year]

        try:
            # Task 1: Create transition matrix
            print("\nTask 1: Creating transition matrix...")
            transition_matrix, transition_area_matrix, pixel_area_km2 = create_transition_matrix(
                from_year_path, to_year_path, from_year, to_year
            )

            # Skip to next pair if transition matrix creation failed
            if transition_matrix is None:
                print(f"Skipping {from_year}-{to_year}: Failed to create transition matrix")
                continue

            # Task 2: Visualize transition matrix as heatmap
            print("\nTask 2: Visualizing transition matrix as heatmap...")
            heatmap_path = visualize_transition_heatmap(
                transition_matrix, transition_area_matrix, from_year, to_year, output_dir
            )

            # Task 3: Generate summary table
            print("\nTask 3: Generating summary table...")
            summary_table = generate_summary_table(
                transition_matrix, transition_area_matrix, pixel_area_km2,
                from_year, to_year, class_names
            )

            # Save transition matrix and summary table to CSV
            transition_matrix_path = os.path.join(output_dir, f"transition_matrix_{from_year}_{to_year}.csv")
            pd.DataFrame(
                transition_matrix,
                index=[f"{class_names[i]}" for i in range(len(class_names))],
                columns=[f"{class_names[i]}" for i in range(len(class_names))]
            ).to_csv(transition_matrix_path)

            # Save area transition matrix to CSV
            transition_area_matrix_path = os.path.join(output_dir, f"transition_area_matrix_{from_year}_{to_year}.csv")
            pd.DataFrame(
                transition_area_matrix,
                index=[f"{class_names[i]}" for i in range(len(class_names))],
                columns=[f"{class_names[i]}" for i in range(len(class_names))]
            ).to_csv(transition_area_matrix_path)

            summary_table_path = os.path.join(output_dir, f"summary_table_{from_year}_{to_year}.csv")
            summary_table.to_csv(summary_table_path, index=False)

            # Store results
            results[(from_year, to_year)] = {
                'transition_matrix': transition_matrix,
                'transition_area_matrix': transition_area_matrix,
                'summary_table': summary_table,
                'pixel_area_km2': pixel_area_km2,
                'heatmap_path': heatmap_path
            }

            print(f"\nChange detection completed for {from_year}-{to_year}")

        except Exception as e:
            print(f"Error processing {from_year}-{to_year}: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\nChange detection analysis with heatmaps completed.")
    return results

# Run the change detection analysis
if __name__ == "__main__":
    execute_change_detection()


# In[18]:


# Change Detection Analysis for Gurgaon Urban Expansion Study
# Compatible with Stratified CV Models from Phase 3 - WITH LAT/LON COORDINATES

# Required imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from rasterio.warp import reproject, Resampling
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# LULC class definitions - matching Phase 3
class_names = {
    0: 'Urban',
    1: 'Vegetation',
    2: 'Water',
    3: 'Barren'
}

# Define satellite resolutions by year - matching Phase 3
satellite_info = {
    2015: {'satellite': 'Landsat', 'resolution_m': 30.0},
    2020: {'satellite': 'Sentinel', 'resolution_m': 10.0},
    2023: {'satellite': 'Sentinel', 'resolution_m': 10.0}
}

def get_lat_lon_extent(raster_path):
    """
    Get the latitude and longitude extent from a raster file

    Parameters:
    raster_path (str): Path to the raster file

    Returns:
    tuple: (min_lon, max_lon, min_lat, max_lat)
    """
    with rasterio.open(raster_path) as src:
        # Get the bounds in the CRS of the raster
        bounds = src.bounds

        # Extract coordinates
        min_lon, min_lat, max_lon, max_lat = bounds.left, bounds.bottom, bounds.right, bounds.top

        return min_lon, max_lon, min_lat, max_lat

def calculate_aspect_ratio(min_lon, max_lon, min_lat, max_lat):
    """
    Calculate proper aspect ratio for geographic coordinates
    """
    lat_range = max_lat - min_lat
    lon_range = max_lon - min_lon

    # Adjust for latitude (approximately cos(latitude) correction)
    lat_center = (max_lat + min_lat) / 2
    aspect_ratio = lon_range / lat_range * np.cos(np.radians(lat_center))

    return aspect_ratio

def get_pixel_area_km2(year):
    """
    Calculate pixel area in square kilometers based on satellite resolution
    Uses the same function name as Phase 3 for consistency
    """
    if year in satellite_info:
        resolution_m = satellite_info[year]['resolution_m']
        area_km2 = (resolution_m * resolution_m) / 1000000.0
        return area_km2
    else:
        raise ValueError(f"Satellite information not available for year {year}")

def resample_raster(src_path, target_path, output_path=None):
    """
    Resample a raster to match the dimensions of a target raster
    """
    print(f"Resampling {os.path.basename(src_path)} to match {os.path.basename(target_path)}...")

    # If output path is not provided, create one
    if output_path is None:
        src_dir = os.path.dirname(src_path)
        src_name = os.path.basename(src_path)
        output_path = os.path.join(src_dir, f"resampled_{src_name}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with rasterio.open(src_path) as src:
        src_data = src.read(1)
        src_meta = src.meta.copy()

        with rasterio.open(target_path) as target:
            # Get target dimensions
            target_height = target.height
            target_width = target.width
            target_transform = target.transform
            target_crs = target.crs

            # Update metadata for resampled output
            dst_meta = src_meta.copy()
            dst_meta.update({
                'height': target_height,
                'width': target_width,
                'transform': target_transform,
                'crs': target_crs
            })

            # Create destination array
            dst_data = np.ones((target_height, target_width), dtype=src_data.dtype) * 255

            # Perform reprojection/resampling
            try:
                reproject(
                    source=src.read(1),
                    destination=dst_data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest
                )

                # Write resampled raster to disk
                with rasterio.open(output_path, 'w', **dst_meta) as dst:
                    dst.write(dst_data, 1)

                print(f"Resampled raster saved to: {output_path}")
                return output_path

            except Exception as e:
                print(f"Error during resampling: {str(e)}")
                return None

def create_transition_matrix(from_year_path, to_year_path, from_year, to_year, resample=True):
    """
    Create a transition matrix comparing land cover changes between two years
    """
    print(f"Creating transition matrix from {from_year} to {to_year}...")

    # Display satellite information for both years
    if from_year in satellite_info:
        from_sat = satellite_info[from_year]
        print(f"From: {from_sat['satellite']} ({from_year}), Resolution: {from_sat['resolution_m']}m")
    if to_year in satellite_info:
        to_sat = satellite_info[to_year]
        print(f"To: {to_sat['satellite']} ({to_year}), Resolution: {to_sat['resolution_m']}m")

    # Check if rasters have the same dimensions
    with rasterio.open(from_year_path) as src_from, rasterio.open(to_year_path) as src_to:
        from_shape = (src_from.height, src_from.width)
        to_shape = (src_to.height, src_to.width)

        print(f"From shape: {from_shape}, To shape: {to_shape}")

        # If shapes don't match and resample=True, resample the from_year raster to match to_year
        if from_shape != to_shape and resample:
            print("Rasters have different dimensions. Resampling...")

            # Create a directory for resampled rasters if it doesn't exist
            resample_dir = "resampled_rasters"
            if not os.path.exists(resample_dir):
                os.makedirs(resample_dir)

            # Define output path for resampled raster
            resampled_path = os.path.join(resample_dir, f"resampled_{os.path.basename(from_year_path)}")

            # Resample from_year raster to match to_year dimensions
            resampled_from_path = resample_raster(from_year_path, to_year_path, resampled_path)

            if resampled_from_path is None:
                print("Resampling failed. Cannot create transition matrix.")
                return None, None, None

            # Update the from_year path to use the resampled raster
            from_year_path = resampled_from_path

    # Calculate the pixel area for the "to_year" raster - using function from Phase 3
    pixel_area_km2 = get_pixel_area_km2(to_year)
    print(f"Pixel area: {pixel_area_km2:.6f} km²")

    # Now process with potentially resampled data
    with rasterio.open(from_year_path) as src_from, rasterio.open(to_year_path) as src_to:
        # Read the data
        from_data = src_from.read(1)
        to_data = src_to.read(1)

        # Create masks for valid data (not nodata)
        valid_mask_from = (from_data != 255)
        valid_mask_to = (to_data != 255)

        # Combined valid mask (pixels valid in both rasters)
        valid_mask = valid_mask_from & valid_mask_to

        # Count valid pixels
        valid_count = np.sum(valid_mask)
        print(f"Valid pixels for comparison: {valid_count} out of {from_data.size} ({(valid_count/from_data.size)*100:.2f}%)")

        # Create an empty transition matrix (from class to class)
        n_classes = len(class_names)
        transition_matrix = np.zeros((n_classes, n_classes), dtype=np.int32)

        # Fill the transition matrix
        for from_class in range(n_classes):
            for to_class in range(n_classes):
                # Count pixels that transitioned from from_class to to_class
                transition_count = np.sum((from_data == from_class) & (to_data == to_class) & valid_mask)
                transition_matrix[from_class, to_class] = transition_count

        # Calculate areas using the calculated pixel area
        transition_area_matrix = transition_matrix * pixel_area_km2

    # Create DataFrames for better visualization
    transition_df = pd.DataFrame(
        transition_matrix,
        index=[f"{class_names[i]}" for i in range(n_classes)],
        columns=[f"{class_names[i]}" for i in range(n_classes)]
    )

    print("\nTransition Matrix (pixel counts):")
    print(transition_df)

    transition_area_df = pd.DataFrame(
        transition_area_matrix,
        index=[f"{class_names[i]}" for i in range(n_classes)],
        columns=[f"{class_names[i]}" for i in range(n_classes)]
    )

    print("\nTransition Matrix (area in km²):")
    print(transition_area_df)

    return transition_matrix, transition_area_matrix, pixel_area_km2

def visualize_transition_heatmap(transition_matrix, transition_area_matrix, from_year, to_year, output_dir):
    """
    Create and save heatmap visualizations of transition matrices
    """
    # Display satellite information
    if from_year in satellite_info and to_year in satellite_info:
        from_sat = satellite_info[from_year]['satellite']
        to_sat = satellite_info[to_year]['satellite']
        print(f"Creating transition heatmap: {from_sat} ({from_year}) → {to_sat} ({to_year})")

    n_classes = len(class_names)

    # Create DataFrames with proper labels
    transition_df = pd.DataFrame(
        transition_matrix,
        index=[f"{class_names[i]}" for i in range(n_classes)],
        columns=[f"{class_names[i]}" for i in range(n_classes)]
    )

    transition_area_df = pd.DataFrame(
        transition_area_matrix,
        index=[f"{class_names[i]}" for i in range(n_classes)],
        columns=[f"{class_names[i]}" for i in range(n_classes)]
    )

    # Calculate percentage matrix for visualization
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    percent_matrix = (transition_matrix / row_sums) * 100

    percent_df = pd.DataFrame(
        percent_matrix,
        index=[f"{class_names[i]}" for i in range(n_classes)],
        columns=[f"{class_names[i]}" for i in range(n_classes)]
    )

    # Create figure with 3 subplots for different visualizations
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # 1. Pixel count heatmap
    sns.heatmap(
        transition_df,
        annot=True,
        fmt=",d",
        cmap="YlGnBu",
        ax=axes[0],
        cbar_kws={'label': 'Pixel Count'}
    )
    axes[0].set_title(f'Transition Matrix (Pixel Count): {from_year} to {to_year}')
    axes[0].set_xlabel('To Class (End State)')
    axes[0].set_ylabel('From Class (Start State)')

    # 2. Area (km²) heatmap
    sns.heatmap(
        transition_area_df,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        ax=axes[1],
        cbar_kws={'label': 'Area (km²)'}
    )
    axes[1].set_title(f'Transition Matrix (Area in km²): {from_year} to {to_year}')
    axes[1].set_xlabel('To Class (End State)')
    axes[1].set_ylabel('From Class (Start State)')

    # 3. Percentage heatmap
    sns.heatmap(
        percent_df,
        annot=True,
        fmt=".1f",
        cmap="viridis",
        ax=axes[2],
        cbar_kws={'label': 'Percentage of Start Class (%)'}
    )
    axes[2].set_title(f'Transition Matrix (Percentage): {from_year} to {to_year}')
    axes[2].set_xlabel('To Class (End State)')
    axes[2].set_ylabel('From Class (Start State)')

    plt.tight_layout()

    # Save figure
    heatmap_path = os.path.join(output_dir, f"transition_heatmap_{from_year}_{to_year}.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight', format='png')
    print(f"Transition heatmap saved to: {heatmap_path}")
    plt.close()

    return heatmap_path

def create_change_maps(from_year_path, to_year_path, from_year, to_year, output_dir):
    """
    Create change maps showing spatial distribution of land cover transitions
    WITH LATITUDE/LONGITUDE COORDINATES
    """
    print(f"\nCreating change map from {from_year} to {to_year}...")

    # Display satellite information like in Phase 3
    if from_year in satellite_info:
        from_sat = satellite_info[from_year]
        print(f"From: {from_sat['satellite']} data (Resolution: {from_sat['resolution_m']}m)")
    if to_year in satellite_info:
        to_sat = satellite_info[to_year]
        print(f"To: {to_sat['satellite']} data (Resolution: {to_sat['resolution_m']}m)")

    # Check if resampling is needed
    with rasterio.open(from_year_path) as src_from, rasterio.open(to_year_path) as src_to:
        from_shape = (src_from.height, src_from.width)
        to_shape = (src_to.height, src_to.width)

        if from_shape != to_shape:
            print("Rasters have different dimensions. Using resampled version...")
            resample_dir = "resampled_rasters"
            resampled_from_path = os.path.join(resample_dir, f"resampled_{os.path.basename(from_year_path)}")

            if os.path.exists(resampled_from_path):
                from_year_path = resampled_from_path
            else:
                print("Warning: No resampled raster found. Change map may not be accurate.")

    # Get coordinates for proper display
    min_lon, max_lon, min_lat, max_lat = get_lat_lon_extent(to_year_path)
    aspect_ratio = calculate_aspect_ratio(min_lon, max_lon, min_lat, max_lat)

    # Calculate figure size for proper aspect ratio
    base_height = 12
    fig_width = base_height * aspect_ratio

    # Read the classified rasters
    with rasterio.open(from_year_path) as src_from, rasterio.open(to_year_path) as src_to:
        from_data = src_from.read(1)
        to_data = src_to.read(1)
        meta = src_to.meta.copy()

    # Create the change map
    change_map = np.full_like(from_data, 255, dtype=np.uint8)

    # Create masks for valid data
    valid_mask = (from_data != 255) & (to_data != 255)

    # Calculate pixel area for this year
    pixel_area_km2 = get_pixel_area_km2(to_year)
    print(f"Pixel area: {pixel_area_km2:.6f} km²")

    # Define transition codes
    transition_codes = {
        (0, 0): 0,  # Urban to Urban (no change)
        (0, 1): 1,  # Urban to Vegetation
        (0, 2): 2,  # Urban to Water
        (0, 3): 3,  # Urban to Barren
        (1, 0): 4,  # Vegetation to Urban
        (1, 1): 5,  # Vegetation to Vegetation (no change)
        (1, 2): 6,  # Vegetation to Water
        (1, 3): 7,  # Vegetation to Barren
        (2, 0): 8,  # Water to Urban
        (2, 1): 9,  # Water to Vegetation
        (2, 2): 10, # Water to Water (no change)
        (2, 3): 11, # Water to Barren
        (3, 0): 12, # Barren to Urban
        (3, 1): 13, # Barren to Vegetation
        (3, 2): 14, # Barren to Water
        (3, 3): 15  # Barren to Barren (no change)
    }

    # Fill the change map
    for (from_class, to_class), code in transition_codes.items():
        mask = (from_data == from_class) & (to_data == to_class) & valid_mask
        change_map[mask] = code

    # Save the change map
    change_map_path = os.path.join(output_dir, f"change_map_{from_year}_{to_year}.tif")
    meta.update(dtype='uint8', nodata=255)
    with rasterio.open(change_map_path, 'w', **meta) as dst:
        dst.write(change_map, 1)
    print(f"Change map raster saved to: {change_map_path}")

    # Create visualization (simplified map focusing on key transitions)
    plt.figure(figsize=(fig_width, base_height))

    # Create a simplified change map
    simplified_change_map = np.full_like(change_map, 255, dtype=np.uint8)

    # Key transitions to highlight (as specified in requirements)
    key_transitions = {
        4: 0,   # Vegetation to Urban (red)
        1: 1,   # Urban to Vegetation (green)
        8: 2,   # Water to Urban (blue)
        12: 3,  # Barren to Urban (yellow)
        0: 4,   # No change Urban (gray)
        5: 4,   # No change Vegetation (gray)
        10: 4,  # No change Water (gray)
        15: 4   # No change Barren (gray)
    }

    # Map to simplified categories
    for original, simplified in key_transitions.items():
        mask = change_map == original
        simplified_change_map[mask] = simplified

    # Define colors as specified in requirements
    colors = ['#FF0000',   # Vegetation to Urban (red)
              '#00FF00',   # Urban to Vegetation (green)
              '#0000FF',   # Water to Urban (blue)
              '#FFFF00',   # Barren to Urban (yellow)
              '#808080']   # No change (gray)

    cmap = ListedColormap(colors)

    # Create masked array to hide no-data values
    masked_simplified = np.ma.masked_where(simplified_change_map == 255, simplified_change_map)

    # Plot the change map with proper coordinates and aspect ratio
    plt.imshow(masked_simplified, cmap=cmap, vmin=0, vmax=4,
               extent=[min_lon, max_lon, min_lat, max_lat], aspect='equal')

    # Create legend with specified transitions
    legend_elements = [
        Patch(facecolor='#FF0000', label='Vegetation to Urban'),
        Patch(facecolor='#00FF00', label='Urban to Vegetation'),
        Patch(facecolor='#0000FF', label='Water to Urban'),
        Patch(facecolor='#FFFF00', label='Barren to Urban'),
        Patch(facecolor='#808080', label='No Change')
    ]

    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

    # Add title with satellite information
    if from_year in satellite_info and to_year in satellite_info:
        title = f'Land Cover Change Map: {from_year} ({satellite_info[from_year]["satellite"]}) to {to_year} ({satellite_info[to_year]["satellite"]})'
    else:
        title = f'Land Cover Change Map: {from_year} to {to_year}'
    plt.title(title, fontsize=16, fontweight='bold')

    # Add coordinate labels
    plt.xlabel('Longitude (°)', fontsize=14)
    plt.ylabel('Latitude (°)', fontsize=14)

    # Add coordinate information text
    coord_text = f'Extent: {min_lon:.3f}° to {max_lon:.3f}°E, {min_lat:.3f}° to {max_lat:.3f}°N'
    plt.annotate(coord_text, xy=(0.02, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                 fontsize=10)

    plt.tight_layout()

    # Save the visualization
    vis_path = os.path.join(output_dir, f"change_map_visual_{from_year}_{to_year}.png")
    plt.savefig(vis_path, dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print(f"Change map visualization saved to: {vis_path}")

    # Calculate and print statistics
    print("\nChange Map Statistics:")
    total_valid = np.sum(valid_mask)
    print(f"Geographic extent: {min_lon:.4f}° to {max_lon:.4f}°E, {min_lat:.4f}° to {max_lat:.4f}°N")

    # Calculate area for each key transition
    transition_areas = {}
    transition_labels = {
        4: 'Vegetation to Urban',
        1: 'Urban to Vegetation',
        8: 'Water to Urban',
        12: 'Barren to Urban'
    }

    for code, label in transition_labels.items():
        pixel_count = np.sum(change_map == code)
        area_km2 = pixel_count * pixel_area_km2
        percentage = (pixel_count / total_valid) * 100 if total_valid > 0 else 0
        transition_areas[label] = {
            'pixels': pixel_count,
            'area_km2': area_km2,
            'percentage': percentage
        }
        print(f"{label}: {pixel_count} pixels ({area_km2:.2f} km², {percentage:.2f}%)")

    return {
        'change_map_path': change_map_path,
        'visualization_path': vis_path,
        'transition_areas': transition_areas,
        'meta': meta,
        'coordinates': {
            'min_lon': min_lon,
            'max_lon': max_lon,
            'min_lat': min_lat,
            'max_lat': max_lat
        }
    }

def generate_summary_table(transition_matrix, transition_area_matrix, pixel_area_km2, from_year, to_year):
    """
    Generate a summary table of key transitions
    Following Phase 3 style for consistency
    """
    print(f"\nGenerating summary table for {from_year}-{to_year}...")

    n_classes = len(class_names)

    # Calculate total area for each class in the from_year
    total_pixels_by_class = transition_matrix.sum(axis=1)

    # Create a list to store summary data
    summary_data = []

    # Focus on key transitions as specified
    key_transitions = [
        (1, 0, 'Vegetation to Urban'),
        (0, 1, 'Urban to Vegetation'),
        (2, 0, 'Water to Urban'),
        (3, 0, 'Barren to Urban')
    ]

    # Calculate metrics for each key transition
    for from_class, to_class, label in key_transitions:
        pixel_count = transition_matrix[from_class, to_class]
        area_km2 = pixel_count * pixel_area_km2

        # Calculate percentage of initial class
        if total_pixels_by_class[from_class] > 0:
            percent_of_initial = (pixel_count / total_pixels_by_class[from_class]) * 100
        else:
            percent_of_initial = 0

        summary_data.append({
            'Transition': label,
            'From Class': class_names[from_class],
            'To Class': class_names[to_class],
            'Pixels (N)': pixel_count,
            'Area (km²)': area_km2,
            '% of Initial Class': percent_of_initial
        })

    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)

    # Calculate total urban gain
    urban_gain_pixels = sum(transition_matrix[i, 0] for i in range(1, n_classes))
    urban_gain_area = urban_gain_pixels * pixel_area_km2

    # Add summary row
    summary_df = pd.concat([
        summary_df,
        pd.DataFrame([{
            'Transition': 'Net Urban Gain',
            'From Class': 'Multiple',
            'To Class': 'Urban',
            'Pixels (N)': urban_gain_pixels,
            'Area (km²)': urban_gain_area,
            '% of Initial Class': 'N/A'
        }])
    ], ignore_index=True)

    print("\nSummary Table:")
    print(summary_df)

    return summary_df

# Main function matching Phase 3 style
def execute_change_detection_stratified(year_pairs=None):
    """
    Execute the Change Detection Analysis with Stratified CV classified images
    Following the pattern of execute_phase3_stratified
    WITH FULL LATITUDE/LONGITUDE COORDINATE SUPPORT
    """
    print("\n" + "="*80)
    print("EXECUTING CHANGE DETECTION ANALYSIS WITH LAT/LON COORDINATES")
    print("Using Stratified CV Classified Images from Phase 3")
    print("="*80 + "\n")

    # Create output directory
    output_dir = 'change_detection'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define paths to stratified CV classified images (matching Phase 3 output)
    classified_paths = {
        2015: os.path.join('15', 'RFclassified_2015.tif'),
        2020: os.path.join('20', 'RFclassified_2020.tif'),
        2023: os.path.join('23', 'RFclassified_2023.tif')
    }

    # Verify files exist
    missing_files = []
    for year, path in classified_paths.items():
        if not os.path.exists(path):
            missing_files.append(f"{year}: {path}")

    if missing_files:
        print("WARNING: The following classified images were not found:")
        for missing in missing_files:
            print(f"  - {missing}")
        print("\nPlease run Phase 3 with Stratified CV models first.")
        return None

    # Default year pairs if not specified
    if year_pairs is None:
        year_pairs = [
            (2015, 2020),
            (2020, 2023),
            (2015, 2023)
        ]

    # Process each year pair
    results = {}
    for from_year, to_year in year_pairs:
        print(f"\n{'='*50}")
        print(f"Change Detection Analysis: {from_year} to {to_year}")
        if from_year in satellite_info and to_year in satellite_info:
            print(f"{satellite_info[from_year]['satellite']} → {satellite_info[to_year]['satellite']}")
        print(f"{'='*50}")

        # Get paths
        from_year_path = classified_paths[from_year]
        to_year_path = classified_paths[to_year]

        try:
            # Task 1: Create transition matrix
            print("\nTask 1: Creating transition matrix...")
            transition_matrix, transition_area_matrix, pixel_area_km2 = create_transition_matrix(
                from_year_path, to_year_path, from_year, to_year
            )

            if transition_matrix is None:
                print(f"Skipping {from_year}-{to_year}: Failed to create transition matrix")
                continue

            # Task 2: Visualize transition matrix
            print("\nTask 2: Visualizing transition matrix...")
            heatmap_path = visualize_transition_heatmap(
                transition_matrix, transition_area_matrix, from_year, to_year, output_dir
            )

            # Task 3: Generate summary table
            print("\nTask 3: Generating summary table...")
            summary_table = generate_summary_table(
                transition_matrix, transition_area_matrix, pixel_area_km2,
                from_year, to_year
            )

            # Task 4: Create change maps with coordinates
            print("\nTask 4: Creating change maps with lat/lon coordinates...")
            change_map_results = create_change_maps(
                from_year_path, to_year_path, from_year, to_year, output_dir
            )

            # Save results as CSV
            transition_matrix_path = os.path.join(output_dir, f"transition_matrix_{from_year}_{to_year}.csv")
            pd.DataFrame(
                transition_matrix,
                index=[class_names[i] for i in range(len(class_names))],
                columns=[class_names[i] for i in range(len(class_names))]
            ).to_csv(transition_matrix_path)

            summary_table_path = os.path.join(output_dir, f"summary_table_{from_year}_{to_year}.csv")
            summary_table.to_csv(summary_table_path, index=False)

            # Store results
            results[(from_year, to_year)] = {
                'transition_matrix': transition_matrix,
                'transition_area_matrix': transition_area_matrix,
                'summary_table': summary_table,
                'pixel_area_km2': pixel_area_km2,
                'heatmap_path': heatmap_path,
                'change_map_results': change_map_results
            }

            print(f"\nChange detection completed successfully for {from_year}-{to_year}")
            if 'coordinates' in change_map_results:
                coords = change_map_results['coordinates']
                print(f"Geographic extent: {coords['min_lon']:.4f}° to {coords['max_lon']:.4f}°E, {coords['min_lat']:.4f}° to {coords['max_lat']:.4f}°N")

        except Exception as e:
            print(f"Error processing {from_year}-{to_year}: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("CHANGE DETECTION ANALYSIS WITH COORDINATES COMPLETED")
    print("Results saved in:", output_dir)
    print("All change maps now include proper latitude/longitude coordinates!")
    print("="*80)

    return results

def create_comparative_change_visualization(results, output_dir):
    """
    Create comparative visualization showing urbanization trends across time periods
    """
    print("Creating comparative change visualization...")

    # Extract urbanization data
    periods = []
    urban_gains = []

    for (from_year, to_year), result in results.items():
        periods.append(f"{from_year}-{to_year}")

        # Get vegetation to urban transition
        transition_matrix = result['transition_matrix']
        pixel_area_km2 = result['pixel_area_km2']

        # Vegetation to Urban (class 1 to class 0)
        veg_to_urban_pixels = transition_matrix[1, 0]
        veg_to_urban_area = veg_to_urban_pixels * pixel_area_km2
        urban_gains.append(veg_to_urban_area)

    # Create bar chart
    plt.figure(figsize=(12, 8))
    bars = plt.bar(periods, urban_gains, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])

    # Add value labels on bars
    for bar, value in zip(bars, urban_gains):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(urban_gains)*0.01,
                f'{value:.2f} km²', ha='center', va='bottom', fontweight='bold')

    plt.title('Urbanization Trends in Gurgaon (Vegetation to Urban Conversion)',
              fontsize=16, fontweight='bold')
    plt.xlabel('Time Period', fontsize=14)
    plt.ylabel('Area Converted to Urban (km²)', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add total urbanization text
    total_urban_gain = sum(urban_gains)
    plt.text(0.02, 0.95, f'Total Urban Gain: {total_urban_gain:.2f} km²',
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.8),
             fontsize=12, fontweight='bold')

    plt.tight_layout()

    # Save comparison chart
    comparison_path = os.path.join(output_dir, 'urbanization_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print(f"Comparative visualization saved to: {comparison_path}")

    return comparison_path

def generate_comprehensive_report(results, output_dir):
    """
    Generate a comprehensive report of all change detection results
    """
    print("Generating comprehensive change detection report...")

    report_data = []

    for (from_year, to_year), result in results.items():
        transition_matrix = result['transition_matrix']
        pixel_area_km2 = result['pixel_area_km2']

        # Calculate key metrics
        total_pixels = np.sum(transition_matrix)

        # Key transitions
        veg_to_urban = transition_matrix[1, 0] * pixel_area_km2
        urban_to_veg = transition_matrix[0, 1] * pixel_area_km2
        water_to_urban = transition_matrix[2, 0] * pixel_area_km2
        barren_to_urban = transition_matrix[3, 0] * pixel_area_km2

        # Net urban gain
        net_urban_gain = sum(transition_matrix[i, 0] for i in range(1, len(class_names))) * pixel_area_km2

        # Urban loss
        urban_loss = sum(transition_matrix[0, i] for i in range(1, len(class_names))) * pixel_area_km2

        report_data.append({
            'Period': f"{from_year}-{to_year}",
            'From_Year': from_year,
            'To_Year': to_year,
            'Satellite_From': satellite_info[from_year]['satellite'],
            'Satellite_To': satellite_info[to_year]['satellite'],
            'Resolution_From_m': satellite_info[from_year]['resolution_m'],
            'Resolution_To_m': satellite_info[to_year]['resolution_m'],
            'Pixel_Area_km2': pixel_area_km2,
            'Vegetation_to_Urban_km2': veg_to_urban,
            'Urban_to_Vegetation_km2': urban_to_veg,
            'Water_to_Urban_km2': water_to_urban,
            'Barren_to_Urban_km2': barren_to_urban,
            'Net_Urban_Gain_km2': net_urban_gain,
            'Urban_Loss_km2': urban_loss,
            'Net_Urban_Change_km2': net_urban_gain - urban_loss
        })

    # Create comprehensive DataFrame
    report_df = pd.DataFrame(report_data)

    # Save comprehensive report
    report_path = os.path.join(output_dir, 'comprehensive_change_report.csv')
    report_df.to_csv(report_path, index=False)

    print("\nComprehensive Change Detection Report:")
    print("="*80)
    print(report_df.to_string(index=False))
    print(f"\nReport saved to: {report_path}")

    return report_df

# Main execution
if __name__ == "__main__":
    # Execute change detection with default year pairs
    print("Starting Change Detection Analysis with Latitude/Longitude Coordinates...")
    results = execute_change_detection_stratified()

    if results:
        # Create comparative visualization
        comparison_path = create_comparative_change_visualization(results, 'change_detection')

        # Generate comprehensive report
        report_df = generate_comprehensive_report(results, 'change_detection')

        print("\n" + "="*80)
        print("CHANGE DETECTION ANALYSIS COMPLETE!")
        print("="*80)
        print("✅ All change maps include latitude/longitude coordinates")
        print("✅ Proper aspect ratios prevent map elongation")
        print("✅ Geographic extent information displayed")
        print("✅ High-quality PNG outputs for all visualizations")
        print("✅ Comprehensive CSV reports generated")
        print("✅ Comparative analysis across time periods")
        print("="*80)

    else:
        print("No results generated. Please check if classified images exist.")

    # Alternative usage examples:
    # For specific year pairs only:
    # results = execute_change_detection_stratified([(2015, 2020), (2020, 2023)])

    # For single year pair:
    # results = execute_change_detection_stratified([(2015, 2023)])


# In[12]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# # Set working directory - update this to your project folder path
# base_dir = r"C:\Users\divya\Desktop\urbanization in gurgaon"
# os.chdir(base_dir)

# LULC class definitions
class_names = {
    0: 'Urban',
    1: 'Vegetation',
    2: 'Water',
    3: 'Barren'
}

def get_lat_lon_extent(raster_path):
    """
    Get the latitude and longitude extent from a raster file

    Parameters:
    raster_path (str): Path to the raster file

    Returns:
    tuple: (min_lon, max_lon, min_lat, max_lat)
    """
    with rasterio.open(raster_path) as src:
        # Get the bounds in the CRS of the raster
        bounds = src.bounds

        # Extract coordinates
        min_lon, min_lat, max_lon, max_lat = bounds.left, bounds.bottom, bounds.right, bounds.top

        return min_lon, max_lon, min_lat, max_lat

def calculate_aspect_ratio(min_lon, max_lon, min_lat, max_lat):
    """
    Calculate proper aspect ratio for geographic coordinates
    """
    lat_range = max_lat - min_lat
    lon_range = max_lon - min_lon

    # Adjust for latitude (approximately cos(latitude) correction)
    lat_center = (max_lat + min_lat) / 2
    aspect_ratio = lon_range / lat_range * np.cos(np.radians(lat_center))

    return aspect_ratio

def calculate_tif_values(year_folder, year_full=None):
    """
    Calculate T, I, and F values directly from NDBI and NDVI

    Parameters:
    year_folder (str): Folder name (e.g., '15')
    year_full (int, optional): Full year (e.g., 2015)

    Returns:
    dict: Dictionary with T, I, F arrays and their paths
    """
    # Define folder based on year
    folder = str(year_folder)

    # Define the full year for file naming
    if year_full is None:
        if year_folder == '15':
            year_full = 2015
        elif year_folder == '20':
            year_full = 2020
        elif year_folder == '23':
            year_full = 2023

    print(f"Calculating T, I, F values for year {year_full}...")

    # Define paths to NDBI and NDVI rasters
    ndbi_path = os.path.join(folder, f"Gurgaon_{year_full}_NDBI.tif")
    ndvi_path = os.path.join(folder, f"Gurgaon_{year_full}_NDVI.tif")

    # Check if files exist
    if not os.path.exists(ndbi_path):
        raise FileNotFoundError(f"NDBI file not found: {ndbi_path}")
    if not os.path.exists(ndvi_path):
        raise FileNotFoundError(f"NDVI file not found: {ndvi_path}")

    # Load NDBI and NDVI rasters
    with rasterio.open(ndbi_path) as src_ndbi, rasterio.open(ndvi_path) as src_ndvi:
        # Read data
        ndbi = src_ndbi.read(1)
        ndvi = src_ndvi.read(1)

        # Get metadata for output
        out_meta = src_ndbi.meta.copy()

        # Create masks for valid data
        valid_mask_ndbi = ~np.isnan(ndbi)
        valid_mask_ndvi = ~np.isnan(ndvi)

        # Combined valid mask
        valid_mask = valid_mask_ndbi & valid_mask_ndvi

        # Calculate min and max values for normalization, using only valid values
        ndbi_valid = ndbi[valid_mask]
        ndvi_valid = ndvi[valid_mask]

        ndbi_min = np.min(ndbi_valid)
        ndbi_max = np.max(ndbi_valid)
        ndvi_min = np.min(ndvi_valid)
        ndvi_max = np.max(ndvi_valid)

        print(f"NDBI range: {ndbi_min:.4f} to {ndbi_max:.4f}")
        print(f"NDVI range: {ndvi_min:.4f} to {ndvi_max:.4f}")

        # Initialize T, I, F arrays with NaN
        T = np.full_like(ndbi, np.nan)
        F = np.full_like(ndbi, np.nan)
        I = np.full_like(ndbi, np.nan)

        # Calculate T(x) - Urban/Built-up Likelihood (from NDBI)
        # Normalize NDBI to [0,1] range
        T[valid_mask] = (ndbi[valid_mask] - ndbi_min) / (ndbi_max - ndbi_min)

        # Calculate F(x) - Green Space Likelihood (from NDVI)
        # Normalize NDVI to [0,1] range
        F[valid_mask] = (ndvi[valid_mask] - ndvi_min) / (ndvi_max - ndvi_min)

        # Calculate I(x) - Indeterminacy/Uncertainty
        # I(x) = 1 - |T(x) - F(x)|
        I[valid_mask] = 1 - np.abs(T[valid_mask] - F[valid_mask])

        # Define output paths
        T_path = os.path.join(folder, f"T_raster_{year_full}.tif")
        I_path = os.path.join(folder, f"I_raster_{year_full}.tif")
        F_path = os.path.join(folder, f"F_raster_{year_full}.tif")

        # Save T, I, F rasters
        for path, data in [(T_path, T), (I_path, I), (F_path, F)]:
            with rasterio.open(path, 'w', **out_meta) as dst:
                dst.write(data, 1)

        print(f"T, I, F rasters saved to: {folder}")

        return {
            'T_path': T_path,
            'I_path': I_path,
            'F_path': F_path,
            'T': T,
            'I': I,
            'F': F,
            'meta': out_meta,
            'valid_mask': valid_mask
        }

def visualize_tif_histograms(tif_data, year_full=None, bins=50):
    """
    Visualize histograms of T, I, F values to help with threshold selection

    Parameters:
    tif_data (dict): Dictionary with T, I, F arrays
    year_full (int, optional): Full year for title
    bins (int): Number of bins for histogram
    """
    # Extract T, I, F arrays and valid mask
    T = tif_data['T']
    I = tif_data['I']
    F = tif_data['F']
    valid_mask = tif_data['valid_mask']

    # Create figure for histograms
    plt.figure(figsize=(18, 6))

    # Plot histogram for T (Urban)
    plt.subplot(1, 3, 1)
    plt.hist(T[valid_mask].flatten(), bins=bins, color='red', alpha=0.7)
    plt.title(f'Urban Likelihood (T) Distribution for {year_full}')
    plt.xlabel('T Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    # Calculate and show statistics
    t_mean = np.mean(T[valid_mask])
    t_median = np.median(T[valid_mask])
    t_std = np.std(T[valid_mask])
    plt.axvline(t_mean, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {t_mean:.3f}')
    plt.axvline(t_median, color='black', linestyle='dashed', linewidth=1.5, label=f'Median: {t_median:.3f}')
    plt.axvline(t_mean + t_std, color='darkred', linestyle='dotted', linewidth=1.5, label=f'Mean+Std: {t_mean + t_std:.3f}')
    plt.axvline(t_mean - t_std, color='darkred', linestyle='dotted', linewidth=1.5, label=f'Mean-Std: {t_mean - t_std:.3f}')
    plt.legend()

    # Plot histogram for I (Indeterminacy)
    plt.subplot(1, 3, 2)
    plt.hist(I[valid_mask].flatten(), bins=bins, color='blue', alpha=0.7)
    plt.title(f'Indeterminacy (I) Distribution for {year_full}')
    plt.xlabel('I Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    # Calculate and show statistics
    i_mean = np.mean(I[valid_mask])
    i_median = np.median(I[valid_mask])
    i_std = np.std(I[valid_mask])
    plt.axvline(i_mean, color='blue', linestyle='dashed', linewidth=1.5, label=f'Mean: {i_mean:.3f}')
    plt.axvline(i_median, color='black', linestyle='dashed', linewidth=1.5, label=f'Median: {i_median:.3f}')
    plt.axvline(i_mean + i_std, color='darkblue', linestyle='dotted', linewidth=1.5, label=f'Mean+Std: {i_mean + i_std:.3f}')
    plt.axvline(i_mean - i_std, color='darkblue', linestyle='dotted', linewidth=1.5, label=f'Mean-Std: {i_mean - i_std:.3f}')
    plt.legend()

    # Plot histogram for F (Green)
    plt.subplot(1, 3, 3)
    plt.hist(F[valid_mask].flatten(), bins=bins, color='green', alpha=0.7)
    plt.title(f'Green Space Likelihood (F) Distribution for {year_full}')
    plt.xlabel('F Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    # Calculate and show statistics
    f_mean = np.mean(F[valid_mask])
    f_median = np.median(F[valid_mask])
    f_std = np.std(F[valid_mask])
    plt.axvline(f_mean, color='green', linestyle='dashed', linewidth=1.5, label=f'Mean: {f_mean:.3f}')
    plt.axvline(f_median, color='black', linestyle='dashed', linewidth=1.5, label=f'Median: {f_median:.3f}')
    plt.axvline(f_mean + f_std, color='darkgreen', linestyle='dotted', linewidth=1.5, label=f'Mean+Std: {f_mean + f_std:.3f}')
    plt.axvline(f_mean - f_std, color='darkgreen', linestyle='dotted', linewidth=1.5, label=f'Mean-Std: {f_mean - f_std:.3f}')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'TIF_histograms_{year_full}.png', dpi=300, bbox_inches='tight', format='png')
    plt.show()

    # Calculate and print additional statistics to help with threshold selection
    print(f"\nStatistics for {year_full}:")
    print(f"T (Urban) - Min: {np.min(T[valid_mask]):.3f}, Max: {np.max(T[valid_mask]):.3f}, Mean: {t_mean:.3f}, Median: {t_median:.3f}, Std: {t_std:.3f}")
    print(f"I (Indeterminacy) - Min: {np.min(I[valid_mask]):.3f}, Max: {np.max(I[valid_mask]):.3f}, Mean: {i_mean:.3f}, Median: {i_median:.3f}, Std: {i_std:.3f}")
    print(f"F (Green) - Min: {np.min(F[valid_mask]):.3f}, Max: {np.max(F[valid_mask]):.3f}, Mean: {f_mean:.3f}, Median: {f_median:.3f}, Std: {f_std:.3f}")

    # Percentiles to help with threshold selection
    percentiles = [10, 25, 50, 75, 90]
    print("\nPercentiles (to help with threshold selection):")

    print("T (Urban):")
    for p in percentiles:
        value = np.percentile(T[valid_mask], p)
        print(f"{p}th percentile: {value:.3f}")

    print("\nI (Indeterminacy):")
    for p in percentiles:
        value = np.percentile(I[valid_mask], p)
        print(f"{p}th percentile: {value:.3f}")

    print("\nF (Green):")
    for p in percentiles:
        value = np.percentile(F[valid_mask], p)
        print(f"{p}th percentile: {value:.3f}")

    # Return statistics for later use
    return {
        'T_stats': {'mean': t_mean, 'median': t_median, 'std': t_std},
        'I_stats': {'mean': i_mean, 'median': i_median, 'std': i_std},
        'F_stats': {'mean': f_mean, 'median': f_median, 'std': f_std}
    }

def create_tif_composite(tif_data, year_folder, year_full=None):
    """
    Create an RGB composite where:
    - Red = T (Urban likelihood)
    - Green = F (Green space likelihood)
    - Blue = I (Indeterminacy)

    Parameters:
    tif_data (dict): Dictionary with T, I, F arrays and paths
    year_folder (str): Folder name
    year_full (int, optional): Full year

    Returns:
    str: Path to the RGB composite
    """
    # Define folder and year
    folder = str(year_folder)
    if year_full is None:
        if year_folder == '15':
            year_full = 2015
        elif year_folder == '20':
            year_full = 2020
        elif year_folder == '23':
            year_full = 2023

    print(f"Creating TIF composite for year {year_full}...")

    # Get coordinates from one of the T-I-F rasters
    min_lon, max_lon, min_lat, max_lat = get_lat_lon_extent(tif_data['T_path'])
    aspect_ratio = calculate_aspect_ratio(min_lon, max_lon, min_lat, max_lat)

    # Calculate figure size
    base_height = 10
    fig_width = base_height * aspect_ratio

    # Extract T, I, F arrays
    T = tif_data['T']
    I = tif_data['I']
    F = tif_data['F']

    # Create mask for valid data
    valid_mask = ~np.isnan(T) & ~np.isnan(I) & ~np.isnan(F)

    # Scale to 0-255 for RGB visualization
    T_scaled = np.zeros_like(T, dtype=np.uint8)
    I_scaled = np.zeros_like(I, dtype=np.uint8)
    F_scaled = np.zeros_like(F, dtype=np.uint8)

    # Apply min-max scaling only to valid pixels
    if np.any(valid_mask):
        T_scaled[valid_mask] = np.round(T[valid_mask] * 255).astype(np.uint8)
        I_scaled[valid_mask] = np.round(I[valid_mask] * 255).astype(np.uint8)
        F_scaled[valid_mask] = np.round(F[valid_mask] * 255).astype(np.uint8)

    # Create RGB composite
    rgb = np.dstack((T_scaled, F_scaled, I_scaled))

    # Path for composite image
    composite_path = os.path.join(folder, f"TIF_composite_{year_full}.tif")

    # Save composite image
    meta = tif_data['meta'].copy()
    meta.update(count=3, dtype='uint8')
    with rasterio.open(composite_path, 'w', **meta) as dst:
        dst.write(T_scaled, 1)  # Red = Urban likelihood
        dst.write(F_scaled, 2)  # Green = Green space likelihood
        dst.write(I_scaled, 3)  # Blue = Indeterminacy

    # Display the composite image with lat/lon coordinates
    plt.figure(figsize=(fig_width, base_height))
    plt.imshow(rgb, extent=[min_lon, max_lon, min_lat, max_lat], aspect='equal')
    plt.title(f'T-I-F Composite for {year_full}\nRed=Urban, Green=Vegetation, Blue=Indeterminacy')
    plt.xlabel('Longitude (°)')
    plt.ylabel('Latitude (°)')
    plt.colorbar(label='Intensity', shrink=0.8)
    plt.savefig(os.path.join(folder, f"TIF_composite_{year_full}.png"),
                dpi=300, bbox_inches='tight', format='png')
    plt.show()

    return {
        'composite_path': composite_path,
        'T_scaled': T_scaled,
        'I_scaled': I_scaled,
        'F_scaled': F_scaled
    }

def analyze_urban_green_transitions(tif_data, year_folder, year_full=None, stats=None):
    """
    Analyze urban-green transitions and ambiguity based on T, I, F values

    Parameters:
    tif_data (dict): Dictionary with T, I, F arrays
    year_folder (str): Folder name
    year_full (int, optional): Full year
    stats (dict, optional): Statistics from histogram analysis

    Returns:
    dict: Transition analysis results
    """
    # Define folder and year
    folder = str(year_folder)
    if year_full is None:
        if year_folder == '15':
            year_full = 2015
        elif year_folder == '20':
            year_full = 2020
        elif year_folder == '23':
            year_full = 2023

    print(f"Analyzing urban-green transitions for year {year_full}...")

    # Get coordinates from one of the T-I-F rasters
    min_lon, max_lon, min_lat, max_lat = get_lat_lon_extent(tif_data['T_path'])
    aspect_ratio = calculate_aspect_ratio(min_lon, max_lon, min_lat, max_lat)

    # Calculate figure size
    base_height = 10
    fig_width = base_height * aspect_ratio

    # Extract T, I, F arrays
    T = tif_data['T']
    I = tif_data['I']
    F = tif_data['F']

    # Create mask for valid data
    valid_mask = ~np.isnan(T) & ~np.isnan(I) & ~np.isnan(F)

    # Initialize transition map
    transition_map = np.full(T.shape, 255, dtype=np.uint8)  # Initialize with no-data

    # If stats are available, set thresholds based on statistics
    if stats:
        # Set thresholds based on statistics (mean + 0.5*std is often a good threshold)
        t_threshold = stats['T_stats']['mean'] + 0.5 * stats['T_stats']['std']
        f_threshold = stats['F_stats']['mean'] + 0.5 * stats['F_stats']['std']
        i_threshold = stats['I_stats']['mean'] + 0.5 * stats['I_stats']['std']

        # Ensure thresholds are in 0-1 range
        t_threshold = min(max(t_threshold, 0.0), 1.0)
        f_threshold = min(max(f_threshold, 0.0), 1.0)
        i_threshold = min(max(i_threshold, 0.0), 1.0)
    else:
        # Default thresholds if stats not available
        t_threshold = 0.6
        f_threshold = 0.6
        i_threshold = 0.7

    print(f"Using thresholds - Urban: {t_threshold:.3f}, Green: {f_threshold:.3f}, Ambiguity: {i_threshold:.3f}")

    # Apply modified classification rules:
    # 1. First identify urban areas with high T and low F
    urban_mask = (T > t_threshold) & (F < 0.5) & valid_mask
    transition_map[urban_mask] = 0

    # 2. Then identify green areas with high F and low T
    green_mask = (F > f_threshold) & (T < 0.5) & valid_mask & (transition_map == 255)
    transition_map[green_mask] = 1

    # 3. Identify mixed-use areas where both T and F have moderate values
    mixed_mask = (T >= 0.3) & (T <= 0.7) & (F >= 0.3) & (F <= 0.7) & valid_mask & (transition_map == 255)
    transition_map[mixed_mask] = 2

    # 4. Identify high ambiguity areas
    ambiguity_mask = (I > i_threshold) & valid_mask & (transition_map == 255)
    transition_map[ambiguity_mask] = 3

    # Calculate statistics
    total_valid = np.sum(valid_mask)
    urban_percent = np.sum(urban_mask) / total_valid * 100 if total_valid > 0 else 0
    green_percent = np.sum(green_mask) / total_valid * 100 if total_valid > 0 else 0
    mixed_percent = np.sum(mixed_mask) / total_valid * 100 if total_valid > 0 else 0
    ambiguity_percent = np.sum(ambiguity_mask) / total_valid * 100 if total_valid > 0 else 0

    print(f"Urban likelihood: {urban_percent:.2f}%")
    print(f"Green space likelihood: {green_percent:.2f}%")
    print(f"Mixed-use: {mixed_percent:.2f}%")
    print(f"High ambiguity: {ambiguity_percent:.2f}%")

    # Save transition map
    transition_path = os.path.join(folder, f"transition_map_{year_full}.tif")
    meta = tif_data['meta'].copy()
    meta.update(dtype='uint8')
    with rasterio.open(transition_path, 'w', **meta) as dst:
        dst.write(transition_map, 1)

    # Custom colormap for transition map
    transition_colors = ['#FF0000', '#00FF00', '#FFA500', '#0000FF']  # Red, Green, Orange, Blue
    transition_cmap = ListedColormap(transition_colors)

    # Legend labels
    transition_labels = ['Likely Urban', 'Likely Green', 'Mixed Use', 'High Ambiguity']

    # Display transition map with proper coordinates
    plt.figure(figsize=(fig_width, base_height))
    transition_masked = np.ma.masked_where(transition_map == 255, transition_map)
    plt.imshow(transition_masked, cmap=transition_cmap, vmin=0, vmax=3,
               extent=[min_lon, max_lon, min_lat, max_lat], aspect='equal')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=label)
                     for color, label in zip(transition_colors, transition_labels)]
    plt.legend(handles=legend_elements, title='Land Use Transitions', fontsize=12)

    # Add statistics text
    stat_text = (
        f"Urban: {urban_percent:.1f}%\n"
        f"Green: {green_percent:.1f}%\n"
        f"Mixed: {mixed_percent:.1f}%\n"
        f"Ambiguity: {ambiguity_percent:.1f}%"
    )
    plt.annotate(stat_text, xy=(0.02, 0.05), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                 fontsize=10)

    plt.title(f'Urban-Green Transition Analysis for {year_full}', fontsize=14)
    plt.xlabel('Longitude (°)')
    plt.ylabel('Latitude (°)')
    plt.savefig(os.path.join(folder, f"transition_analysis_{year_full}.png"),
                dpi=300, bbox_inches='tight', format='png')
    plt.show()

    return {
        'transition_path': transition_path,
        'urban_percent': urban_percent,
        'green_percent': green_percent,
        'mixed_percent': mixed_percent,
        'ambiguity_percent': ambiguity_percent,
        'urban_mask': urban_mask,
        'green_mask': green_mask,
        'mixed_mask': mixed_mask,
        'ambiguity_mask': ambiguity_mask,
        'thresholds': {
            'urban': t_threshold,
            'green': f_threshold,
            'ambiguity': i_threshold
        }
    }

def experiment_with_thresholds(tif_data, year_full=None):
    """
    Interactive threshold adjustment for urban-green transition analysis

    Parameters:
    tif_data (dict): Dictionary with T, I, F arrays
    year_full (int, optional): Full year
    """
    from ipywidgets import interact, FloatSlider

    # Get coordinates for proper display
    min_lon, max_lon, min_lat, max_lat = get_lat_lon_extent(tif_data['T_path'])
    aspect_ratio = calculate_aspect_ratio(min_lon, max_lon, min_lat, max_lat)
    base_height = 8
    fig_width = base_height * aspect_ratio

    # Extract T, I, F arrays
    T = tif_data['T']
    I = tif_data['I']
    F = tif_data['F']

    # Create mask for valid data
    valid_mask = ~np.isnan(T) & ~np.isnan(I) & ~np.isnan(F)

    def update_map(t_threshold=0.6, f_threshold=0.6, i_threshold=0.7):
        # Initialize transition map
        transition_map = np.full(T.shape, 255, dtype=np.uint8)

        # Apply classification rules
        urban_mask = (T > t_threshold) & (F < 0.5) & valid_mask
        transition_map[urban_mask] = 0

        green_mask = (F > f_threshold) & (T < 0.5) & valid_mask & (transition_map == 255)
        transition_map[green_mask] = 1

        mixed_mask = (T >= 0.3) & (T <= 0.7) & (F >= 0.3) & (F <= 0.7) & valid_mask & (transition_map == 255)
        transition_map[mixed_mask] = 2

        ambiguity_mask = (I > i_threshold) & valid_mask & (transition_map == 255)
        transition_map[ambiguity_mask] = 3

        # Calculate statistics
        total_valid = np.sum(valid_mask)
        urban_percent = np.sum(urban_mask) / total_valid * 100 if total_valid > 0 else 0
        green_percent = np.sum(green_mask) / total_valid * 100 if total_valid > 0 else 0
        mixed_percent = np.sum(mixed_mask) / total_valid * 100 if total_valid > 0 else 0
        ambiguity_percent = np.sum(ambiguity_mask) / total_valid * 100 if total_valid > 0 else 0

        # Custom colormap for transition map
        transition_colors = ['#FF0000', '#00FF00', '#FFA500', '#0000FF']  # Red, Green, Orange, Blue
        transition_cmap = ListedColormap(transition_colors)

        # Display transition map with coordinates
        plt.figure(figsize=(fig_width, base_height))
        transition_masked = np.ma.masked_where(transition_map == 255, transition_map)
        plt.imshow(transition_masked, cmap=transition_cmap, vmin=0, vmax=3,
                   extent=[min_lon, max_lon, min_lat, max_lat], aspect='equal')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=label)
                        for color, label in zip(transition_colors,
                                               ['Likely Urban', 'Likely Green', 'Mixed Use', 'High Ambiguity'])]
        plt.legend(handles=legend_elements, title='Land Use Transitions', fontsize=12)

        # Add statistics text
        stat_text = (
            f"Urban: {urban_percent:.1f}%\n"
            f"Green: {green_percent:.1f}%\n"
            f"Mixed: {mixed_percent:.1f}%\n"
            f"Ambiguity: {ambiguity_percent:.1f}%"
        )
        plt.annotate(stat_text, xy=(0.02, 0.05), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                    fontsize=10)

        plt.title(f'Urban-Green Transition Analysis for {year_full}', fontsize=14)
        plt.xlabel('Longitude (°)')
        plt.ylabel('Latitude (°)')
        plt.show()

        print(f"Urban threshold: {t_threshold:.2f} → {urban_percent:.1f}%")
        print(f"Green threshold: {f_threshold:.2f} → {green_percent:.1f}%")
        print(f"Ambiguity threshold: {i_threshold:.2f} → {ambiguity_percent:.1f}%")
        print(f"Mixed-use: {mixed_percent:.1f}%")

    # Create interactive sliders
    interact(
        update_map,
        t_threshold=FloatSlider(value=0.6, min=0.1, max=0.9, step=0.05, description='Urban:'),
        f_threshold=FloatSlider(value=0.6, min=0.1, max=0.9, step=0.05, description='Green:'),
        i_threshold=FloatSlider(value=0.7, min=0.1, max=0.9, step=0.05, description='Ambiguity:')
    )

def create_comparative_visualization(years_results, output_dir='.'):
    """
    Create comparative visualization for urban-green transitions across years

    Parameters:
    years_results (dict): Dictionary with analysis results for each year
    output_dir (str): Output directory for visualizations
    """
    print("Creating comparative visualization across years...")

    # Extract years and data
    years = sorted(years_results.keys())

    # Bar chart data
    categories = ['Urban', 'Green', 'Mixed', 'Ambiguity']  # Changed 'Mixed Use' to 'Mixed'
    data = {
        'Urban': [years_results[year]['urban_percent'] for year in years],
        'Green': [years_results[year]['green_percent'] for year in years],
        'Mixed': [years_results[year]['mixed_percent'] for year in years],
        'Ambiguity': [years_results[year]['ambiguity_percent'] for year in years]
    }

    # Create bar chart
    plt.figure(figsize=(12, 8))
    x = np.arange(len(years))
    width = 0.2

    # Plot bars
    for i, category in enumerate(categories):
        plt.bar(x + i*width - 0.3, data[category], width,
               label=category,
               color=['#FF0000', '#00FF00', '#FFA500', '#0000FF'][i])

    # Add labels and legend
    plt.xlabel('Year')
    plt.ylabel('Percentage (%)')
    plt.title('Urban-Green Transition Analysis Across Years')
    plt.xticks(x, years)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save chart
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'urban_green_comparison.png'),
                dpi=300, bbox_inches='tight', format='png')
    plt.show()

def execute_urban_green_analysis(analyze_years=None, interactive=False):
    """
    Execute the complete Urban-Green Ambiguity Analysis

    Parameters:
    analyze_years (list, optional): List of years to analyze, defaults to all
    interactive (bool): Whether to use interactive threshold selection

    Returns:
    dict: Results dictionary
    """
    print("\n" + "="*80)
    print("EXECUTING URBAN-GREEN AMBIGUITY ANALYSIS WITH LAT/LON COORDINATES")
    print("="*80 + "\n")

   # Default years to analyze
    if analyze_years is None:
        analyze_years = [2015, 2020, 2023]

    # Map years to folder names
    year_folders = {
        2015: '15',
        2020: '20',
        2023: '23'
    }

    years_results = {}

    for year_full in analyze_years:
        year_folder = year_folders.get(year_full)
        if not year_folder:
            print(f"Skipping year {year_full} - no folder mapping found")
            continue

        print(f"\n{'='*50}")
        print(f"Processing Year {year_full} (Folder: {year_folder})")
        print(f"{'='*50}")

        try:
            # Step 1: Calculate T, I, F values
            print("\nStep 1: Calculating T, I, F values...")
            tif_data = calculate_tif_values(year_folder, year_full)

            # Step 2: Visualize histograms to help with threshold selection
            print("\nStep 2: Visualizing T, I, F histograms...")
            stats = visualize_tif_histograms(tif_data, year_full)

            # Step 3: Create TIF composite visualization
            print("\nStep 3: Creating TIF composite visualization...")
            composite_data = create_tif_composite(tif_data, year_folder, year_full)

            # Step 4: Interactive threshold selection (if requested)
            if interactive:
                print("\nStep 4: Interactive threshold selection...")
                print("Use the sliders to adjust thresholds and observe the results")
                experiment_with_thresholds(tif_data, year_full)

                # Ask user if they want to proceed with manual thresholds
                user_input = input("Do you want to specify custom thresholds? (y/n): ")
                if user_input.lower() == 'y':
                    t_threshold = float(input("Urban threshold (0.0-1.0): "))
                    f_threshold = float(input("Green threshold (0.0-1.0): "))
                    i_threshold = float(input("Ambiguity threshold (0.0-1.0): "))

                    # Override stats with user-specified thresholds
                    stats = {
                        'T_stats': {'mean': t_threshold - 0.1, 'median': 0, 'std': 0.2},
                        'F_stats': {'mean': f_threshold - 0.1, 'median': 0, 'std': 0.2},
                        'I_stats': {'mean': i_threshold - 0.1, 'median': 0, 'std': 0.2}
                    }

                    # Step 5: Analyze urban-green transitions with custom thresholds
                    print("\nStep 5: Analyzing urban-green transitions with custom thresholds...")
                    transition_results = analyze_urban_green_transitions(tif_data, year_folder, year_full, stats)
                else:
                    # Step 5: Analyze urban-green transitions with automatic thresholds
                    print("\nStep 5: Analyzing urban-green transitions with automatic thresholds...")
                    transition_results = analyze_urban_green_transitions(tif_data, year_folder, year_full, stats)
            else:
                # Step 4: Analyze urban-green transitions
                print("\nStep 4: Analyzing urban-green transitions...")
                transition_results = analyze_urban_green_transitions(tif_data, year_folder, year_full, stats)

            # Store results for comparative analysis
            years_results[str(year_full)] = transition_results

            print(f"\nUrban-Green Analysis completed successfully for year {year_full}")

        except Exception as e:
            print(f"Error processing year {year_full}: {str(e)}")
            import traceback
            traceback.print_exc()

    # Create comparative visualization
    if len(years_results) > 1:
        create_comparative_visualization(years_results)

    print("\nUrban-Green Ambiguity Analysis completed successfully")
    print("All maps now include proper latitude/longitude coordinates!")
    print("All visualizations saved as PNG files with geographic referencing.")
    return years_results

# Use this function to analyze a single year with interactivity
def analyze_single_year(year_folder, year_full=None, interactive=True):
    """
    Analyze a single year with interactive threshold selection

    Parameters:
    year_folder (str): Folder name (e.g., '15')
    year_full (int, optional): Full year (e.g., 2015)
    interactive (bool): Whether to use interactive threshold selection

    Returns:
    dict: Results dictionary
    """
    # Define year if not specified
    if year_full is None:
        if year_folder == '15':
            year_full = 2015
        elif year_folder == '20':
            year_full = 2020
        elif year_folder == '23':
            year_full = 2023

    # Execute analysis for this year only
    return execute_urban_green_analysis(analyze_years=[year_full], interactive=interactive)

# Main execution
if __name__ == "__main__":
    # Execute analysis for all years with lat/lon coordinates
    results = execute_urban_green_analysis()
    print("\n" + "="*80)
    print("T-I-F ANALYSIS WITH COORDINATES COMPLETED SUCCESSFULLY!")
    print("="*80)


# In[11]:


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import rasterio
from rasterio.plot import show
from scipy.ndimage import generic_filter
from skimage import filters
import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# # Set working directory - update this to your project folder path
# base_dir = r"C:\Users\divya\Desktop\urbanization in gurgaon"
# os.chdir(base_dir)

# LULC class definitions
class_names = {
    0: 'Urban',
    1: 'Vegetation',
    2: 'Water',
    3: 'Barren'
}

# Define color map for visualization of LULC classes
colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00']  # Red, Green, Blue, Yellow
lulc_cmap = ListedColormap(colors)

def get_lat_lon_extent(raster_path):
    """
    Get the latitude and longitude extent from a raster file

    Parameters:
    raster_path (str): Path to the raster file

    Returns:
    tuple: (min_lon, max_lon, min_lat, max_lat)
    """
    with rasterio.open(raster_path) as src:
        # Get the bounds in the CRS of the raster
        bounds = src.bounds

        # If the CRS is not geographic (lat/lon), we might need to transform
        # For now, assuming the coordinates are already in a geographic system
        # or can be used directly

        min_lon, min_lat, max_lon, max_lat = bounds.left, bounds.bottom, bounds.right, bounds.top

        return min_lon, max_lon, min_lat, max_lat

def calculate_texture_based_ambiguity(classified_image, window_size=5):
    """
    Calculate ambiguity based on local class diversity and edge detection

    Parameters:
    classified_image: Array of classified pixels (your RF classification result)
    window_size: Size of moving window for analysis (odd number)

    Returns:
    ambiguity: Array of ambiguity values ranging from 0-1
    """
    # Create a copy to avoid modifying the original
    classified = classified_image.copy().astype(float)

    # Handle no-data values if any (replace with NaN)
    if 255 in np.unique(classified):
        classified[classified == 255] = np.nan

    # Function to calculate class diversity in a window
    def class_diversity(window):
        # Remove NaN values
        valid = window[~np.isnan(window)]
        if len(valid) == 0:
            return 0.0  # Return 0 instead of NaN for empty windows

        # Count unique classes in window
        unique_classes = np.unique(valid)

        # Calculate diversity metrics
        n_pixels = len(valid)
        n_unique = len(unique_classes)

        # If all pixels are the same class, diversity is 0
        if n_unique <= 1:
            return 0.0

        # Calculate Shannon diversity index
        diversity = 0
        for cls in unique_classes:
            p = np.sum(valid == cls) / n_pixels
            if p > 0:  # Avoid log(0)
                diversity -= p * np.log(p)

        # Normalize by maximum possible diversity
        max_diversity = np.log(n_unique) if n_unique > 0 else 1
        norm_diversity = diversity / max_diversity if max_diversity > 0 else 0

        return norm_diversity

    # Apply edge detection (Sobel filter) - handle NaN values
    mask = ~np.isnan(classified)
    temp_classified = classified.copy()
    temp_classified[~mask] = 0  # Temporarily replace NaNs with 0 for edge detection

    edges_h = filters.sobel_h(temp_classified)
    edges_v = filters.sobel_v(temp_classified)
    edges = np.sqrt(edges_h*2 + edges_v*2)

    # Restore NaN values
    edges[~mask] = np.nan

    # Normalize edge strength to [0,1]
    valid_edges = edges[~np.isnan(edges)]
    if len(valid_edges) > 0:
        min_edge = np.min(valid_edges)
        max_edge = np.max(valid_edges)
        if max_edge > min_edge:
            edge_strength = (edges - min_edge) / (max_edge - min_edge)
        else:
            edge_strength = np.zeros_like(edges)
    else:
        edge_strength = np.zeros_like(edges)

    # Apply moving window analysis for class diversity
    diversity = generic_filter(classified,
                               class_diversity,
                               size=window_size,
                               mode='constant',
                               cval=0.0)  # Use 0.0 instead of np.nan for edge handling

    # Replace any NaN values with zeros
    diversity = np.nan_to_num(diversity, nan=0.0)

    # Combine edge strength and diversity to get final ambiguity score
    # Weight can be adjusted (higher weight on edges vs diversity)
    edge_weight = 0.7
    diversity_weight = 0.3

    ambiguity = (edge_weight * edge_strength + diversity_weight * diversity)

    # Replace any remaining NaN values with zeros
    ambiguity = np.nan_to_num(ambiguity, nan=0.0)

    # Normalize final result to [0,1] range
    valid_ambiguity = ambiguity[mask]
    if len(valid_ambiguity) > 0:
        min_amb = np.min(valid_ambiguity)
        max_amb = np.max(valid_ambiguity)
        if max_amb > min_amb:
            normalized_ambiguity = (ambiguity - min_amb) / (max_amb - min_amb)
        else:
            normalized_ambiguity = np.zeros_like(ambiguity)
    else:
        normalized_ambiguity = np.zeros_like(ambiguity)

    # Make sure there are no NaN values in the final result
    normalized_ambiguity = np.nan_to_num(normalized_ambiguity, nan=0.0)

    return normalized_ambiguity

def analyze_texture_based_transitions(year_folder, year_full=None, window_size=5, ambiguity_percentile=75):
    """
    Analyze urban-green transitions using texture-based ambiguity

    Parameters:
    year_folder (str): Folder name
    year_full (int, optional): Full year
    window_size (int): Size of window for texture analysis
    ambiguity_percentile (float): Percentile threshold for high ambiguity areas

    Returns:
    dict: Transition analysis results
    """
    # Define folder and year
    folder = str(year_folder)
    if year_full is None:
        if year_folder == '15':
            year_full = 2015
        elif year_folder == '20':
            year_full = 2020
        elif year_folder == '23':
            year_full = 2023

    print(f"Analyzing texture-based urban-green transitions for year {year_full}...")

    # Path to the classified raster
    classified_path = os.path.join(folder, f"RFclassified_{year_full}.tif")

    # Check if classified file exists
    if not os.path.exists(classified_path):
        print(f"WARNING: Classified image not found: {classified_path}")
        return None

    # Open the classified image
    with rasterio.open(classified_path) as src:
        # Read the classified data
        classified = src.read(1)
        meta = src.meta.copy()

    # Get lat/lon extent for proper axis labeling
    min_lon, max_lon, min_lat, max_lat = get_lat_lon_extent(classified_path)

    # Calculate texture-based ambiguity
    print(f"Calculating texture-based ambiguity with window size {window_size}...")
    ambiguity = calculate_texture_based_ambiguity(classified, window_size=window_size)

    # Save ambiguity map
    ambiguity_path = os.path.join(folder, f"ambiguity_texture_{year_full}.tif")
    meta_float = meta.copy()
    meta_float.update(dtype='float32')
    with rasterio.open(ambiguity_path, 'w', **meta_float) as dst:
        dst.write(ambiguity.astype('float32'), 1)

    # Create transition map
    print("Creating transition map...")
    transition_map = np.full_like(classified, 255, dtype=np.uint8)

    # Create mask for valid data (not no-data in original classification)
    valid_mask = classified != 255

    # Calculate ambiguity threshold based on percentile
    ambiguity_threshold = np.percentile(ambiguity[valid_mask], ambiguity_percentile)
    print(f"Ambiguity threshold (percentile {ambiguity_percentile}): {ambiguity_threshold:.3f}")

    # Create masks for different transition categories
    # Category 0: Urban areas (class 0 in original classification)
    urban_mask = (classified == 0) & valid_mask

    # Category 1: Green areas (class 1 in original classification)
    green_mask = (classified == 1) & valid_mask

    # Category 3: High ambiguity areas (top percentage of ambiguity values)
    high_ambiguity_mask = (ambiguity > ambiguity_threshold) & valid_mask

    # Category 2: Mixed use - not clearly urban or green, but not high ambiguity
    mixed_mask = (~urban_mask & ~green_mask & valid_mask) & ~high_ambiguity_mask

    # Apply urban and green masks without high ambiguity regions
    transition_map[urban_mask & ~high_ambiguity_mask] = 0
    transition_map[green_mask & ~high_ambiguity_mask] = 1

    # Apply mixed-use and high ambiguity masks
    transition_map[mixed_mask] = 2
    transition_map[high_ambiguity_mask] = 3

    # Calculate statistics
    total_valid = np.sum(valid_mask)
    urban_percent = np.sum(transition_map == 0) / total_valid * 100 if total_valid > 0 else 0
    green_percent = np.sum(transition_map == 1) / total_valid * 100 if total_valid > 0 else 0
    mixed_percent = np.sum(transition_map == 2) / total_valid * 100 if total_valid > 0 else 0
    ambiguity_percent = np.sum(transition_map == 3) / total_valid * 100 if total_valid > 0 else 0

    print(f"Urban: {urban_percent:.2f}%")
    print(f"Green space: {green_percent:.2f}%")
    print(f"Mixed-use: {mixed_percent:.2f}%")
    print(f"High ambiguity: {ambiguity_percent:.2f}%")

    # Save transition map
    transition_path = os.path.join(folder, f"transition_map_texture_{year_full}.tif")
    with rasterio.open(transition_path, 'w', **meta) as dst:
        dst.write(transition_map, 1)

    # Custom colormap for transition map
    transition_colors = ['#FF0000', '#00FF00', '#FFA500', '#0000FF']  # Red, Green, Orange, Blue
    from matplotlib.patches import Patch

    # Create transition colormap using the imported ListedColormap from top of file
    transition_cmap = ListedColormap(transition_colors)

    # Legend labels
    transition_labels = ['Urban', 'Green', 'Mixed Use', 'High Ambiguity']

    # Calculate proper aspect ratio for the geographic area
    lat_range = max_lat - min_lat
    lon_range = max_lon - min_lon

    # Adjust for latitude (approximately cos(latitude) correction)
    lat_center = (max_lat + min_lat) / 2
    aspect_ratio = lon_range / lat_range * np.cos(np.radians(lat_center))

    # Calculate appropriate figure size
    base_height = 10
    fig_width = base_height * aspect_ratio

    # Display transition map with lat/lon coordinates
    plt.figure(figsize=(fig_width, base_height))
    transition_masked = np.ma.masked_where(transition_map == 255, transition_map)

    # Use extent parameter to set proper coordinate system with equal aspect
    plt.imshow(transition_masked, cmap=transition_cmap, vmin=0, vmax=3,
               extent=[min_lon, max_lon, min_lat, max_lat], aspect='equal')

    # Add legend
    legend_elements = [Patch(facecolor=color, label=label)
                     for color, label in zip(transition_colors, transition_labels)]
    plt.legend(handles=legend_elements, title='Land Use Transitions', fontsize=12)

    # Set axis labels
    plt.xlabel('Longitude (°)', fontsize=12)
    plt.ylabel('Latitude (°)', fontsize=12)
    plt.title(f'Texture-Based Urban-Green Transition Analysis for {year_full}', fontsize=14)

    # Save as PNG
    plt.savefig(os.path.join(folder, f"transition_analysis_texture_{year_full}.png"),
                dpi=300, bbox_inches='tight', format='png')
    plt.show()

    # Visualize the raw ambiguity map with improved colormap
    plt.figure(figsize=(fig_width, base_height))

    # Create a masked array where areas outside Gurgaon (no-data) are white
    ambiguity_masked = np.ma.masked_where(classified == 255, ambiguity)

    # Import matplotlib colormap utilities
    import matplotlib.cm as cm

    # Get the plasma colormap
    plasma_cmap = cm.get_cmap('plasma')

    # Create a new colormap with white for bad values
    plasma_cmap.set_bad(color='white')

    # Use the colormap for visualization with equal aspect ratio
    im = plt.imshow(ambiguity_masked, cmap=plasma_cmap,
                    extent=[min_lon, max_lon, min_lat, max_lat], aspect='equal')

    plt.colorbar(im, label='Ambiguity', shrink=0.8)
    plt.xlabel('Longitude (°)', fontsize=12)
    plt.ylabel('Latitude (°)', fontsize=12)
    plt.title(f'Texture-Based Ambiguity for {year_full}', fontsize=14)

    # Save as PNG
    plt.savefig(os.path.join(folder, f"ambiguity_texture_{year_full}.png"),
                dpi=300, bbox_inches='tight', format='png')
    plt.show()

    return {
        'transition_path': transition_path,
        'ambiguity_path': ambiguity_path,
        'urban_percent': urban_percent,
        'green_percent': green_percent,
        'mixed_percent': mixed_percent,
        'ambiguity_percent': ambiguity_percent
    }

def create_comparative_visualization(years_data, output_dir='.'):
    """
    Create comparative visualization for texture-based urban-green transitions across years

    Parameters:
    years_data (dict): Dictionary with analysis results for each year
    output_dir (str): Output directory for visualizations
    """
    print("Creating comparative visualization across years...")

    # Extract years and data
    years = sorted(years_data.keys())

    # Bar chart data
    categories = ['Urban', 'Green', 'Mixed', 'Ambiguity']
    data = {
        'Urban': [years_data[year]['urban_percent'] for year in years],
        'Green': [years_data[year]['green_percent'] for year in years],
        'Mixed': [years_data[year]['mixed_percent'] for year in years],
        'Ambiguity': [years_data[year]['ambiguity_percent'] for year in years]
    }

    # Create bar chart
    plt.figure(figsize=(12, 8))
    x = np.arange(len(years))
    width = 0.2

    # Plot bars
    for i, category in enumerate(categories):
        plt.bar(x + i*width - 0.3, data[category], width,
               label=category,
               color=['#FF0000', '#00FF00', '#FFA500', '#0000FF'][i])

    # Add labels and legend
    plt.xlabel('Year')
    plt.ylabel('Percentage (%)')
    plt.title('Texture-Based Urban-Green Transition Analysis Across Years')
    plt.xticks(x, years)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save chart as PNG
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'urban_green_texture_comparison.png'),
                dpi=300, bbox_inches='tight', format='png')
    plt.show()

def execute_texture_based_analysis(window_size=7, ambiguity_percentile=75):
    """
    Execute the complete Texture-Based Urban-Green Ambiguity Analysis

    Parameters:
    window_size (int): Size of window for texture analysis
    ambiguity_percentile (float): Percentile threshold for high ambiguity areas
    """
    print("\n" + "="*80)
    print("EXECUTING TEXTURE-BASED URBAN-GREEN AMBIGUITY ANALYSIS")
    print("="*80 + "\n")

    # Process each year
    years = [
        ('15', 2015),
        ('20', 2020),
        ('23', 2023)
    ]

    years_results = {}

    for year_folder, year_full in years:
        print(f"\n{'='*50}")
        print(f"Processing Year {year_full} (Folder: {year_folder})")
        print(f"{'='*50}")

        try:
            # Analyze texture-based transitions
            results = analyze_texture_based_transitions(
                year_folder,
                year_full,
                window_size=window_size,
                ambiguity_percentile=ambiguity_percentile
            )

            if results:
                # Store results for comparative analysis
                years_results[str(year_full)] = results

                print(f"\nTexture-Based Urban-Green Analysis completed successfully for year {year_full}")

        except Exception as e:
            print(f"Error processing year {year_full}: {str(e)}")
            import traceback
            traceback.print_exc()

    # Create comparative visualization
    if len(years_results) > 1:
        create_comparative_visualization(years_results)

    print("\nTexture-Based Urban-Green Ambiguity Analysis completed successfully")
    print("All results saved as PNG files")

if __name__ == "__main__":
    # You can adjust these parameters
    window_size = 7  # Size of window for texture analysis (larger = smoother boundaries)
    ambiguity_percentile = 75  # Percentile threshold for high ambiguity (higher = fewer ambiguity zones)

    execute_texture_based_analysis(
        window_size=window_size,
        ambiguity_percentile=ambiguity_percentile
    )

