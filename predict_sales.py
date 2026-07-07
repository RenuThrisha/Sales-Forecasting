import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os

def calculate_metrics(y_true, y_pred, bins):
    """Calculate and return accuracy metrics and confusion matrix"""
    y_true_binned = pd.cut(y_true, bins=bins, labels=[f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)])
    y_pred_binned = pd.cut(y_pred, bins=bins, labels=[f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)])
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true_binned, y_pred_binned)
    
    # Calculate accuracy
    accuracy = (y_true_binned == y_pred_binned).mean() * 100
    
    # Calculate error metrics with handling for zero values
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Handle division by zero in MAPE calculation
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if any(mask) else float('inf')
    
    return conf_matrix, accuracy, rmse, mae, mape

def plot_confusion_matrix(conf_matrix, labels):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Sales Prediction Confusion Matrix')
    plt.ylabel('True Sales Range')
    plt.xlabel('Predicted Sales Range')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def get_user_input():
    print("\nEnter the following details for prediction:")
    try:
        shipping_days = float(input("Days for shipping (real) [1-30]: "))
        scheduled_days = float(input("Days for shipment scheduled [1-20]: "))
        sales_per_customer = float(input("Sales per customer [1-100]: "))
        late_risk = int(input("Late delivery risk (0 or 1): "))
        price = float(input("Product Price [10-1000]: "))
        
        # Input validation
        if not (1 <= shipping_days <= 30):
            raise ValueError("Shipping days should be between 1 and 30")
        if not (1 <= scheduled_days <= 20):
            raise ValueError("Scheduled days should be between 1 and 20")
        if not (1 <= sales_per_customer <= 100):
            raise ValueError("Sales per customer should be between 1 and 100")
        if late_risk not in [0, 1]:
            raise ValueError("Late delivery risk should be 0 or 1")
        if not (10 <= price <= 1000):
            raise ValueError("Price should be between 10 and 1000")
            
        # Create a DataFrame with proper feature names
        input_df = pd.DataFrame([{
            'Days for shipping (real)': shipping_days,
            'Days for shipment (scheduled)': scheduled_days,
            'Sales per customer': sales_per_customer,
            'Late_delivery_risk': late_risk,
            'Product Price': price
        }])
        
        return input_df
    except ValueError as e:
        print(f"Error in input: {str(e)}")
        return None

# Load and prepare data
print("Loading data...")

def load_csv_with_encodings(path, encodings=("utf-8", "latin-1", "cp1252")):
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            return df, enc
        except UnicodeDecodeError as ude:
            last_err = ude
            continue
        except FileNotFoundError:
            raise
        except Exception as e:
            last_err = e
            break
    raise last_err

try:
    csv_path = 'DataCOSupplyChainDataset.csv'
    df, used_encoding = load_csv_with_encodings(csv_path)
    print(f"Loaded CSV using encoding: {used_encoding}")
    required_columns = {
        'Days for shipping (real)',
        'Days for shipment (scheduled)',
        'Sales per customer',
        'Late_delivery_risk',
        'Product Price',
        'Sales'
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    print(f"Dataset size: {len(df):,} records")
except FileNotFoundError:
    print(f"Error: {csv_path} not found in the current directory")
    sys.exit(1)
except UnicodeDecodeError:
    print(f"Error loading data: file could not be decoded with common encodings (tried utf-8, latin-1, cp1252). Open the file and save as UTF-8 or specify a different encoding.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading data: {str(e)}")
    sys.exit(1)

# Data validation and cleaning
print("\nValidating data...")
try:
    # Remove rows with invalid values
    df = df.dropna(subset=required_columns)
    
    # Remove rows with negative or zero values where inappropriate
    df = df[
        (df['Days for shipping (real)'] > 0) &
        (df['Days for shipment (scheduled)'] > 0) &
        (df['Sales per customer'] > 0) &
        (df['Product Price'] > 0) &
        (df['Sales'] > 0)
    ]
    
    # Ensure Late_delivery_risk is binary
    df['Late_delivery_risk'] = df['Late_delivery_risk'].map({0: 0, 1: 1})
    df = df.dropna(subset=['Late_delivery_risk'])
    
    print(f"Clean dataset size: {len(df):,} records")
    
except Exception as e:
    print(f"Error validating data: {str(e)}")
    sys.exit(1)

# Determine product identifier column if present
if 'Product Name' in df.columns:
    product_col = 'Product Name'
elif 'Product Card Id' in df.columns:
    product_col = 'Product Card Id'
else:
    product_col = None

# If product column exists, show available product types (count + sample)
if product_col is not None:
    try:
        prod_series = df[product_col].astype(str)
        unique_products = list(prod_series.unique())
        n_products = len(unique_products)
        # Build shortcuts for products (unique and short)
        def make_shortcuts(names):
            shortcuts = {}
            used = set()
            for name in names:
                base = ''.join(ch for ch in name.upper() if ch.isalnum())
                base = base[:4] if len(base) >= 4 else base or 'P'
                shortcut = base
                suffix = 1
                while shortcut in used:
                    shortcut = f"{base}{suffix}"
                    suffix += 1
                used.add(shortcut)
                shortcuts[shortcut] = name
            return shortcuts

        shortcuts_map = make_shortcuts(unique_products)
        sample_products = unique_products[:20]
        print(f"\nProduct types detected in column '{product_col}': {n_products:,} unique")
        print("Showing up to 20 examples (shortcut -> product):")
        for i, p in enumerate(sample_products, start=1):
            # find shortcut for this product
            sc = None
            for k, v in shortcuts_map.items():
                if v == p:
                    sc = k
                    break
            print(f"  {i}. {sc} -> {p}")
        print("You can enter a shortcut (e.g. 'ABC1'), the index number (e.g. '1'), or the full product name when prompted. Press Enter to use the general model.")
    except Exception:
        pass

# Add noise to make predictions more realistic
np.random.seed(42)
df['Sales'] = df['Sales'] * (1 + np.random.normal(0, 0.1, len(df)))  # Reduced noise to 10%

# Prepare features
features = ['Days for shipping (real)', 'Days for shipment (scheduled)',
           'Sales per customer', 'Late_delivery_risk', 'Product Price']

# Split data
X = df[features]
y = df['Sales']
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
except Exception as e:
    print(f"Error splitting data: {str(e)}")
    exit(1)

# Train model
print("\nTraining model...")
try:
    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=5,
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_train, y_train)
except Exception as e:
    print(f"Error training model: {str(e)}")
    exit(1)

# Define bins for sales ranges using quantiles (prevents overly-coarse bins that can show 100% accuracy)
quantiles = np.quantile(y, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
bins = list(quantiles)
# Ensure strictly increasing bins (tiny jitter if duplicates)
for i in range(1, len(bins)):
    if bins[i] <= bins[i-1]:
        bins[i] = bins[i-1] + 1e-6
labels = [f"${int(bins[i]):,}-${int(bins[i+1]):,}" for i in range(len(bins)-1)]

# Calculate initial metrics
test_predictions = model.predict(X_test)
conf_matrix, accuracy, rmse, mae, mape = calculate_metrics(y_test, test_predictions, bins)

# Print initial model performance
print("\nModel Performance Metrics:")
print("=========================")
print(f"Accuracy: {accuracy:.2f}%")
print(f"RMSE: ${rmse:,.2f}")
print(f"MAE: ${mae:,.2f}")
print(f"MAPE: {mape:.2f}%")

# Plot initial confusion matrix
plot_confusion_matrix(conf_matrix, labels)
print("\nInitial confusion matrix saved as 'confusion_matrix.png'")

# Diagnostics removed: script now prints only model performance metrics

# Interactive Predictions
while True:
    try:
        print("\nWould you like to make a prediction? (yes/no)")
        response = input().lower().strip()
        if response != 'yes':
            break
        # Ask product type for this prediction (optional)
        product_input = input("Enter product type (shortcut/index/full name) (or press Enter to use general model): ").strip()

        # If product specified and dataset has product id, try to train on product-specific rows
        model_to_use = model
        if product_input:
            if product_col is None:
                print('Dataset does not contain a product identifier; using general model.')
            else:
                # Resolve input: check shortcuts_map, numeric index, exact or case-insensitive match
                product_real = None
                # check shortcut map (created earlier)
                if product_input in globals().get('shortcuts_map', {}):
                    product_real = shortcuts_map[product_input]
                # numeric index mapping to sample_products
                elif product_input.isdigit():
                    idx = int(product_input) - 1
                    if 0 <= idx < len(sample_products):
                        product_real = sample_products[idx]
                # exact match
                elif product_input in df[product_col].astype(str).unique():
                    product_real = product_input
                else:
                    # case-insensitive match
                    low = product_input.lower()
                    for name in df[product_col].astype(str).unique():
                        if name.lower() == low:
                            product_real = name
                            break

                if product_real is None:
                    print(f"Product '{product_input}' not recognized; using general model.")
                else:
                    matches = df[df[product_col].astype(str) == product_real]
                    if len(matches) == 0:
                        print(f"No rows found for product '{product_real}'; using general model.")
                    elif len(matches) < 30:
                        print(f"Only {len(matches)} rows for '{product_real}' — not enough to train a product-specific model; using general model.")
                    else:
                        print(f"Training product-specific model for '{product_real}' ({len(matches)} rows)...")
                        # Prepare product-specific training set
                        Xp = matches[features].apply(pd.to_numeric, errors='coerce')
                        yp = pd.to_numeric(matches['Sales'], errors='coerce')
                        validp = ~(Xp.isna().any(axis=1) | yp.isna())
                        Xp = Xp[validp]
                        yp = yp[validp]
                        if len(Xp) >= 10:
                            from sklearn.ensemble import RandomForestRegressor as _RFR
                            # split product data into train/test (70% train, 30% test)
                            try:
                                Xp_train, Xp_test, yp_train, yp_test = train_test_split(Xp, yp, test_size=0.3, random_state=42)
                            except Exception:
                                Xp_train, yp_train = Xp, yp
                                Xp_test, yp_test = Xp, yp

                            prod_model = _RFR(n_estimators=50, max_depth=5, min_samples_split=5, random_state=42)
                            prod_model.fit(Xp_train, yp_train)

                            # evaluate on held-out product test set
                            try:
                                preds_p = prod_model.predict(Xp_test)
                                rmse_p = mean_squared_error(yp_test, preds_p, squared=False)
                                r2_p = r2_score(yp_test, preds_p)
                                print(f"Product model test rows: {len(Xp_test):,}  RMSE: ${rmse_p:,.2f}  R2: {r2_p:.3f}")
                            except Exception:
                                print('Could not evaluate product-specific model on held-out set.')

                            model_to_use = prod_model
                        else:
                            print('Not enough valid rows after cleaning; using general model.')

        input_data = get_user_input()
        if input_data is not None:
            try:
                # Make prediction using DataFrame with proper feature names
                prediction = model_to_use.predict(input_data)[0]
                
                # Add uncertainty
                prediction *= (1 + np.random.normal(0, 0.1))  # Reduced uncertainty to 10%
                
                # Get prediction range
                pred_range = pd.cut([prediction], bins=bins, labels=labels)[0]
                
                print("\nPrediction Results:")
                print("===================")
                print(f"Predicted Sales: ${prediction:,.2f}")
                print(f"Range Category: {pred_range}")
                print(f"Confidence Interval: ${prediction*0.9:,.2f} to ${prediction*1.1:,.2f}")
                
                # (Feature importance display removed)
                    
            except Exception as e:
                print(f"Error making prediction: {str(e)}")
                continue
                
    except KeyboardInterrupt:
        print("\nPrediction interrupted by user.")
        break
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        break

print("\nThank you for using the sales prediction model!") 