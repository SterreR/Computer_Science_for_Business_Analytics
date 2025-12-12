#%% Import packages
# import numpy as np
import pandas as pd
import numpy as np
import hashlib
import json
import re

pd.set_option('display.max_colwidth', None)

#%% Functions
def standardise_title(df):
    """
    Standardise the 'title' column of the TV-dataset:
    - Convert to lowercase
    - Remove double and trailing spaces
    - Remove website names (like "Amazon", "BestBuy", etc.)
    - Convert unit notations (inch, hertz, lbs, etc.) to a standard format
    - Convert common abbreviations (like "diagonal", "diag", etc.) to a standard format
    - Replace linking words (like "and", "&", etc.) with a standard format
    - Standardise common abbreviations (like "4K", "UHD", etc.)
    - Remove special characters (like "-", "_", etc.)
    - Standardise words (like "television" to "tv", etc.)
    - Remove extra spaces
    """

    df = df.copy()

    # Convert titles to lowercase
    df['title'] = df['title'].str.lower()

    # Remove extra spaces
    df['title'] = df['title'].str.replace(r'\s+', ' ', regex = True).str.strip()
    
    
    # Remove website names
    df['title'] = df['title'].str.replace(
        r'\b(amazon|newegg\.com|best buy|thenerds\.net)\b', '', regex = True
        )
    
    # Convert unit notations to a standard format
    df['title'] = df['title'].str.replace(
        r'(?<=\d)\s*[\"”“″]', 'inch', regex = True
    )
    df['title'] = df['title'].str.replace(
        r'(?<=\d)\s*-?\s*inch(?:es)?\b', 'inch', regex = True
    )
    df['title'] = df['title'].str.replace(
        r'\s*-?\s*hertz\b', 'hz', regex = True
    )
    df['title'] = df['title'].str.replace(
        r'\s*-?\s*hz\b', 'hz', regex = True
    )
    df['title'] = df['title'].str.replace(
        r'\s*-?\s*lbs?\.?\b', 'lb', regex = True
    )
    df['title'] = df['title'].str.replace(
        r'(?<=\d)\s*-?\s*volts?\b', 'v', regex = True
    )
    df['title'] = df['title'].str.replace(
        r'(?<=\d)\s*-?\s*v\b', 'v', regex = True
    )

    # Convert common abbreviations to a standard format
    df['title'] = df['title'].str.replace(
        r'\bdiagonal(?:ly)?\b', 'diag', regex = True
    )
    df['title'] = df['title'].str.replace(
        r'\bdiag\.?\b', 'diag', regex = True
    )
    df['title'] = df['title'].str.replace(
        r'\bled[\s\-\/]?lcd\b', 'led-lcd', regex = True
    )
    df['title'] = df['title'].str.replace(
        r'\blcd[\s\-\/]?led\b', 'led-lcd', regex = True
    )

    # Replace linking words and symbols with a standard format
    df['title'] = df['title'].str.replace(
        r'\b(and|&|or)\b', '', regex = True
    )
    df['title'] = df['title'].str.replace(
        r'(?<!\S)w/\s*', 'with ', regex = True
    )
    df['title'] = df['title'].str.replace(
        r'\s+-\s+', ' ', regex=True
    )
    df['title'] = df['title'].str.replace(
        r'\s+/\s+', ' ', regex=True
    )

    # Remove \x99
    df['title'] = df['title'].str.replace(
        r'\x99', '', regex = True
    )

    # Stardardise "television" to "tv"
    df['title'] = df['title'].str.replace(
        r'\btelevisions?\b', 'tv', regex = True
    )

    # Remove brackets from titles (but not their contents)
    df['title'] = df['title'].str.replace(
        r'[\(\)]', ' ', regex = True
    )

    # Some specs have not been parsed correctly. (E.g. "5412 inch" instead of "54.12 inch" or "54 1/2 inch".) Since there is no way of telling what the correct value is, we remove such specs.
    df['title'] = df['title'].str.replace(
        r'\b(?!\d*[.,])\d{4,}inch\b', '', regex = True
    )

    # Remove extra and trailing spaces again
    df['title'] = df['title'].str.replace(r'\s+', ' ', regex = True).str.strip()

    return df

def get_shingles(title, k):
    """
    Create character shingles of size k from the input title.
    """
    shingles = set()
    for i in range(len(title) - k + 1):
        shingle = title[i:i + k]
        shingles.add(shingle)
    
    return shingles

def shingles_per_product(df, k):
    """
    Save the shingles for each product in a dictionary.
    """
    product_shingles = {}
    for index, row in df.iterrows():
        product_ID = row['product_ID']
        title = row['title']
        shingles = get_shingles(title, k)
        product_shingles[product_ID] = shingles

    return product_shingles

def find_all_unique_shingles(product_shingles):
    """
    Find all unique shingles across all products.
    """
    all_unique_shingles = set()
    for shingles in product_shingles.values():
        all_unique_shingles.update(shingles)
    
    return all_unique_shingles


def get_model_words(title):
    pattern = re.compile(r'[a-z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-z0-9]*')
    
    if pd.isna(title):
        return set()    # Return an empty set if title is NaN
    
    tokens = title.split()
    model_words = set()
    for token in tokens:
        match = pattern.fullmatch(token)
        if match and token not in mw_stopwords:
            model_words.add(token)
    
    return model_words

def model_words_per_product(df):
    product_model_words = {}
    for index, row in df.iterrows():
        product_ID = row['product_ID']
        title = row['title']
        model_words = get_model_words(title)
        product_model_words[product_ID] = model_words

    return product_model_words

def find_all_unique_model_words(product_model_words):
    all_unique_model_words = set()
    for model_words in product_model_words.values():
        all_unique_model_words.update(model_words)
    
    return all_unique_model_words


    
    

def create_signature_matrix(product_mw_IDs, hash_funcs):
    """
    Create the signature matrix using MinHashing.
    """
    n_hashes = len(hash_funcs)
    signature_matrix = {}
    for product_ID in product_mw_IDs.keys():
        signature_matrix[product_ID] = [float('inf')] * n_hashes

    for product_ID, mw_IDs in product_mw_IDs.items():
        if not mw_IDs:
            continue            # Check whether the set of shingles is empty to avoid errors
        for i, hash_func in enumerate(hash_funcs):
            # Compute the minimum hash value for the current product and hash function
            min_hash_value = min(hash_func(mw_ID) for mw_ID in mw_IDs)
            signature_matrix[product_ID][i] = min_hash_value

    return signature_matrix

def hash_LSH(band_signature):
    """
    Hash function for LSH band signatures using SHA-1.
    """

    return hashlib.sha1(str(band_signature).encode()).hexdigest()

def find_cand_pairs(b, r, signature_matrix):
    cand_pairs = set()
    for band in range(b):
        buckets = {}

        for product_ID, signature in signature_matrix.items():
            # Extract the band from the signature
            start_index = band * r
            end_index = start_index + r
            band_signature = tuple(signature[start_index:end_index])

            # Hash the band signature to get a bucket key
            bucket_key = hash_LSH(band_signature)

            buckets.setdefault(bucket_key, []).append(product_ID)
        
        # Now, we have the buckets for the current band.
        # We find all candidate pairs within each bucket.
        for buckets_product_IDs in buckets.values():
            if len(buckets_product_IDs) > 1:
                # More than one product in the same bucket, so they are candidate pairs.
                for i in range(len(buckets_product_IDs)):
                    for j in range(i + 1, len(buckets_product_IDs)):
                        pair = tuple(sorted((buckets_product_IDs[i], buckets_product_IDs[j])))
                        cand_pairs.add(pair)
    
    return cand_pairs

def jaccard_similarity(set1, set2):
    """
    Compute the Jaccard similarity between two sets.
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0.0
    return intersection / union

def compute_jaccard_for_cand_pairs(cand_pairs, product_features, t):
    """
    Compute the Jaccard similarity for each candidate pair.
    """
    pred_duplicates = set()
    for product_ID_1, product_ID_2 in cand_pairs:
        feature_1 = product_features[product_ID_1]
        feature_2 = product_features[product_ID_2]
        sim = jaccard_similarity(feature_1, feature_2)
        if sim >= t:
            pair = tuple(sorted((product_ID_1, product_ID_2)))
            pred_duplicates.add(pair)

    return pred_duplicates

def calculate_f_1_score(TP, FP, FN):
    denom  = 2 * TP + FP + FN
    if denom == 0:
        return 0.0
    
    return (2 * TP) / denom

def calculate_performance(true_duplicates, pred_duplicates, cand_pairs, num_products):
    """
    Calculate perfomance metrics: pair quality, pair completeness, F*-score, fraction of comparisons, TP, FP, FN, TN.
    """
    total_num_pairs = num_products * (num_products - 1) // 2
    num_cand_pairs = len(cand_pairs)
    num_true_duplicates = len(true_duplicates)

    TP_set = true_duplicates.intersection(pred_duplicates)
    TP = len(TP_set)
    FP_set = pred_duplicates.difference(true_duplicates)
    FP = len(FP_set)
    FN_set = true_duplicates.difference(pred_duplicates)
    FN = len(FN_set)
    TN = total_num_pairs - TP - FP - FN

    pair_quality = TP / num_cand_pairs if num_cand_pairs > 0 else 0.0
    pair_completeness = TP / num_true_duplicates if num_true_duplicates > 0 else 0.0
    F_star = (2 * pair_quality * pair_completeness) / (pair_quality + pair_completeness) if (pair_quality + pair_completeness) > 0 else 0.0
    fraction_of_comparisons = num_cand_pairs / total_num_pairs if total_num_pairs > 0 else 0.0

    return TP, FP, FN, TN, pair_quality, pair_completeness, F_star, fraction_of_comparisons


#%% Load data
with open('TVs-all-merged.json', 'r', encoding = "utf-8") as f:
    data = json.load(f)      # dict: modelID -> list of shop entries

# Convert JSON data to DataFrame
records = []
for modelID, listings in data.items():
    for item in listings:
        # Extract fields
        shop = item.get("shop")
        title = item.get("title")
        url  = item.get("url")
        features = item.get("featuresMap", {})

        # Combine everything into one dictionary (flattened row)
        row = {
            "model_ID": modelID,
            "title": title,
            "shop": shop,
            "url": url
        }
        row.update(features)  # Add features to the row
        records.append(row)

# Create a pandas DataFrame
df = pd.DataFrame(records)

# Add a unique product ID for each row
df.insert(0, 'product_ID', range(1, len(df) + 1))

# We only take into account the 'title' column to find the duplicates.
data = df[['product_ID', 'model_ID', 'title']].copy()
#print(data.shape)

# The notation of the titles is not consistent. Therefore, we need to standardise the notation of the different specificants in the titles.
# For example, "Samsung 55 inch 4k TV" and "Samsung 55" 4K Television" refer to the same product but are written differently.
# Using the function defined above, we standardise the notation of specifications like 'inch', 'hertz', or 'television' in the titles.
# Also, we remove the website names and double or trailing spaces.

data_clean = standardise_title(data)

#%% Find true duplicates
true_duplicates = set()
for model_ID, group in data_clean.groupby('model_ID'):
    product_IDs = group['product_ID'].tolist()
    if len(product_IDs) > 1:
        for i in range(len(product_IDs)):
            for j in range(i + 1, len(product_IDs)):
                pair = tuple(sorted((product_IDs[i], product_IDs[j])))
                true_duplicates.add(pair)

print(f"Number of true duplicate pairs: {len(true_duplicates)}")

# Now, we create character shingles of size k  and model words for each title in the cleaned dataset.
mw_stopwords = {"720p","1080p","4k","3d","60hz","120hz","240hz","600hz"}

product_model_words = model_words_per_product(data_clean)
all_unique_model_words = find_all_unique_model_words(product_model_words)
print(f"Number of unique model words: {len(all_unique_model_words)}")

k = 3    # Size of the character shingles
product_shingles = shingles_per_product(data_clean, k = k)
all_unique_shingles = find_all_unique_shingles(product_shingles)
print(f"Number of unique {k}-shingles: {len(all_unique_shingles)}")

# We give each unique model word and shingle an index.
all_unique_features = all_unique_model_words.union(all_unique_shingles)
feature_ID = {feature: idx for idx, feature in enumerate(all_unique_features)}
print(f"Total number of unique features (shingles + model words): {len(feature_ID)}")

# We create a dictionary that maps each product ID to the set of model word IDs.
product_feature_IDs = {}
for pid in data_clean['product_ID']:
    model_words = product_model_words.get(pid, set())
    shingles = product_shingles.get(pid, set())
    features = model_words.union(shingles)
    product_feature_IDs[pid] = {feature_ID[feature] for feature in features}

n_hashes = 100    # We define the hash functions for MinHashing.
P = 2**31 - 1                   # A large prime number that is larger than the number of unique model words.
hash_funcs = []
np.random.seed(27)              # Set seed for reproducibility
for _ in range(n_hashes):
    a_1 = np.random.randint(1, P)
    a_2 = np.random.randint(0, P)

    def hash_function(x, a_1 = a_1, a_2 = a_2, P = P):
        """
        A simple hash function of the form: h(x) = (a_1 * x + a_2) mod P
        where P is a large prime number that is larger than the number of unique shingles.
        """
        return (a_1 * x + a_2) % P

    hash_funcs.append(hash_function)

# Now, we create the signature matrix using the MinHashing technique.
signature_matrix = create_signature_matrix(product_feature_IDs, hash_funcs)
#print(list(signature_matrix.items())[:5])  # Print first 5 entries of the signature matrix

# We set the number of rows per band r and the number of bands b.       
r = 5                   # Rows per band
b = n_hashes // r       # Number of bands

B = 1000        # Number of bootstraps
b_frac = 0.63   # Fraction of products to sample in each bootstrap
results_per_t = {t: [] for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}
for bootstrap in range(B):
    sample_size = int(b_frac * len(data_clean))
    boot_ids = np.random.choice(data_clean['product_ID'], size = sample_size, replace = True)
    num_boot_products = len(boot_ids)

    boot_signature_matrix = {pid: signature_matrix[pid] for pid in set(boot_ids)}
    boot_mws = {pid: product_feature_IDs[pid] for pid in set(boot_ids)}

    boot_true_duplicates = {
        pair for pair in true_duplicates if pair[0] in boot_ids and pair[1] in boot_ids
    }

    cand_pairs_boot = find_cand_pairs(b, r, boot_signature_matrix)

    for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        pred_duplicates_boot = compute_jaccard_for_cand_pairs(cand_pairs_boot, boot_mws, t)

        TP, FP, FN, TN, pair_quality, pair_completeness, F_star, fraction_of_comparisons = calculate_performance(
            boot_true_duplicates, pred_duplicates_boot, cand_pairs_boot, num_boot_products
        )

        F_1_boot = calculate_f_1_score(TP, FP, FN)

        results_per_t[t].append(F_1_boot)

# After all bootstraps, we compute the average results for each threshold t.
for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    F_1_vals = np.array(results_per_t[t])
    mean_F_1 = np.mean(F_1_vals)
    std_F_1 = np.std(F_1_vals)
    CI_lb, CI_ub = np.quantile(F_1_vals, [0.025, 0.975])

    print(f"Bootstrap results for n_hashes = {n_hashes}, b = {b}, r = {r}, t = {t}:")
    print(f"Mean F1-score: {mean_F_1:.4f}")
    print(f"Standard Deviation of F1-score: {std_F_1:.4f}")
    print(f"95% Confidence Interval for F1-score: ({CI_lb:.4f}, {CI_ub:.4f})")
    print("--------------------------------------------------")

print("--------------------------------------------------")

# Finally, we use Locality Sensitive Hashing (LSH) to find candidate pairs of duplicate products.
cand_pairs = find_cand_pairs(b, r, signature_matrix)
print(f"Number of candidate pairs found: {len(cand_pairs)}")
#print(list(cand_pairs)[:5])  # Print first 5 candidate pairs

for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    # We compute the Jaccard similarity for each candidate pair and determine the predicted duplicates.
    pred_duplicates = compute_jaccard_for_cand_pairs(cand_pairs, product_feature_IDs, t)
    print(f"Number of predicted duplicate pairs: {len(pred_duplicates)}")

    # Calculate performance
    TP, FP, FN, TN, pair_quality, pair_completeness, F_star, fraction_of_comparisons = calculate_performance(true_duplicates, pred_duplicates, cand_pairs, len(data_clean))
    print(f"Results for n_hashes = {n_hashes}, b = {b}, r = {r}, t = {t}:")
    print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
    print(f"Pair Quality: {pair_quality:.4f}")
    print(f"Pair Completeness: {pair_completeness:.4f}")
    print(f"F*-score: {F_star:.4f}")
    print(f"Fraction of Comparisons: {fraction_of_comparisons:.6f}")
    print("--------------------------------------------------")
