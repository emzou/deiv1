# outtakes from supernova 
### **Number of Unique Threads vs. Reappearance Density**  
#The relationship between the number of unique threads a word appeared in and its reappearance density 
# remains strong, with **R² = 0.90 (uncapped) and R² = 0.92 (capped)**. While the correlation decreases 
# from **0.80 to 0.45** when moving from uncapped to capped words, no additional patterns emerge. 
# Importantly, unrecognized words do not significantly disrupt this relationship, reinforcing 
# that they follow similar behavioral trends as recognized words.

def thread_density_poly(known_political_words, x_col='unique_threads', y_col='normalized_reappearance_density_zscore'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # Scatter plot of recognized words (Left Plot)
    axes[0].scatter(known_political_words[x_col], known_political_words[y_col], 
                    color='#B2BEB5', alpha=0.8, label="Recognized Words")  
    # Annotate top words
    top_reappear = known_political_words.nlargest(10, y_col)
    top_spread = known_political_words.nlargest(10, x_col)
    top_words = pd.concat([top_reappear, top_spread]).drop_duplicates(subset=['word'])
    text_objects = []
    for _, row in top_words.iterrows():
        text_objects.append(axes[0].annotate(row['word'], 
                                             (row[x_col], row[y_col]), 
                                             fontsize=8, color='black', weight='bold'))
    adjust_text(text_objects, ax=axes[0], arrowprops=dict(arrowstyle="-", color="gray", lw=0.5)) 
    axes[0].set_xlabel("Number of Threads Word Appears In")
    axes[0].set_ylabel("Normalized Reappearance Density(Z-Score)")
    axes[0].set_title("Appearance compared to Reappearance Density of Recognized Words", loc='left')
    axes[0].legend()
    axes[0].grid(True, linestyle="--", linewidth=0.5)
    # Scatter plot with Polynomial Regression overlay (Right Plot)
    X = known_political_words[[x_col]].values.reshape(-1, 1)
    y = known_political_words[y_col].values
    poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    poly_model.fit(X, y)
    X_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)  # Ensure smooth curve
    y_pred_poly = poly_model.predict(X_range)
    r2 = r2_score(y, poly_model.predict(X))  # Compute R² for polynomial regression
    axes[1].scatter(known_political_words[x_col], known_political_words[y_col], 
                    color='#B2BEB5', alpha=0.8, label="Recognized Words")
    axes[1].plot(X_range, y_pred_poly, color='#da4f4a', linewidth=2, 
                 label=f'Polynomial Fit (R² = {r2:.2f})')
    # Identify outliers based on regression residuals
    residuals = np.abs(y - poly_model.predict(X))
    residual_threshold = np.percentile(residuals, 96)  # Higher threshold (98th percentile) for stricter outlier detection
    regression_outliers = known_political_words[residuals > residual_threshold]
    regression_outliers = regression_outliers.nlargest(15, y_col)  # Keep only top 15 outliers
    axes[1].scatter(regression_outliers[x_col], regression_outliers[y_col], color='#71797E', alpha=0.8, label="Regression Outliers")
    # Annotate top 15 outliers
    text_objects = []
    for _, row in regression_outliers.iterrows():
        text_objects.append(axes[1].annotate(row['word'], 
                                             (row[x_col], row[y_col]), 
                                             fontsize=8, color='black', weight='bold'))
    adjust_text(text_objects, ax=axes[1], arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))
    axes[1].set_xlabel("Number of Threads Word Appears In")
    axes[1].set_ylabel("Normalized Reappearance Density (Z-Score)")
    axes[1].set_title("Recognized Words with Polynomial Fit", loc='left')
    axes[1].legend()
    axes[1].grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

    thread_density_poly(bigger_known_words)
    thread_density_poly(the_known_words)

    def thread_density_compare(gecko, known_political_words, x_col='unique_threads', y_col='normalized_reappearance_density_zscore'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # === Left plot: ONLY Unrecognized Words ===
    axes[0].scatter(gecko[x_col], gecko[y_col], color='#428bca', alpha=0.6, label="Unrecognized Words")  # Blue dots only

    # Annotate top unrecognized words
    top_density_unrec = gecko.nlargest(10, y_col)
    top_threads_unrec = gecko.nlargest(10, x_col)
    top_words_unrec = pd.concat([top_density_unrec, top_threads_unrec]).drop_duplicates(subset=['word'])
    
    for _, row in top_words_unrec.iterrows():
        text = axes[0].annotate(row['word'], 
                                (row[x_col], row[y_col] + 0.1),  
                                fontsize=8, color='black', ha='center', weight='bold')
        text.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground="white"), path_effects.Normal()])

    axes[0].set_xlabel("Number of Threads Word Appears In")
    axes[0].set_ylabel("Normalized Reappearance Density")
    axes[0].set_title("Unrecognized Words: Thread Appearance vs. Reappearance Density", loc='left')
    axes[0].legend()
    axes[0].grid(True, linestyle="--", linewidth=0.5)

    # Capture axis limits from the first plot BEFORE creating the second plot
    x_limits = axes[0].get_xlim()
    y_limits = axes[0].get_ylim()

    # === Right plot: Recognized words FIRST, then overlay Unrecognized words ===
    axes[1].scatter(known_political_words[x_col], known_political_words[y_col], color='#B2BEB5', alpha=0.6, label="Recognized Words")  # Grey dots (background)
    axes[1].scatter(gecko[x_col], gecko[y_col], color='#428bca', alpha=0.6, label="Unrecognized Words")  # Blue dots (foreground)

    # Polynomial Regression Fit (Recognized Words Only)
    X_known = known_political_words[[x_col]].values.reshape(-1, 1)
    y_known = known_political_words[y_col].values
    poly_model_known = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    poly_model_known.fit(X_known, y_known)
    X_range = np.linspace(X_known.min(), X_known.max(), 300).reshape(-1, 1)
    y_pred_poly_known = poly_model_known.predict(X_range)
    r2_known = r2_score(y_known, poly_model_known.predict(X_known))  # Compute R² for known words

    # Polynomial Regression Fit (Recognized + Unrecognized Words)
    combined_data = pd.concat([known_political_words, gecko])
    X_combined = combined_data[[x_col]].values.reshape(-1, 1)
    y_combined = combined_data[y_col].values
    poly_model_combined = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    poly_model_combined.fit(X_combined, y_combined)
    y_pred_poly_combined = poly_model_combined.predict(X_range)
    r2_combined = r2_score(y_combined, poly_model_combined.predict(X_combined))  # Compute R² for combined words

    # Regression line for recognized words only
    axes[1].plot(X_range, y_pred_poly_known, color='#da4f4a', linewidth=2, label=f'Polynomial Fit (R² = {r2_known:.2f}, Recognized)')

    # Regression line for combined words
    axes[1].plot(X_range, y_pred_poly_combined, color='#2ca02c', linewidth=2, linestyle='dashed', label=f'Polynomial Fit (R² = {r2_combined:.2f}, Combined)')

    # Apply the same x and y limits from the first plot
    axes[1].set_xlim(x_limits)
    axes[1].set_ylim(y_limits)
    axes[1].set_xlabel("Number of Threads Word Appears In")
    axes[1].set_ylabel("Normalized Reappearance Density")
    axes[1].set_title("Unrecognized and Recognized Words: Thread Appearance vs. Reappearance Density", loc='left')
    axes[1].legend()
    axes[1].grid(True, linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.show()

    thread_density_compare(gecko, the_known_words)

### **Reappearance Rate vs. Reappearance Density**  
# Since we find (and confirm) that number of threads and reappearance density are highly correlated,
# we expect to find the same results between 1) number of threads and reappearance rate and 2) 
# reappearance rate and reappearance density. 

def plot_reappearance_rate_vs_reappearance_density(known_political_words, x_col='normalized_reappearance_rate_zscore', y_col='normalized_reappearance_density_zscore'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # Scatter plot of recognized words (Left Plot)
    axes[0].scatter(known_political_words[x_col], known_political_words[y_col], 
                    color='#B2BEB5', alpha=0.8, label="Recognized Words")
    # Annotate top words
    #top_reappearance = known_political_words.nlargest(10, y_col)
    top_density = known_political_words.nlargest(10, x_col)
    top_words = pd.concat([top_density]).drop_duplicates(subset=['word'])
    text_objects = []
    for _, row in top_words.iterrows():
        text_objects.append(axes[0].annotate(row['word'], 
                                             (row[x_col], row[y_col]), 
                                             fontsize=8, color='black', weight='bold'))
    
    adjust_text(text_objects, ax=axes[0], arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))
    axes[0].set_xlabel("Normalized Reappearance Rate (Z-Score)")
    axes[0].set_ylabel("Normalized Reappearance Density (Z-Score)")
    axes[0].set_title("Recognized Words: Reappearance Density vs. Reappearance Rate", loc='left')
    axes[0].legend()
    axes[0].grid(True, linestyle="--", linewidth=0.5)
    X = known_political_words[[x_col]].values.reshape(-1, 1)
    y = known_political_words[y_col].values
    poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    poly_model.fit(X, y)
    X_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    y_pred_poly = poly_model.predict(X_range)
    r2 = r2_score(y, poly_model.predict(X))  # Compute R² for polynomial regression
    axes[1].scatter(known_political_words[x_col], known_political_words[y_col], 
                    color='#B2BEB5', alpha=0.8, label="Recognized Words")
    axes[1].plot(X_range, y_pred_poly, color='#da4f4a', linewidth=2, 
                 label=f'Polynomial Fit (R² = {r2:.2f})')
    # Identify outliers based on the largest distance from the polynomial regression line
    y_pred_actual = poly_model.predict(X)
    distances = np.abs(y - y_pred_actual)  # Absolute distance from the expected value
    known_political_words['distance_from_fit'] = distances  # Store distances in DataFrame
    regression_outliers = known_political_words.nlargest(6, 'distance_from_fit')  # Select top 15 outliers based on distance
    axes[1].scatter(regression_outliers[x_col], regression_outliers[y_col], color='#71797E', alpha=0.8, label="Regression Outliers")
    # Annotate outliers
    text_objects = []
    for _, row in regression_outliers.iterrows():
        text_objects.append(axes[1].annotate(row['word'], 
                                             (row[x_col], row[y_col]), 
                                             fontsize=8, color='black', weight='bold'))
    adjust_text(text_objects, ax=axes[1], arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))
    axes[1].set_xlabel("Normalized Reappearance Rate (Z-Score)")
    axes[1].set_ylabel("Normalized Reappearance Density (Z-Score)")
    axes[1].set_title("Recognized Words with Polynomial Fit", loc='left')
    axes[1].legend()
    axes[1].grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

    plot_reappearance_rate_vs_reappearance_density(bigger_known_words)
    plot_reappearance_rate_vs_reappearance_density(the_known_words)

    def ratedens_plot_and_quantify_relationship(df, 
                                   x_col='normalized_reappearance_rate_zscore', 
                                   y_col='normalized_reappearance_density_zscore', 
                                   color_col='normalized_time_entropy_zscore'):
    x = df[x_col].values.reshape(-1, 1)
    y = df[y_col].values.reshape(-1, 1)
    z = df[color_col].values.reshape(-1, 1)

    mask = df[color_col] != 1
    color_min, color_max = -5, 1  # Explicitly setting the range from -5 to 1

    plt.figure(figsize=(8, 6))
    plt.scatter(df[x_col][~mask], df[y_col][~mask], c='red', alpha=0.5, s=50)

    # **Pass raw `z` values and use `vmin` and `vmax` directly**
    scatter = plt.scatter(df[x_col][mask], df[y_col][mask], c=df[color_col][mask], 
                          cmap='coolwarm', alpha=0.9, s=50, vmin=color_min, vmax=color_max)

    cbar = plt.colorbar(scatter)
    cbar.set_label("Time Entropy (Red = 1, Blue = -5)")
    cbar.set_ticks(np.linspace(-5, 1, num=7))  # Explicit tick marks from -5 to 1

    plt.xlabel("Normalized Reappearance Rate Z-Score", fontsize=9)
    plt.ylabel("Normalized Reappearance Density Z-Score", fontsize=9)
    plt.title("Rates vs. Density with Time Entropy", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

    ratedens_plot_and_quantify_relationship(the_known_words)

    def compare_reappear_density(gecko, known_political_words, x_col='normalized_reappearance_rate_zscore', y_col='normalized_reappearance_density_zscore'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ### Left Plot: Unrecognized Words (Gecko)
    axes[0].scatter(gecko[x_col], gecko[y_col], color='#428bca', alpha=0.6, label="Unrecognized Words")
    # Annotate top words for unrecognized words
    top_reappearance_unrec = gecko.nlargest(10, y_col)
    top_density_unrec = gecko.nlargest(10, x_col)
    top_words_unrec = pd.concat([top_reappearance_unrec, top_density_unrec]).drop_duplicates(subset=['word'])
    text_objects = []
    for _, row in top_words_unrec.iterrows():
        text = axes[0].annotate(row['word'], 
                                (row[x_col], row[y_col] + 0.1),  # Small vertical offset
                                fontsize=8, color='black', ha='center', weight='bold')
        # Add outline effect
        text.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground="white"), 
                               path_effects.Normal()])
        text_objects.append(text)
    adjust_text(text_objects, ax=axes[0], arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))
    axes[0].set_xlabel("Normalized Reappearance Rate (Z-Score)")
    axes[0].set_ylabel("Normalized Reappearance Density (Z-Score)")
    axes[0].set_title("Unrecognized Words: Reappearance Density vs. Reappearance Rate", loc='left')
    axes[0].legend()
    axes[0].grid(True, linestyle="--", linewidth=0.5)
    ### Capture the axis limits from the first plot
    x_limits = axes[0].get_xlim()
    y_limits = axes[0].get_ylim()
    ### Right Plot: Recognized Words (Known Political Words)
    axes[1].scatter(gecko[x_col], gecko[y_col], color='#428bca', alpha=0.6, label="Unrecognized Words")
    axes[1].scatter(known_political_words[x_col], known_political_words[y_col], color='#B2BEB5', alpha=0.6, label="Recognized Words")
    # Polynomial Regression for Recognized Words
    X = known_political_words[[x_col]].values.reshape(-1, 1)
    y = known_political_words[y_col].values
    poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    poly_model.fit(X, y)
    X_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    y_pred_poly = poly_model.predict(X_range)
    r2 = r2_score(y, poly_model.predict(X))  # Compute R² for polynomial regression
    axes[1].plot(X_range, y_pred_poly, color='#da4f4a', linewidth=2, 
                 label=f'Polynomial Fit (R² = {r2:.2f})')
    # Identify regression outliers
    y_pred_actual = poly_model.predict(X)
    distances = np.abs(y - y_pred_actual)  # Absolute distance from the expected value
    known_political_words['distance_from_fit'] = distances  # Store distances in DataFrame
    regression_outliers = known_political_words.nlargest(9, 'distance_from_fit')  # Select top 15 outliers based on distance
    axes[1].scatter(regression_outliers[x_col], regression_outliers[y_col], color='#71797E', alpha=0.8, label="Regression Outliers")
    # Annotate outliers
    text_objects = []
    for _, row in regression_outliers.iterrows():
        text = axes[1].annotate(row['word'], 
                                (row[x_col], row[y_col] + 0.1),  # Small vertical offset
                                fontsize=8, color='black', ha='center', weight='bold')
        # Add outline for better readability
        text.set_path_effects([path_effects.Stroke(linewidth=1.5, foreground="white"), 
                               path_effects.Normal()])
        text_objects.append(text)
    adjust_text(text_objects, ax=axes[1], arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))
    # Apply the same x and y limits from the first plot
    axes[1].set_xlim(x_limits)
    axes[1].set_ylim(y_limits)
    axes[1].set_xlabel("Normalized Reappearance Rate (Z-Score)")
    axes[1].set_ylabel("Normalized Reappearance Density (Z-Score)")
    axes[1].set_title("Recognized Words with Polynomial Fit", loc='left')
    axes[1].legend()
    axes[1].grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()
    compare_reappear_density(gecko, the_known_words)

metrics = ["normalized_unique_threads_zscore", "normalized_reappearance_rate_zscore", "normalized_threadwise_density_zscore", "normalized_repeat_ratio_zscore","normalized_time_entropy_zscore" ]
# metrics we've dropped: 'first_response_delay', 'time stickiness (same as thread density)'
corr_matrix = bigger_known_words[metrics].corr()
renamed_labels = [col.replace("normalized_", "").replace("_zscore", "").replace("_", " ").title() for col in metrics]
corr_matrix.columns = renamed_labels
corr_matrix.index = renamed_labels
#corr_matrix = the_known_words[metrics].corr()
#renamed_labels = [col.replace("normalized_", "").replace("_zscore", "").replace("_", " ").title() for col in metrics]
#corr_matrix.columns = renamed_labels
#corr_matrix.index = renamed_labels
