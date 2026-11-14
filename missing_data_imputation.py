# %% [markdown]
# # Missing Data Imputation on the Bike Sharing Dataset

# %% [markdown]
# ## 1. Setup and Data Loading

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"D:\University\Machine Learning Sources\Missing Imputation\Data\day.csv", parse_dates=["dteday"])
print("rows and columns:", df.shape)
df.head()

# %% [markdown]
# ## 2. Exploratory Data Analysis (EDA)

# %% [markdown]
# ### 2.1 Data structure:

# %%
df.info()
print("-" * 40)
df.describe().T

# %% [markdown]
# ### 2.2 Data quality

# %%
cols = ["temp", "atemp", "hum", "windspeed"]
print((df[cols] == 0).sum())
bad = df[(df[cols] == 0).any(axis=1)]
display(bad[["dteday"] + cols])

# %%
import numpy as np

df.loc[df["hum"] == 0, "hum"] = np.nan 
print(df[["temp","atemp","hum", "windspeed"]].isna().sum())

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

num_cols = ["temp", "atemp", "hum", "windspeed"]
cat_cols = ["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit"]
target_col = "cnt"

# %%
print("duplicate rows:", df.duplicated().sum())
print("duplicate date", df["dteday"].duplicated().sum())
print("missing values:", df.isna().sum().sum())

# %%
full = pd.date_range(start=df["dteday"].min(), end=df["dteday"].max(), freq="D")
diff = full.difference(df["dteday"])
print("missing dates:", len(diff))

# %%
print("cnt != casual + registered:", (df["cnt"] != df["casual"] + df["registered"]).sum())

# %% [markdown]
# ### 2.3 Univariate analysis

# %%
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for i, c in enumerate(num_cols):
    sns.histplot(df[c], kde=True, ax=axes[0, i])
    sns.boxplot(x=df[c], ax=axes[1, i])
plt.tight_layout(); plt.show()

# %%
print(df[num_cols + [target_col]].skew())

# %% [markdown]
# ### 2.4 Outlier detection

# %%
for c in num_cols + [target_col]:
    q1, q3 = df[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n = ((df[c] < low) | (df[c] > high)).sum()
    print(f"{c:10s} outlier: {n:3d}   (lower {low:.3f}، higher {high:.3f})")

# %% [markdown]
# ### 2.5 Bivariate and multivariate analysis

# %%
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, c in zip(axes, num_cols):
    sns.scatterplot(x=df[c], y=df[target_col], ax=ax, alpha=0.5)

# %%
corr = df[num_cols + ["casual", "registered", target_col]].corr()
print(corr)

# %%
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)

# %%
sns.pairplot(df[num_cols + [target_col]], corner=True)

# %% [markdown]
# ### 2.6 Time trend and target variable

# %%
sns.lineplot(x=df["dteday"], y=df[target_col], color="blue", label="cnt")

# %%
fig, axes = plt.subplots(2, 1, figsize=(14, 11))
sns.histplot(df[target_col], kde=True, ax=axes[0])
sns.boxplot(x="yr", y=target_col, data=df, ax=axes[1])


# %%
print(df[target_col].describe())

# %% [markdown]
# ### 2.7 EDA conclusions

# %%
df.nsmallest(5, "cnt")[["dteday", "weathersit", "windspeed", "cnt"]]

# %% [markdown]
# ## 3. Simulating Missing Data

# %%
main_data = df[cols].copy()

# %% [markdown]
# ### 3.1 Design

# %%
def make_missing(main_data, cols, mechanism, driver = "atemp", p_high = 0.30, p_low = 0.05, p_mcar = 0.20, seed = 0):

    rng = np.random.default_rng(seed)
    miss = pd.DataFrame(False, index=main_data.index, columns=main_data.columns)

    for col in cols:
        if mechanism == "MCAR":
            p = np.full(len(main_data), p_mcar)
        elif mechanism == "MAR":
            cold = main_data[driver] < main_data[driver].median()
            p = np.where(cold , p_high, p_low)
        elif mechanism == "MNAR":
            high = main_data[col] > main_data[col].median()
            p = np.where(high, p_high, p_low)
        else:
            raise ValueError("Invalid mechanism. Choose from 'MCAR', 'MAR', or 'MNAR'.")
        miss[col] = rng.random(len(main_data)) < p 

    return main_data.mask(miss) , miss



# %% [markdown]
# ### 3.2 Generating the three datasets

# %%
miss_cols = ["temp", "hum", "windspeed"]
data = {}
miss_model = {}
for k, mech in enumerate(["MCAR", "MAR", "MNAR"]):
    data[mech], miss_model[mech] = make_missing(main_data, miss_cols, mech, seed=k)

pd.DataFrame({m : data[m].isna().mean() * 100 for m in data}).round(1)

# %%
data["MAR"].head(10)

# %% [markdown]
# ### 3.3 Sanity checks

# %%
rows = {m : f"{data[m].dropna().shape[0]} from {len(main_data)} ({data[m].dropna().shape[0] / len(main_data):.0%})" for m in data}
pd.Series(rows, name="Complete Rows")

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, mech in zip(axes, data):
    ax.imshow(miss_model[mech].iloc[:100].values, aspect="auto", cmap="Greys", interpolation="nearest")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45)
    ax.set_title(mech)
    ax.set_ylabel("day")
    ax.grid(False)
plt.tight_layout(); plt.show()

# %%
row = []
for mech in data:
    for col in miss_cols:
        observed_mean = data[mech][col].mean()
        main_mean = main_data[col].mean()
        row.append({"mechanism" : mech , "column" : col,
                    "main_mean" : main_mean, "observed mean" : observed_mean,
                    "error (in sd)" : (observed_mean - main_mean) / main_data[col].std()})
bias = pd.DataFrame(row)
bias.pivot(index="column", columns="mechanism", values="error (in sd)").round(3)

# %% [markdown]
# ## 4. Exploring Simple Methods on temp (MCAR)

# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mech = "MCAR"
target = "temp"
predicator = ["atemp"]

df_miss = data[mech].copy()

miss_target = miss_model["MCAR"]["temp"]

print("Dataset shape:", df_miss.shape)
print("Missing temp:", miss_target.sum())



# %% [markdown]
# ### 4.1 Regression imputation with atemp

# %%
train_mask = ~miss_target

x_train = df_miss.loc[train_mask , predicator]
y_train = df_miss.loc[train_mask, target]

model = LinearRegression()
model.fit(x_train, y_train)

print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)

# %%
X_missing = df_miss.loc[miss_target, predicator]
predicted_values = model.predict(X_missing)

df_regression = df_miss.copy()

df_regression.loc[miss_target, target] = predicted_values
print("Missing before:", df_miss[target].isna().sum())
print("Missing after:", df_regression[target].isna().sum())

print(df_regression.loc[miss_target, ["atemp", "temp"]].head(10))

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

main_y = main_data.loc[miss_target , target]
pred_y = df_regression.loc[miss_target, target]

mae = mean_absolute_error(main_y, pred_y)
rmse = np.sqrt(mean_squared_error(main_y, pred_y))

print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))

# %% [markdown]
# ### 4.2 Mean and median imputation

# %%
from sklearn.impute import SimpleImputer
mean_imputer = SimpleImputer(strategy="mean")
df_mean = df_miss.copy()
X = df_mean[[target]]
X_filled = mean_imputer.fit_transform(X)
df_mean[[target]] = X_filled
y_true = main_data.loc[miss_target, target]
y_pred_mean = df_mean.loc[miss_target, target]
mae_mean = mean_absolute_error(y_true, y_pred_mean)
rmse_mean = np.sqrt(mean_squared_error(y_true, y_pred_mean))

print("Mean MAE:", round(mae_mean, 4))
print("Mean RMSE:", round(rmse_mean, 4))

# %%
from sklearn.impute import SimpleImputer
median_imputer = SimpleImputer(strategy = "median")
df_median = df_miss.copy()
y = df_median[[target]]
y_filled =  median_imputer.fit_transform(y)
df_median[[target]] = y_filled
z_true = main_data.loc[miss_target, target]
z_pred_median = df_median.loc[miss_target, target]
mae_median = mean_absolute_error(z_true, z_pred_median)
rmse_median = np.sqrt(mean_squared_error(z_true, z_pred_median))

print("Median MAE:", round(mae_median, 4))
print("Median RMSE:", round(rmse_median, 4))

# %% [markdown]
# ### 4.3 KNN imputation and choosing k

# %%
from sklearn.impute import KNNImputer

knn_cols = ["temp" , "atemp"]
df_knn = df_miss.copy()

KNN_Imputer = KNNImputer(n_neighbors = 5) 
df_knn[knn_cols] = KNN_Imputer.fit_transform(df_knn[knn_cols])

y_true = main_data.loc[miss_target , target]
y_pred_knn = df_knn.loc[miss_target, target]

mae_knn = mean_absolute_error(y_true, y_pred_knn)
rmse_knn = np.sqrt(mean_squared_error(y_true, y_pred_knn))

print("KNN MAE:", round(mae_knn, 4))
print("KNN RMSE:", round(rmse_knn, 4))

# %%
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

k_values = [1, 3, 5, 7, 9, 15]

knn_result = []
knn_cols = ["temp" , "atemp"]
y_true = main_data.loc[miss_target, target]

for k in k_values:
    knn_imputer = KNNImputer(n_neighbors = k)
    df_temp = df_miss.copy()
    df_temp[knn_cols] = knn_imputer.fit_transform(df_temp[knn_cols]) 
    y_pred = df_temp.loc[miss_target, target]

    mae_k = mean_absolute_error(y_true, y_pred)
    rmse_k = np.sqrt(mean_squared_error(y_true, y_pred))

    knn_result.append({
        "K": k,
        "MAE": mae_k,
        "RMSE": rmse_k
    })

knn_table = pd.DataFrame(knn_result)
print("KNN Comparison:")
display(knn_table.round(4))

best_row = knn_table.loc[knn_table["RMSE"].idxmin()]
best_k = int(best_row["K"])
print("-----------------------")
print("Best K:", best_k)
print("Best MAE:", round(best_row["MAE"], 4))
print("Best RMSE:", round(best_row["RMSE"], 4))

# %% [markdown]
# ## 5. Main Methods under MCAR

# %% [markdown]
# ### 5.1 IterativeImputer (MICE)

# %%
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

mech = "MCAR"
iter_cols = ["atemp", "temp", "hum", "windspeed"]
df_iter = df_miss.copy()

iter_imputer = IterativeImputer(
    estimator = BayesianRidge(),
    max_iter = 10,
    initial_strategy = "mean",
    random_state = 40
)

df_iter[iter_cols] = iter_imputer.fit_transform(df_iter[iter_cols])

print("Missing before:")
print(data[mech][iter_cols].isna().sum())

print("\nMissing after:")
print(df_iter[iter_cols].isna().sum())

print("\nIterations used:", iter_imputer.n_iter_)


# %%
iter_result = []

for col in ["temp", "hum", "windspeed"]:
    mask_col = miss_model[mech][col]
    y_true = main_data.loc[mask_col, col]
    y_pred_mice = df_iter.loc[mask_col, col]

    mae_col = mean_absolute_error(y_true, y_pred_mice)
    rmse_col = np.sqrt(mean_squared_error(y_true, y_pred_mice))

    iter_result.append({
        "Method": "IterativeImputer",
        "Mechanism": mech,
        "Target": col,
        "MAE": mae_col,
        "RMSE": rmse_col
    })

iter_table = pd.DataFrame(iter_result)

display(iter_table.round(4))

# %% [markdown]
# ### 5.2 MissForest

# %%
import sklearn
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from missforest import MissForest

rf_regression = RandomForestRegressor (
    n_estimators = 100,
    random_state = 42,
    n_jobs = -1
)

rf_classifier = RandomForestClassifier(
    n_estimators = 100,
    random_state = 42,
    n_jobs = -1
)

mf = MissForest(
    clf = rf_classifier,
    rgr = rf_regression,
    categorical = []
)
print("MissForest model created successfully!")

# %%
mech = "MCAR"

mf_cols = ["atemp", "temp", "hum", "windspeed"]
df_mf = data[mech].copy()
x_mf = df_mf[mf_cols].copy()

print("Dataset shape:", x_mf.shape)
print("Missing values:")
print(x_mf.isna().sum())

# %%
df_mf_filled = mf.fit_transform(x= x_mf)
print("Missing values after imputation:")
print(df_mf_filled.isna().sum())

display(df_mf_filled.head())

# %%
mf_results = []


for col in ["temp", "hum", "windspeed"]:

    
    mask_col = miss_model[mech][col]

    y_true = main_data.loc[mask_col, col]


    y_pred = df_mf_filled.loc[mask_col, col]

    
    mae_mf = mean_absolute_error(y_true, y_pred)

    rmse_mf = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    
    mf_results.append({
        "Method": "MissForest",
        "Mechanism": mech,
        "Target": col,
        "MAE": mae_mf,
        "RMSE": rmse_mf
    })


mf_table = pd.DataFrame(mf_results)

display(mf_table.round(4))

# %%
mf_simple = MissForest(
    clf=rf_classifier,
    rgr=rf_regression,
    categorical=[]
)

simple_cols = ["atemp", "temp"]
X_simple = data[mech][simple_cols].copy()
df_simple = mf_simple.fit_transform(X_simple)

mask_temp = miss_model[mech]["temp"]

y_true = main_data.loc[mask_temp, "temp"]
y_pred = df_simple.loc[mask_temp, "temp"]

print("MAE:", round(mean_absolute_error(y_true, y_pred), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_true, y_pred)), 4))

# %%
tree_values = [50, 100, 200]
tree_results = []
mf_cols = ["atemp", "temp", "hum", "windspeed"]

X_mf = data["MCAR"][mf_cols].copy()

for n_tree in tree_values:
    rf_reg = RandomForestRegressor(
        n_estimators=n_tree,
        random_state=42,
        n_jobs=-1
    )

    rf_clf = RandomForestClassifier(
        n_estimators=n_tree,
        random_state=42,
        n_jobs=-1
    )
    mf_test = MissForest(
        clf=rf_clf,
        rgr=rf_reg,
        categorical=[]
    )
    
    df_filled = mf_test.fit_transform(
        X_mf.copy()
    )

    for col in ["temp", "hum", "windspeed"]:
        mask_col = miss_model["MCAR"][col]
        y_true = main_data.loc[mask_col, col]
        y_pred = df_filled.loc[mask_col, col]
        mae_value = mean_absolute_error(
            y_true, y_pred
        )

        rmse_value = np.sqrt(
            mean_squared_error(y_true, y_pred)
        )

        tree_results.append({
            "Trees": n_tree,
            "Target": col,
            "MAE": mae_value,
            "RMSE": rmse_value
        })


tree_table = pd.DataFrame(tree_results)

display(
    tree_table.round(4)
)

# %% [markdown]
# ### 5.3 Linear interpolation

# %%
interpol_cols = ["temp", "hum", "windspeed"]
mech = "MCAR"
df_interp = data["MCAR"].copy()
df_interp[interpol_cols] = df_interp[interpol_cols].interpolate( method="linear")
print("Missing after interpolation:")
print(df_interp[interpol_cols].isna().sum())


# %%
interp_result = []

for col in interpol_cols:
    mask_col = miss_model[mech][col]
    y_true = main_data.loc[mask_col,col]
    y_pred = df_interp.loc[mask_col,col]

    mae_value = mean_absolute_error(y_true, y_pred)
    rmse_value = np.sqrt(mean_squared_error(y_true, y_pred))
    interp_result.append({
        "Method": "Linear Interpolation",
        "Mechanism": "MCAR",
        "Target": col,
        "MAE": mae_value,
        "RMSE": rmse_value
    })

interp_table = pd.DataFrame(interp_result)

display(interp_table.round(4))

# %% [markdown]
# ### 5.4 MCAR comparison table

# %%
final_comparison = pd.concat(
    [
        iter_table,
        mf_table,
        interp_table
    ],
    ignore_index=True
)
final_comparison = final_comparison[
    ["Method", "Mechanism", "Target", "MAE", "RMSE"]
]
final_comparison = final_comparison.sort_values(
    by=["Target", "RMSE"]
).reset_index(drop=True)
display(final_comparison.round(4))

# %% [markdown]
# ## 6. Main Methods under MAR
# ### 6.1 IterativeImputer، 7.2 MissForest، 7.3 Linear interpolation، 7.4 MAR comparison table

# %%

mech = "MAR"
mar_cols = ["temp", "hum", "windspeed"]
print("Dataset shape:", data[mech].shape)
print("\nMissing values:")
print(data[mech][mar_cols].isna().sum())

print("\nMissing percentage:")
print(
    (data[mech][mar_cols].isna().mean() * 100).round(2)
)
print("\nMask counts:")
print(miss_model[mech][mar_cols].sum())

# %%
mech = "MAR"
iter_cols = ["atemp", "temp", "hum", "windspeed"]
df_iter_mar = data[mech].copy()
iter_imputer_mar = IterativeImputer(
    estimator=BayesianRidge(),
    max_iter=10,
    initial_strategy="mean",
    random_state=42
)
df_iter_mar[iter_cols] = iter_imputer_mar.fit_transform(
    df_iter_mar[iter_cols]
)

print("Missing after imputation:")
print(df_iter_mar[iter_cols].isna().sum())

print("Iterations used:", iter_imputer_mar.n_iter_)

# %%
iter_mar_results = []
for col in ["temp", "hum", "windspeed"]:
    mask_col = miss_model["MAR"][col]
    y_true = main_data.loc[mask_col, col]
    y_pred = df_iter_mar.loc[mask_col, col]
    mae_value = mean_absolute_error(y_true, y_pred)

    rmse_value = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    iter_mar_results.append({
        "Method": "IterativeImputer",
        "Mechanism": "MAR",
        "Target": col,
        "MAE": mae_value,
        "RMSE": rmse_value
    })


iter_mar_table = pd.DataFrame(iter_mar_results)

display(iter_mar_table.round(4))

# %%
mf_cols = ["atemp", "temp", "hum", "windspeed"]
X_mf_mar = data["MAR"][mf_cols].copy()
rf_reg_mar = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf_clf_mar = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
mf_mar = MissForest(
    clf=rf_clf_mar,
    rgr=rf_reg_mar,
    categorical=[]
)
df_mf_mar = mf_mar.fit_transform(X_mf_mar.copy())
print("Missing after imputation:")
print(df_mf_mar.isna().sum())

# %%
# ----------- MAE and RMSE for MissForest -----------
mf_mar_results = []

for col in ["temp", "hum", "windspeed"]:
    mask_col = miss_model["MAR"][col]
    y_true = main_data.loc[mask_col, col]
    y_pred = df_mf_mar.loc[mask_col, col]
    mae_value = mean_absolute_error(y_true, y_pred)

    rmse_value = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )
    mf_mar_results.append({
        "Method": "MissForest",
        "Mechanism": "MAR",
        "Target": col,
        "MAE": mae_value,
        "RMSE": rmse_value
    })

mf_mar_table = pd.DataFrame(mf_mar_results)

display(mf_mar_table.round(4))

# %%
df_interp_mar = data["MAR"].copy()
df_interp_mar["dteday"] = df.loc[
    df_interp_mar.index, "dteday"
]
df_interp_mar = df_interp_mar.sort_values("dteday")

interp_cols = ["temp", "hum", "windspeed"]

df_interp_mar[interp_cols] = df_interp_mar[
    interp_cols
].interpolate(method="linear")

print(df_interp_mar[interp_cols].isna().sum())

# %%
interp_mar_results = []

for col in ["temp", "hum", "windspeed"]:
    mask_col = miss_model["MAR"][col]
    y_true = main_data.loc[mask_col, col]
    y_pred = df_interp_mar.loc[mask_col, col]
    mae_value = mean_absolute_error(y_true, y_pred)
    rmse_value = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )
    interp_mar_results.append({
        "Method": "Linear Interpolation",
        "Mechanism": "MAR",
        "Target": col,
        "MAE": mae_value,
        "RMSE": rmse_value
    })

interp_mar_table = pd.DataFrame(interp_mar_results)

display(interp_mar_table.round(4))

# %%
mar_comparison = pd.concat(
    [
        iter_mar_table,
        mf_mar_table,
        interp_mar_table
    ],
    ignore_index=True
)

mar_comparison = mar_comparison.sort_values(
    by=["Target", "RMSE"]
).reset_index(drop=True)

display(mar_comparison.round(4))

# %% [markdown]
# ## 7. Main Methods under MNAR
# ### 7.1 IterativeImputer، 8.2 MissForest، 8.3 Linear interpolation، 8.4 MNAR comparison table

# %%

mech = "MNAR"
mnar_cols = ["temp", "hum", "windspeed"]
print("Dataset shape:", data[mech].shape)
print("\nMissing values:")
print(data[mech][mnar_cols].isna().sum())
print("\nMissing percentage:")
print(
    (data[mech][mnar_cols].isna().mean() * 100).round(2)
)
print("\nMask counts:")
print(miss_model[mech][mnar_cols].sum())

# %%
iter_cols = ["atemp", "temp", "hum", "windspeed"]
df_iter_mnar = data["MNAR"].copy()

iter_imputer_mnar = IterativeImputer(
    estimator=BayesianRidge(),
    max_iter=10,
    initial_strategy="mean",
    random_state=42
)

df_iter_mnar[iter_cols] = iter_imputer_mnar.fit_transform(
    df_iter_mnar[iter_cols]
)

print(df_iter_mnar[iter_cols].isna().sum())

print("Iterations:", iter_imputer_mnar.n_iter_)

# %%
iter_mnar_results = []
for col in ["temp", "hum", "windspeed"]:
    mask_col = miss_model["MNAR"][col]
    y_true = main_data.loc[mask_col, col]
    y_pred = df_iter_mnar.loc[mask_col, col]
    mae_value = mean_absolute_error(y_true, y_pred)
    rmse_value = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )
    iter_mnar_results.append({
        "Method": "IterativeImputer",
        "Mechanism": "MNAR",
        "Target": col,
        "MAE": mae_value,
        "RMSE": rmse_value
    })

iter_mnar_table = pd.DataFrame(iter_mnar_results)

display(iter_mnar_table.round(4))

# %%
mf_cols = ["atemp", "temp", "hum", "windspeed"]
X_mf_mnar = data["MNAR"][mf_cols].copy()
rf_reg_mnar = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf_clf_mnar = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
mf_mnar = MissForest(
    clf=rf_clf_mnar,
    rgr=rf_reg_mnar,
    categorical=[]
)

df_mf_mnar = mf_mnar.fit_transform(X_mf_mnar.copy())

print("Missing after imputation:")
print(df_mf_mnar.isna().sum())

# %%
mf_mnar_results = []

for col in ["temp", "hum", "windspeed"]:
    mask_col = miss_model["MNAR"][col]
    y_true = main_data.loc[mask_col, col]
    y_pred = df_mf_mnar.loc[mask_col, col]
    mae_value = mean_absolute_error(y_true, y_pred)
    rmse_value = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )
    mf_mnar_results.append({
        "Method": "MissForest",
        "Mechanism": "MNAR",
        "Target": col,
        "MAE": mae_value,
        "RMSE": rmse_value
    })

mf_mnar_table = pd.DataFrame(mf_mnar_results)

display(mf_mnar_table.round(4))

# %%
df_interp_mnar = data["MNAR"].copy()
df_interp_mnar["dteday"] = df.loc[
    df_interp_mnar.index, "dteday"
]
df_interp_mnar = df_interp_mnar.sort_values("dteday")
interp_cols = ["temp", "hum", "windspeed"]
df_interp_mnar[interp_cols] = df_interp_mnar[
    interp_cols
].interpolate(method="linear")

print(df_interp_mnar[interp_cols].isna().sum())

# %%
interp_mnar_results = []
for col in ["temp", "hum", "windspeed"]:
    mask_col = miss_model["MNAR"][col]
    y_true = main_data.loc[mask_col, col]
    y_pred = df_interp_mnar.loc[mask_col, col]
    mae_value = mean_absolute_error(y_true, y_pred)
    rmse_value = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )
    interp_mnar_results.append({
        "Method": "Linear Interpolation",
        "Mechanism": "MNAR",
        "Target": col,
        "MAE": mae_value,
        "RMSE": rmse_value
    })

interp_mnar_table = pd.DataFrame(interp_mnar_results)

display(interp_mnar_table.round(4))

# %%
mnar_comparison = pd.concat(
    [
        iter_mnar_table,
        mf_mnar_table,
        interp_mnar_table
    ],
    ignore_index=True
)
mnar_comparison = mnar_comparison.sort_values(
    by=["Target", "RMSE"]
).reset_index(drop=True)

display(mnar_comparison.round(4))

# %%
all_results = pd.concat(
    [
        # MCAR
        iter_table,
        mf_table,
        interp_table,

        # MAR
        iter_mar_table,
        mf_mar_table,
        interp_mar_table,

        # MNAR
        iter_mnar_table,
        mf_mnar_table,
        interp_mnar_table
    ],
    ignore_index=True
)
all_results = all_results.sort_values(
    by=["Mechanism", "Target", "RMSE"]
).reset_index(drop=True)

display(all_results.round(4))

print("Total experiments:", len(all_results))

# %%
best_results = all_results.loc[
    all_results.groupby(
        ["Mechanism", "Target"]
    )["RMSE"].idxmin()
]
best_results = best_results.sort_values(
    by=["Mechanism", "Target"]
).reset_index(drop=True)
display(best_results.round(4))
winner_table = best_results.pivot(
    index="Target",
    columns="Mechanism",
    values="Method"
)
display(winner_table)


