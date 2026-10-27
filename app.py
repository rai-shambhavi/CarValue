"""FairValue (CarDekho): Streamlit dashboard.  Run:  streamlit run app.py
Needs `fairvalue_bundle.joblib`, created by running FairValue_CarDekho.ipynb. Nothing is retrained here.
Libraries: streamlit, pandas, numpy, matplotlib, seaborn, joblib (+ scikit-learn for the saved model)."""
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import streamlit as st
import joblib

BUNDLE = "fairvalue_bundle.joblib"
st.set_page_config(page_title="FairValue", page_icon="🚗", layout="wide")
sns.set_theme(style="whitegrid")
inr = lambda v: f"₹{v:,.0f}"


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


@st.cache_resource(show_spinner="Loading trained model…")
def load(path):
    return joblib.load(path)


if not os.path.exists(BUNDLE):
    st.error(f"`{BUNDLE}` not found. Run all cells of **FairValue_CarDekho.ipynb** first (in the same folder as app.py), then restart the app.")
    st.stop()

B = load(BUNDLE)
pipe, FEATS, NUM, CAT, OPT = B["pipeline"], B["features"], B["num"], B["cat"], B["options"]
eda, res, info = B["eda_sample"], B["results"], B["info"]
mae = float(res.loc[res["Model"] == B["best_model"], "MAE"].iloc[0])
BRANDS = sorted(OPT["brand"])


def predict(rows):
    """Blank/'unknown' values become NaN (imputed by the pipeline); unseen categories are handled by the encoders."""
    d = pd.DataFrame(rows)
    d["km_per_year"] = d["km_driven"] / d["vehicle_age"].clip(lower=1)
    for c in FEATS:
        if c not in d:
            d[c] = np.nan
    for c in CAT:
        d[c] = d[c].map(lambda v: v.strip() if isinstance(v, str) and v.strip() and v != "unknown" else np.nan)
    return pipe.predict(d[FEATS])


def defaults(brand, model):
    sub = eda[(eda["brand"] == brand) & (eda["model"] == model)]
    sub = sub if len(sub) else eda[eda["brand"] == brand]
    return {c: float(sub[c].median()) for c in ["mileage", "engine", "max_power", "seats"] if c in sub}


page = st.sidebar.radio("Navigate", ["Price Prediction", "Model Performance", "EDA", "Depreciation", "Distance (km) Insights", "Dataset Info"])
st.sidebar.caption(f"Best model: **{B['best_model']}**\n\nTypical error (test MAE): **{inr(mae)}**")
st.title("🚗 CarValue")
st.caption("Used-car price prediction & depreciation analyzer · CarDekho listings · prices in Indian Rupees (₹)")

# ───────────── Prediction ─────────────
if page == "Price Prediction":
    st.subheader("Estimate price")
    c1, c2, c3 = st.columns(3)
    brand = c1.selectbox("Brand", BRANDS, index=BRANDS.index("Maruti") if "Maruti" in BRANDS else 0)
    model = c2.selectbox("Model", OPT["models_by_brand"].get(brand, []))
    age = c3.number_input("Vehicle age (years)", 0, 30, 5)
    km = c1.number_input("Distance driven (km)", 0, 600_000, 50_000, 1000)
    fuel = c2.selectbox("Fuel type", ["unknown"] + OPT.get("fuel_type", []))
    trans = c3.selectbox("Transmission", ["unknown"] + OPT.get("transmission_type", []))
    seller = c1.selectbox("Seller type", ["unknown"] + OPT.get("seller_type", []))
    d0, k = defaults(brand, model), f"{brand}_{model}"
    eng = c2.number_input("Engine (cc)", 300, 6000, int(d0.get("engine", 1200)), 50, key="e" + k)
    pw = c3.number_input("Max power (bhp)", 20.0, 700.0, float(min(max(d0.get("max_power", 80.0), 20.0), 700.0)), 1.0, key="p" + k)
    mil = c1.number_input("Fuel efficiency (kmpl)", 3.0, 40.0, float(min(max(d0.get("mileage", 18.0), 3.0), 40.0)), 0.1, key="m" + k)
    seats = c2.number_input("Seats", 2, 10, int(min(max(d0.get("seats", 5), 2), 10)), key="s" + k)

    if st.button("Predict price", type="primary"):
        if km / max(age, 1) > 60_000:
            st.warning("That distance is unusually high for the car's age. The estimate may be unreliable.")
        row = dict(vehicle_age=int(age), km_driven=float(km), brand=brand, model=model, fuel_type=fuel, transmission_type=trans,
                   seller_type=seller, engine=float(eng), max_power=float(pw), mileage=float(mil), seats=float(seats))
        p = float(predict([row])[0])
        m1, m2, m3 = st.columns(3)
        m1.metric("Estimated fair price", inr(p))
        m2.metric(f"Typical range (± test MAE {inr(mae)})", f"{inr(max(p - mae, 0))} – {inr(p + mae)}")
        m3.metric("In lakh", f"₹{p/1e5:,.2f} L")
        sim = eda[(eda["brand"] == brand) & (eda["vehicle_age"].between(age - 1, age + 1))]

# ───────────── Performance ─────────────
elif page == "Model Performance":
    st.subheader("Model comparison (held-out 20% test set)")
    st.dataframe(res.round(3), hide_index=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, m in zip(ax, ["MAE", "RMSE", "R2"]):
        sns.barplot(data=res, x="Model", y=m, color="steelblue", ax=a); a.set_title(m + (" (lower is better)" if m != "R2" else " (higher is better)")); a.tick_params(axis="x", rotation=15)
    plt.tight_layout(); show(fig)
    ts = B["test_sample"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].scatter(ts["actual"], ts["predicted"], s=6, alpha=.3); ax[0].plot([0, ts["actual"].max()], [0, ts["actual"].max()], "r--")
    ax[0].set(title=f"{B['best_model']}: actual vs predicted", xlabel="Actual (₹)", ylabel="Predicted (₹)")
    B["importance"].plot.barh(ax=ax[1], title="Feature importance (MAE increase when shuffled, ₹)")
    plt.tight_layout(); show(fig)

# ───────────── EDA ─────────────
elif page == "EDA":
    st.subheader("Exploratory analysis")
    t1, t2, t3, t4 = st.tabs(["Price", "Brands", "Age & distance", "Correlation"])
    with t1:
        fig, ax = plt.subplots(figsize=(8, 3.5)); sns.histplot(eda["selling_price"], bins=50, kde=True, ax=ax); ax.set_title("Selling price distribution (₹)"); show(fig)
        col = st.selectbox("Break down price by", [c for c in ["fuel_type", "transmission_type", "seller_type", "seats"] if c in eda])
        fig, ax = plt.subplots(figsize=(8, 3.5)); sns.boxplot(data=eda, x=col, y="selling_price", showfliers=False, ax=ax); ax.set_title(f"Price by {col}"); show(fig)
    with t2:
        n = st.slider("Top N brands by listings", 5, 30, 15)
        g = eda[eda["brand"].isin(eda["brand"].value_counts().head(n).index)].groupby("brand")["selling_price"].median().sort_values()
        fig, ax = plt.subplots(figsize=(8, 0.3 * n + 2)); g.plot.barh(ax=ax, color="steelblue"); ax.set_title("Median price by brand (₹)"); show(fig)
    with t3:
        g = eda.groupby("vehicle_age")["selling_price"].agg(["median", "count"]).query("count >= 15")
        fig, ax = plt.subplots(1, 2, figsize=(13, 4))
        ax[0].plot(g.index, g["median"], marker="o"); ax[0].set(title="Median price by vehicle age", xlabel="Vehicle age (years)", ylabel="Price (₹)")
        s = eda.sample(min(4000, len(eda)), random_state=1)
        sc = ax[1].scatter(s["km_driven"], s["selling_price"], c=s["vehicle_age"], cmap="viridis", alpha=.5, s=10); plt.colorbar(sc, ax=ax[1], label="Vehicle age")
        ax[1].set(title="Distance vs price", xlabel="km driven", ylabel="Price (₹)")
        plt.tight_layout(); show(fig)
    with t4:
        fig, ax = plt.subplots(figsize=(7, 5)); sns.heatmap(eda[[c for c in NUM + ["selling_price"] if c in eda]].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax); show(fig)

# ───────────── Depreciation ─────────────
elif page == "Depreciation":
    st.subheader("Depreciation analysis")
    st.info("**Observed** = median asking price of *different* cars at each age (cross-section, not one car tracked over time). "
            "**Simulated** = the model's estimate for one reference car when only its age changes. Treat it as an estimate, not history.")
    c1, c2, c3 = st.columns(3)
    brand = c1.selectbox("Brand", BRANDS, index=BRANDS.index("Maruti") if "Maruti" in BRANDS else 0, key="db")
    model = c2.selectbox("Model", OPT["models_by_brand"].get(brand, []), key="dm")
    mode = c3.radio("Distance assumption", ["Grows with age", "Fixed"], horizontal=True)
    kpy = c1.number_input("km per year", 1000, 40000, int(B["median_kpy"]), 500)
    fixed = c2.number_input("Fixed distance (km)", 0, 600_000, 50_000, 5000, disabled=(mode == "Grows with age"))
    max_age = c3.slider("Max age (years)", 5, 25, 15)
    fuel = c1.selectbox("Fuel type", ["unknown"] + OPT.get("fuel_type", []), key="df")
    trans = c2.selectbox("Transmission", ["unknown"] + OPT.get("transmission_type", []), key="dt")
    ages = np.arange(0, max_age + 1)
    d0 = defaults(brand, model)
    rows = [dict(vehicle_age=int(a), km_driven=float(a * kpy if mode == "Grows with age" else fixed), brand=brand, model=model, fuel_type=fuel, transmission_type=trans, **d0) for a in ages]
    sim = pd.DataFrame({"age": ages, "price": predict(rows)})
    obs = eda[eda["brand"] == brand].groupby("vehicle_age")["selling_price"].agg(["median", "count"]).query("count >= 10")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(sim["age"], sim["price"], marker="o", label="Simulated (model estimate)")
    ax.plot(obs.index, obs["median"], marker="s", linestyle=":", label=f"Observed median, all {brand} cars")
    ax.set(title="Price vs age", xlabel="Vehicle age (years)", ylabel="Price (₹)"); ax.legend(); show(fig)
    sim["Yearly change %"] = sim["price"].pct_change() * 100
    sim["Retained vs new %"] = sim["price"] / sim["price"].iloc[0] * 100
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(6, 3.5)); ax.bar(sim["age"][1:], sim["Yearly change %"][1:], color="salmon"); ax.set(title="Simulated year-over-year change (%)", xlabel="Age (years)"); show(fig)
    b.dataframe(sim.rename(columns={"price": "Estimated price (₹)"}).round(1), hide_index=True, height=330)
    if len(sim) > 10:
        st.write(f"Model estimate: about **{sim['Retained vs new %'].iloc[5]:.0f}%** of the age-0 value is retained after 5 years and **{sim['Retained vs new %'].iloc[10]:.0f}%** after 10 years.")

# ───────────── Distance ─────────────
elif page == "Distance (km) Insights":
    st.subheader("Distance-driven insights")
    band = pd.cut(eda["km_driven"], [0, 10e3, 25e3, 40e3, 60e3, 80e3, 100e3, 150e3, 600e3])
    g = eda.groupby(band, observed=True)["selling_price"].agg(["median", "count"])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].bar(range(len(g)), g["median"], color="steelblue"); ax[0].set_xticks(range(len(g))); ax[0].set_xticklabels([str(i) for i in g.index], rotation=45, ha="right")
    ax[0].set(title="Median price by distance band", ylabel="Price (₹)")
    sns.histplot(eda["km_per_year"].clip(upper=40000), bins=50, ax=ax[1]); ax[1].set_title("km driven per year")
    plt.tight_layout(); show(fig)
    st.markdown("**Effect of distance with age held constant (model estimate)**")
    age = st.slider("Vehicle age (years)", 0, 20, 5)
    brand = eda["brand"].mode()[0]; model = eda.loc[eda["brand"] == brand, "model"].mode()[0]
    kms = np.arange(0, 200_001, 10_000)
    p = predict([dict(vehicle_age=age, km_driven=float(k), brand=brand, model=model, **defaults(brand, model)) for k in kms])
    fig, ax = plt.subplots(figsize=(8, 4)); ax.plot(kms, p, marker="o"); ax.set(title=f"Predicted price vs km: {brand} {model}, {age} yrs old", xlabel="km driven", ylabel="Price (₹)"); show(fig)
    st.caption("Age and distance are correlated in real data, so extreme combinations (e.g., a 1-year-old car with 200,000 km) are extrapolations.")

# ───────────── Dataset info ─────────────
else:
    st.subheader("Dataset information")
    c = st.columns(4)
    c[0].metric("File size", f"{info['file_mb']:.2f} MB")
    c[1].metric("Raw rows × columns", f"{info['raw_shape'][0]:,} × {info['raw_shape'][1]}")
    c[2].metric("Rows after cleaning", f"{info['rows_clean']:,}")
    c[3].metric("Train / test rows", f"{info['train_rows']:,} / {info['test_rows']:,}")
    st.write("**Columns in the file:**", ", ".join(info["columns"]))
    st.write("**Missing values in raw data (%):**")
    st.dataframe(pd.Series(info["missing_pct_raw"], name="missing %").sort_values(ascending=False).to_frame())
    st.write("**Model features:**", ", ".join(FEATS))
    st.write(f"**Cleaning rules:** duplicates and rows with missing key fields removed; age 0–30 yrs; distance ≤ 600,000 km; price trimmed to {inr(info['price_range'][0])} – {inr(info['price_range'][1])} (0.5–99.5 percentile).")
    st.warning("Prices are asking prices in ₹ from one website. The `mileage` column is fuel efficiency (kmpl), not distance driven. Treat predictions as guidance, not appraisals.")
