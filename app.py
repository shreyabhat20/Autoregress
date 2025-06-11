import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, SGDRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, log_loss, accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sympy import symbols, Eq, solve, latex
from pint import UnitRegistry
import plotly.graph_objs as go

# ------------------------------
# Page Config & Theme
# ------------------------------
st.set_page_config(page_title="📊 AutoRegress AI", layout="wide")
st.markdown("""<h1 style='text-align: center; color: cyan;'>Math Modelling</h1>""", unsafe_allow_html=True)

# ------------------------------
# Data Input Section
# ------------------------------
st.sidebar.header("Step 1: Load Data")
data_option = st.sidebar.radio("Choose data source", ["Upload CSV", "Use synthetic data"])

ureg = UnitRegistry()
df = None

if data_option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded!")
else:
    slope = st.sidebar.slider("Slope (m)", 1.0, 10.0, 3.0)
    intercept = st.sidebar.slider("Intercept (c)", 0.0, 20.0, 7.0)
    noise = st.sidebar.slider("Noise", 0.0, 10.0, 4.0)
    X = np.linspace(0, 10, 100)
    y = slope * X + intercept + np.random.randn(100) * noise
    df = pd.DataFrame({'X': X, 'y': y})
    st.info("📊 Synthetic data generated.")

if df is not None:
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    st.sidebar.header("Step 2: Features")
    x_col = st.sidebar.selectbox("Select X column", df.columns)
    y_choices = [col for col in df.columns if col != x_col]
    y_col = st.sidebar.selectbox("Select y column", y_choices)

    clean_df = df[[x_col, y_col]].dropna()
    if clean_df.empty:
        st.error("❌ No data after removing NaNs.")
        st.stop()

    X = clean_df[[x_col]].values
    y = clean_df[y_col].values

    st.sidebar.header("Step 3: Choose Model")
    model_type = st.sidebar.selectbox("Model", ["Linear Regression", "Polynomial Regression", "Logistic Regression"])
    if model_type == "Polynomial Regression":
        degree = st.sidebar.slider("Degree", 2, 10, 2)

    tab1, tab2, tab3, tab4 = st.tabs(["📐 Equation", "📈 Graph", "📉 Metrics", "🔁 Reverse Solver"])

    if model_type == "Linear Regression":
        model = LinearRegression()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        with tab1:
            coef = model.coef_.ravel()[0]
            intercept = model.intercept_.item()
            x = symbols('x')
            eq = Eq(symbols('y'), coef * x + intercept)
            st.latex(latex(eq))

        with tab2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=X_test.ravel(), y=y_test, mode='markers', name='Actual'))
            fig.add_trace(go.Scatter(x=X_test.ravel(), y=y_pred, mode='lines', name='Prediction'))
            st.plotly_chart(fig)

        with tab3:
            st.metric("MSE", f"{mean_squared_error(y_test, y_pred):.3f}")
            st.metric("R2 Score", f"{r2_score(y_test, y_pred):.3f}")

        with tab4:
            y_input = st.number_input("Enter y to solve x", value=10.0)
            x_solved = (y_input - intercept) / coef
            st.success(f"Solved x: {x_solved:.3f}")

    elif model_type == "Polynomial Regression":
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        with tab1:
            terms = [f"{model.coef_[i]:.3f}x^{i}" for i in range(1, len(model.coef_))]
            intercept = model.intercept_.item()
            eq = " + ".join(terms)
            st.latex(f"y = {eq} + {intercept:.3f}")

        with tab2:
            x_vals = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
            y_vals = model.predict(poly.transform(x_vals))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=X[:, 0], y=y, mode='markers', name='Actual'))
            fig.add_trace(go.Scatter(x=x_vals[:, 0], y=y_vals, mode='lines', name='Prediction'))
            st.plotly_chart(fig)

        with tab3:
            st.metric("MSE", f"{mean_squared_error(y_test, y_pred):.3f}")
            st.metric("R2 Score", f"{r2_score(y_test, y_pred):.3f}")

    else:
        y_bin = (y > np.median(y)).astype(int)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_bin, test_size=0.2, random_state=42)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        with tab1:
            coef = model.coef_[0][0]
            intercept = model.intercept_[0]
            st.latex(r"P(y=1) = \frac{1}{1 + e^{-(%.3f x + %.3f)}}" % (coef, intercept))

        with tab2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=X_test.ravel(), y=y_proba, mode='markers', name='Predicted Prob'))
            fig.add_trace(go.Scatter(x=X_test.ravel(), y=y_test, mode='markers', name='Actual Class'))
            st.plotly_chart(fig)

        with tab3:
            st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
            st.metric("Log Loss", f"{log_loss(y_test, y_proba):.3f}")

else:
    st.warning("Upload a CSV or generate synthetic data to continue.")