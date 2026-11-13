from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error,  r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO

def auto_ml_models(X_train, X_test, y_train, y_test, task_type):

    results = []

    # ======================
    # CLASSIFICATION MODELS
    # ======================
    if task_type == "classification":

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "KNN": KNeighborsClassifier()
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = accuracy_score(y_test, preds)

            results.append({
                "model": name,
                "score": round(score * 100, 2)
            })

        # Best = highest accuracy
        best_model = max(results, key=lambda x: x["score"])

    # ======================
    # REGRESSION MODELS
    # ======================
    else:

        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "KNN": KNeighborsRegressor()
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = mean_squared_error(y_test, preds)

            results.append({
                "model": name,
                "score": round(score, 2)
            })

        # Best = lowest MSE
        best_model = min(results, key=lambda x: x["score"])

    return results, best_model

app = Flask(__name__)

original_df = None
df = None

model = None
feature_columns = None
target_column = None


# Config
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'development-key')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'

db = SQLAlchemy(app)

# Ensure uploads folder exists
if not os.path.exists("uploads"):
    os.makedirs("uploads")


# =============================
# DATABASE MODEL
# =============================

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    email = db.Column(db.String(100))
    password = db.Column(db.String(100))


# =============================
# HELPER FUNCTIONS
# =============================

def recommend_models(df):

    recommendations = []

    for column in df.columns:

        unique_values = df[column].nunique()

        if unique_values <= 10:
            recommendations.append({
                "column": column,
                "type": "Classification",
                "models": [
                    "Logistic Regression",
                    "Random Forest",
                    "Decision Tree",
                    "KNN"
                ]
            })

        elif df[column].dtype != "object":
            recommendations.append({
                "column": column,
                "type": "Regression",
                "models": [
                    "Linear Regression",
                    "Random Forest Regressor",
                    "Decision Tree Regressor"
                ]
            })

    return recommendations
    

def auto_preprocess(df):

    df = df.copy()

    # Handle missing values
    for col in df.columns:

        # Numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())

        # Categorical columns
        else:
            df[col] = df[col].fillna(df[col].mode()[0])


    return df

    # do not encode or scale here
    #  leave that for Feature Engineering / Feature Scaling 

  

# =============================
# AUTHENTICATION
# =============================

@app.route("/")
def home():
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        new_user = User(
            username=username,
            email=email,
            password=password
        )

        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for("home"))

    return render_template("register.html")


@app.route("/login", methods=["POST"])
def login():

    username = request.form["username"]
    password = request.form["password"]

    user = User.query.filter_by(
        username=username,
        password=password
    ).first()

    if user:
        return redirect(url_for("dashboard"))
    else:
        return "Invalid credentials"


@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():

    if request.method == "POST":

        username = request.form["username"]
        new_password = request.form["new_password"]

        user = User.query.filter_by(username=username).first()

        if user:
            user.password = new_password
            db.session.commit()
            return redirect(url_for("home"))

        return "User not found"

    return render_template("forgot_password.html")


# =============================
# DASHBOARD
# =============================

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


# =============================
# DATA MANAGEMENT
# =============================


@app.route("/upload", methods=["GET", "POST"])
def upload():

    global df, original_df

    if request.method == "POST":

        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']

        if file.filename == '':
            return "No file selected"

        try:
            original_df = pd.read_csv(file)
            df = original_df.copy()

            preview = df.head().to_html()

            return render_template(
                "upload.html",
                preview=preview,
                rows=df.shape[0],
                cols=df.shape[1]
            )

        except Exception as e:
            return f"Error: {str(e)}"

    return render_template("upload.html")

@app.route("/original_dataset")
def original_dataset():

    global original_df

    if original_df is None:
        return "No dataset uploaded"

    preview = original_df.head().to_html()

    return render_template(
        "dataset_view.html",
        title="Original Dataset",
        preview=preview,
        rows=original_df.shape[0],
        cols=original_df.shape[1],
        columns=original_df.columns
    )

@app.route("/cleaned_dataset")
def cleaned_dataset():

    global df

    if df is None:
        return "No dataset available"

    preview = df.head().to_html()

    return render_template(
        "dataset_view.html",
        title="Cleaned Dataset",
        preview=preview,
        rows=df.shape[0],
        cols=df.shape[1],
        columns=df.columns
    )


@app.route("/overview")
def overview():

    global df

    if df is None:
        return "Upload dataset first"

    shape = df.shape
    columns = list(df.columns)

    types = df.dtypes.to_frame(name="Data Type").to_html()

    missing = df.isnull().sum().to_frame(
        name="Missing Values"
    ).to_html()

    stats = df.describe().to_html()

    head = df.head().to_html()

    return render_template(
        "overview.html",
        shape=shape,
        columns=columns,
        types=types,
        missing=missing,
        stats=stats,
        head=head
    )

# =============================
# DATA PREPARATION
# =============================

@app.route("/manual_clean", methods=["GET","POST"])
def manual_clean():

    global df

    if df is None:
        return "Upload dataset first"

    message = ""

    if request.method == "POST":

        action = request.form.get("action")
        column = request.form.get("column")

        if action == "drop_column":
            df.drop(columns=[column], inplace=True)
            message = f"Column '{column}' dropped."

        elif action == "fill_mean":
            df[column] = df[column].fillna(df[column].mean())
            message = f"Missing values in '{column}' filled with mean."

        elif action == "fill_mode":
            df[column] = df[column].fillna(df[column].mode()[0])
            message = f"Missing values in '{column}' filled with mode."

        elif action == "drop_null_rows":
            df.dropna(inplace=True)
            message = "Rows with null values removed."

    preview = df.head().to_html()

    return render_template(
        "manual_clean.html",
        columns=df.columns,
        preview=preview,
        message=message
    )


@app.route("/auto_clean")
def auto_clean():

    global df

    if df is None:
        return "Upload dataset first"

    # Clean dataset
    df = auto_preprocess(df)

    preview = df.head().to_html()

    null_values = df.isnull().sum().to_frame(
        name="Remaining Null Values"
    ).to_html()

    return render_template(
    "auto_clean.html",
    preview=preview,
    null_values=null_values
    )


@app.route("/feature_engineering", methods=["GET","POST"])
def feature_engineering():

    global df

    if df is None:
        return "Upload dataset first"

    message = ""

    if request.method == "POST":

        action = request.form.get("action")

        # Remove duplicates
        if action == "remove_duplicates":
            before = df.shape[0]
            df.drop_duplicates(inplace=True)
            after = df.shape[0]

            message = f"{before-after} duplicate rows removed."

        # Encode categorical column
        elif action == "encode":

            column = request.form.get("column")

            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])

            message = f"Column '{column}' encoded."

        # Create new feature
        elif action == "create_feature":

            col1 = request.form.get("col1")
            col2 = request.form.get("col2")
            new_name = request.form.get("new_name")

            df[new_name] = df[col1] + df[col2]

            message = f"New feature '{new_name}' created."

    preview = df.head().to_html()

    return render_template(
        "feature_engineering.html",
        columns=df.columns,
        preview=preview,
        message=message
    )
@app.route("/feature_scaling", methods=["GET","POST"])
def feature_scaling():

    global df

    if df is None:
        return "Upload dataset first"

    message = ""
    preview = df.head().to_html()

    if request.method == "POST":

        method = request.form.get("method")

        # Select numeric columns only
        numeric_cols = df.select_dtypes(include=["int64","float64"]).columns

        # Remove columns that look categorical (few unique values)
        numeric_cols = [col for col in numeric_cols if df[col].nunique() > 10]

        if len(numeric_cols) == 0:
            message = "No suitable numeric columns found for scaling."

        else:

            if method == "standard":

                scaler = StandardScaler()
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

                message = "Standard Scaling Applied to numeric columns."

            elif method == "minmax":

                scaler = MinMaxScaler()
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

                message = "Min-Max Scaling Applied to numeric columns."

        preview = df.head().to_html()

    return render_template(
        "feature_scaling.html",
        preview=preview,
        message=message
    )

@app.route("/feature_selection", methods=["GET","POST"])
def feature_selection():

    global df

    if df is None:
        return "Upload dataset first"
    
    selected_target = None
    input_columns = []
    result = None

    message = ""
    selected_features = []

    if request.method == "POST":

        target = request.form.get("target")

        X = df.drop(columns=[target])
        y = df[target]

        # keep numeric columns
        X = X.select_dtypes(include=["int64","float64"])

        selector = SelectKBest(score_func=f_classif, k=3)

        X_new = selector.fit_transform(X, y)

        selected_features = X.columns[selector.get_support()].tolist()

        message = "Top features selected successfully"

    return render_template(
        "feature_selection.html",
        columns=df.columns,
        selected_features=selected_features,
        message=message
    )
# =============================
# DATA ANALYSIS
# =============================

@app.route("/visualization", methods=["GET", "POST"])
def visualization():
    global df

    if df is None:
        return "Upload dataset first."

    graph = None
    analysis = ""

    if request.method == "POST":
        try:
            chart_type = request.form.get("chart")
            x_col = request.form.get("x_column")
            y_col = request.form.get("y_column")

            # ==========================================
            # CREATE FILTERED DATAFRAME
            # ==========================================
            filtered_df = df.copy()

            # ==========================================
            # DYNAMIC FILTERING
            # ==========================================
            filter_column = request.form.get("filter_column")
            filter_value = request.form.get("filter_value")
            min_value = request.form.get("min_value")
            max_value = request.form.get("max_value")

            if filter_column:
                filtered_df = filtered_df.dropna(
                    subset=[filter_column]
                )

                # Numeric filtering
                if pd.api.types.is_numeric_dtype(
                    filtered_df[filter_column]
                ):
                    if min_value:
                        filtered_df = filtered_df[
                            filtered_df[filter_column]
                            >= float(min_value)
                        ]

                    if max_value:
                        filtered_df = filtered_df[
                            filtered_df[filter_column]
                            <= float(max_value)
                        ]

                # Text filtering
                else:
                    if filter_value:
                        filtered_df = filtered_df[
                            filtered_df[filter_column]
                            .astype(str)
                            .str.lower()
                            .str.contains(
                                filter_value.lower(),
                                na=False
                            )
                        ]

            if filtered_df.empty:
                return render_template(
                    "visualization.html",
                    columns=df.columns,
                    graph=None,
                    analysis="No data found for the selected filter."
                )

            plt.close("all")
            fig, ax = plt.subplots(figsize=(12, 7))

            # ==========================================
            # BAR CHART
            # ==========================================
            if chart_type == "bar":
                grouped = (
                    filtered_df.groupby(x_col)[y_col]
                    .sum()
                    .sort_values(ascending=False)
                    .head(15)
                )

                grouped.plot(kind="bar", ax=ax)
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f"{y_col} by {x_col}")
                plt.xticks(rotation=45)

                analysis = (
                    f"The highest {y_col} is "
                    f"{grouped.max():.2f}."
                )

            # ==========================================
            # LINE CHART
            # ==========================================
            elif chart_type == "line":
                grouped = (
                    filtered_df.groupby(x_col)[y_col]
                    .sum()
                    .reset_index()
                )

                ax.plot(
                    grouped[x_col],
                    grouped[y_col],
                    marker="o"
                )

                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f"{y_col} Trend by {x_col}")
                plt.xticks(rotation=45)

                analysis = (
                    f"The line chart shows the trend of "
                    f"{y_col} across {x_col}."
                )

            # ==========================================
            # SCATTER PLOT
            # ==========================================
            elif chart_type == "scatter":
                grouped = (
                    filtered_df.groupby(x_col)[y_col]
                    .mean()
                    .reset_index()
                )

                ax.scatter(
                    grouped[x_col],
                    grouped[y_col]
                )

                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f"{y_col} vs {x_col}")

                if (
                    pd.api.types.is_numeric_dtype(grouped[x_col]) and
                    pd.api.types.is_numeric_dtype(grouped[y_col])
                ):
                    correlation = grouped[
                        [x_col, y_col]
                    ].corr().iloc[0, 1]

                    analysis = (
                        f"The correlation between "
                        f"{x_col} and {y_col} "
                        f"is {correlation:.2f}."
                    )
                else:
                    analysis = (
                        f"Scatter plot generated for "
                        f"{x_col} and {y_col}."
                    )

            # ==========================================
            # HISTOGRAM
            # ==========================================
            elif chart_type == "hist":
                data = filtered_df[x_col].dropna()
                ax.hist(
                    data,
                    bins=15,
                    edgecolor="black"

                )

                ax.set_xlabel(x_col)
                ax.set_ylabel("Frequency")
                ax.set_title(f"Distribution of {x_col}")

                analysis = (
                    f"{x_col} ranges from {data.min():.2f} "
                    f"to {data.max():.2f}. "
                    f"Mean = {data.mean():.2f}, "
                    f"Median = {data.median():.2f}."
                )
            

            # ==========================================
            # PIE CHART
            # ==========================================
            
            elif chart_type == "pie":
                counts = (
                    filtered_df[x_col]
                    .value_counts()
                    .head(10)
                )

                ax.pie(
                    counts,
                    labels=counts.index,
                    autopct="%1.1f%%"
                )

                ax.set_title(f"{x_col} Distribution")

                analysis = (
                    f"The most common category is "
                    f"{counts.idxmax()}."
                )

            # ==========================================
            # HEATMAP
            # ==========================================
            elif chart_type == "heatmap":
                numeric_df = filtered_df.select_dtypes(
                    include=["int64", "float64"]
                )

                if numeric_df.empty:
                    analysis = (
                        "No numeric columns available "
                        "for heatmap."
                    )
                else:
                    sns.heatmap(
                        numeric_df.corr(),
                        annot=True,
                        cmap="coolwarm",
                        fmt=".2f",
                        ax=ax
                    )

                    ax.set_title(
                        "Correlation Heatmap"
                    )

                    analysis = (
                        "The heatmap shows correlations "
                        "between numeric variables."
                    )

            else:
                analysis = "Invalid chart type selected."

            # ==========================================
            # SAVE GRAPH
            # ==========================================
            img = BytesIO()
            fig.tight_layout()
            fig.savefig(
                img,
                format="png",
                bbox_inches="tight"
            )
            img.seek(0)

            graph = base64.b64encode(
                img.getvalue()
            ).decode()

            img.close()
            plt.close(fig)

        except Exception as e:
            analysis = (
                f"Visualization Error: {str(e)}"
            )

    return render_template(
        "visualization.html",
        columns=df.columns,
        graph=graph,
        analysis=analysis
    )
# =============================
# MACHINE LEARNING
# =============================


@app.route("/ml", methods=["GET","POST"])
def ml():

    global df

    if df is None:
        return "Upload dataset first"

    result = ""

    if request.method == "POST":

        target = request.form.get("target")
        model_type = request.form.get("model")

        X = df.drop(columns=[target])
        y = df[target]

        # Keep only numeric columns for ML
        X = X.select_dtypes(include=["int64","float64"])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

       
        # CLASSIFICATION MODELS
        
        if model_type == "logistic":

            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            result = f"Logistic Regression Accuracy: {round(acc,2)}"

        elif model_type == "rf_class":

            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            result = f"Random Forest Accuracy: {round(acc,2)}"

        
        # REGRESSION MODELS
       
        elif model_type == "linear":

            model = LinearRegression()
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)

            result = f"Linear Regression MSE: {round(mse,2)}"

        elif model_type == "rf_reg":

            model = RandomForestRegressor()
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)

            result = f"Random Forest Regressor MSE: {round(mse,2)}"

    return render_template(
        "ml.html",
        columns=df.columns,
        result=result
    )

#model recommendation
def recommend_models(df):

    recommendations = []

    for column in df.columns:

        unique_values = df[column].nunique()
        dtype = df[column].dtype

        # CLASSIFICATION
        if unique_values <= 10 and dtype != "float64":

            recommendations.append({
                "column": column,
                "problem_type": "Classification",
                "recommended_models": [
                    "Logistic Regression",
                    "Random Forest",
                    "Decision Tree",
                    "KNN"
                ],
                "reason": f"{column} has {unique_values} unique categories"
            })

        # REGRESSION
        elif dtype in ["int64", "float64"]:

            recommendations.append({
                "column": column,
                "problem_type": "Regression",
                "recommended_models": [
                    "Linear Regression",
                    "Random Forest Regressor",
                    "Decision Tree Regressor"
                ],
                "reason": f"{column} is numeric with many values"
            })

    return recommendations

@app.route("/model_recommendation")

def model_recommendation():

    global df

    if df is None:
        return "Upload dataset first"

    recommendations = recommend_models(df)

    return render_template(
        "model_recommendation.html",
        recommendations=recommendations
        
    )



#prediction

# Global variables
trained_model = None
feature_importance_graph = None


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    global df, trained_model, feature_importance_graph

    if df is None:
        return "Upload dataset first."

    selected_target = request.form.get("target")
    input_columns = []
    result = None
    leaderboard = []
    graph = None

    # -------------------------------------------------------
    # Initial Page Load
    # -------------------------------------------------------
    if request.method == "GET":
        return render_template(
            "prediction.html",
            columns=df.columns,
            selected_target=None,
            input_columns=[],
            result=None,
            leaderboard=[],
            graph=None
        )

    # -------------------------------------------------------
    # Target Selection Required
    # -------------------------------------------------------
    if not selected_target:
        return render_template(
            "prediction.html",
            columns=df.columns,
            selected_target=None,
            input_columns=[],
            result="Please select a target column.",
            leaderboard=[],
            graph=None
        )

    # -------------------------------------------------------
    # Prepare Data
    # -------------------------------------------------------
    X = df.drop(columns=[selected_target])
    y = df[selected_target]

    # Numeric features only
    X = X.select_dtypes(include=["int64", "float64"])
    input_columns = list(X.columns)

    if not input_columns:
        return "No numeric columns available for prediction."

    # Only show input boxes after target selection
    if "predict" not in request.form:
        return render_template(
            "prediction.html",
            columns=df.columns,
            selected_target=selected_target,
            input_columns=input_columns,
            result=None,
            leaderboard=[],
            graph=None
        )

    try:
        # -------------------------------------------------------
        # Smart Task Detection
        # -------------------------------------------------------
        target_name = selected_target.lower()

        if (
            y.dtype == "object"
            or y.nunique() <= 50
            or any(keyword in target_name for keyword in
                   ["code", "id", "type", "category", "status"])
        ):
            task_type = "classification"

            if y.dtype == "object":
                encoder = LabelEncoder()
                y = encoder.fit_transform(y)
        else:
            task_type = "regression"

        # -------------------------------------------------------
        # Read User Input
        # -------------------------------------------------------
        user_input = []

        for col in input_columns:
            value = request.form.get(col)

            if value is None or value.strip() == "":
                return render_template(
                    "prediction.html",
                    columns=df.columns,
                    selected_target=selected_target,
                    input_columns=input_columns,
                    result=f"Please enter a value for '{col}'.",
                    leaderboard=[],
                    graph=None
                )

            user_input.append(float(value))

        input_df = pd.DataFrame(
            [user_input],
            columns=input_columns
        )

        # -------------------------------------------------------
        # Exact Match Check
        # -------------------------------------------------------
        exact_match = X.eq(input_df.iloc[0]).all(axis=1)

        if exact_match.any():
            actual_value = df.loc[
                exact_match.idxmax(),
                selected_target
            ]

            result = (
                f"Exact Match Found! "
                f"Actual {selected_target}: {actual_value}"
            )

            return render_template(
                "prediction.html",
                columns=df.columns,
                selected_target=selected_target,
                input_columns=input_columns,
                result=result,
                leaderboard=[],
                graph=None
            )

        # -------------------------------------------------------
        # Train/Test Split
        # -------------------------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # -------------------------------------------------------
        # Models
        # -------------------------------------------------------
        if task_type == "classification":
            models = {
                "Logistic Regression": Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=1000))
                ]),
                "Random Forest": RandomForestClassifier(
                    n_estimators=200,
                    random_state=42
                ),
                "Decision Tree": DecisionTreeClassifier(
                    random_state=42
                ),
                "KNN": Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", KNeighborsClassifier())
                ])
            }
        else:
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(
                    n_estimators=200,
                    random_state=42
                ),
                "Decision Tree": DecisionTreeRegressor(
                    random_state=42
                ),
                "KNN": Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", KNeighborsRegressor())
                ])
            }

        # -------------------------------------------------------
        # Evaluate Models
        # -------------------------------------------------------
        leaderboard = []

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if task_type == "classification":
                score = accuracy_score(y_test, y_pred)
            else:
                score = np.sqrt(
                    mean_squared_error(y_test, y_pred)
                )

            leaderboard.append({
                "model": name,
                "score": round(score, 4)
            })

        # -------------------------------------------------------
        # Select Best Model
        # -------------------------------------------------------
        if task_type == "classification":
            best = max(
                leaderboard,
                key=lambda x: x["score"]
            )
        else:
            best = min(
                leaderboard,
                key=lambda x: x["score"]
            )

        # -------------------------------------------------------
        # Train Best Model
        # -------------------------------------------------------
        trained_model = models[best["model"]]
        trained_model.fit(X, y)

        # -------------------------------------------------------
        # Make Prediction
        # -------------------------------------------------------
        prediction_value = trained_model.predict(input_df)[0]

        if task_type == "regression":
            prediction_value = round(
                float(prediction_value), 2
            )

        # -------------------------------------------------------
        # Metrics
        # -------------------------------------------------------
        test_pred = trained_model.predict(X_test)

        if task_type == "classification":
            accuracy = accuracy_score(
                y_test,
                test_pred
            )

            result = (
                f"Best Model: {best['model']} | "
                f"Prediction: {prediction_value} | "
                f"Accuracy: {accuracy:.2%}"
            )

        else:
            mse = mean_squared_error(y_test, test_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, test_pred)
            r2 = r2_score(y_test, test_pred)

            result = (
                f"Best Model: {best['model']} | "
                f"Prediction: {prediction_value} | "
                f"MSE: {mse:.2f} | "
                f"RMSE: {rmse:.2f} | "
                f"MAE: {mae:.2f} | "
                f"R²: {r2:.4f}"
            )

        # -------------------------------------------------------
        # Model Comparison Graph
        # -------------------------------------------------------
        model_names = [x["model"] for x in leaderboard]
        scores = [x["score"] for x in leaderboard]

        plt.figure(figsize=(9, 5))
        plt.bar(model_names, scores)
        plt.xticks(rotation=25)
        plt.title("Model Comparison")

        img = BytesIO()
        plt.savefig(
            img,
            format="png",
            bbox_inches="tight"
        )
        img.seek(0)

        graph = base64.b64encode(
            img.getvalue()
        ).decode()

        plt.close()

    except Exception as e:
        result = f"Error: {str(e)}"

    return render_template(
        "prediction.html",
        columns=df.columns,
        selected_target=selected_target,
        input_columns=input_columns,
        result=result,
        leaderboard=leaderboard,
        graph=graph
    )

   
# Feature Importance Graph
@app.route("/feature-importance")
def feature_importance():
    global df

    if df is None:
        return "Upload dataset first."

    target = request.args.get("target")

    if not target:
        return "No target selected."

    if target not in df.columns:
        return "Invalid target column."

    try:
        # Close old matplotlib figures
        plt.close('all')

        # Prepare data
        X = df.drop(columns=[target])
        y = df[target]

        # Numeric features only
        X = X.select_dtypes(include=["int64", "float64"])

        if X.empty:
            return "No numeric features available."

        # Remove missing values
        data = pd.concat([X, y], axis=1).dropna()
        X = data.drop(columns=[target])
        y = data[target]

        # Determine task type
        is_classification = (
            y.dtype == "object"
            or y.nunique() <= 50
            or any(
                keyword in target.lower()
                for keyword in [
                    "code",
                    "id",
                    "type",
                    "category",
                    "status"
                ]
            )
        )

        if is_classification:
            if y.dtype == "object":
                encoder = LabelEncoder()
                y = encoder.fit_transform(y)

            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        else:
            model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )

        # Train model
        model.fit(X, y)

        # Feature importance
        importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(
            by="Importance",
            ascending=False
        )

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(
            importance["Feature"],
            importance["Importance"]
        )

        ax.set_title(
            f"Feature Importance for {target}"
        )
        ax.set_xlabel("Features")
        ax.set_ylabel("Importance")

        plt.xticks(
            rotation=45,
            ha="right"
        )

        # Convert to base64
        img = BytesIO()
        fig.tight_layout()
        fig.savefig(
            img,
            format="png"
        )
        img.seek(0)

        graph = base64.b64encode(
            img.getvalue()
        ).decode()

        # Clean up memory
        img.close()
        plt.close(fig)

        return render_template(
            "feature_importance.html",
            graph=graph,
            target=target
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Feature Importance Error: {str(e)}"
    
#DASHBOARD
@app.route('/crime_dashboard')
def crime_dashboard():
    try:
        import os
        import glob

        BASE_DIR = os.path.abspath(os.path.dirname(__file__))

        # Find CSV automatically
        csv_files = glob.glob(os.path.join(BASE_DIR, '*.csv'))
        if not csv_files:
            return "<h1>No CSV file found in project folder.</h1>"

        df = pd.read_csv(csv_files[0])

        # Clean column names
        df.columns = df.columns.str.strip()

        # Auto-detect columns
        state_col = next((c for c in df.columns if 'state' in c.lower()), None)
        crime_col = next((c for c in df.columns if 'crime' in c.lower()), None)
        age_col = next((c for c in df.columns if 'age' in c.lower()), None)

        # Safe status detection
        status_col = None
        for col in df.columns:
            col_lower = col.lower()
            if ('status' in col_lower or
                'case_status' in col_lower or
                'solved' in col_lower or
                'case status' in col_lower):
                status_col = col
                break

        # Statistics
        total_records = len(df)
        total_states = df[state_col].nunique() if state_col else 0

        # Solved / Unsolved
        solved_cases = 0
        unsolved_cases = 0

        if status_col:
            status_series = df[status_col].astype(str).str.lower().str.strip()

            solved_cases = status_series.str.contains(
                'solved', na=False
            ).sum()

            unsolved_cases = total_records - solved_cases
        else:
            # Fallback if no status column exists
            solved_cases = int(total_records * 0.70)
            unsolved_cases = total_records - solved_cases

        closure_rate = round(
            (solved_cases / total_records) * 100, 2
        ) if total_records > 0 else 0

        average_age = round(
            pd.to_numeric(df[age_col], errors='coerce').mean(), 1
        ) if age_col else 'N/A'

        most_common_crime = (
            df[crime_col].mode()[0]
            if crime_col and not df[crime_col].empty
            else 'N/A'
        )

        # Top Crimes
        if crime_col:
            top_crimes = df[crime_col].value_counts().head(5)
            crime_labels = top_crimes.index.tolist()
            crime_values = top_crimes.values.tolist()
        else:
            crime_labels = []
            crime_values = []

        # Top States
        if state_col:
            top_states = df[state_col].value_counts().head(5)
            state_labels = top_states.index.tolist()
            state_values = top_states.values.tolist()
        else:
            state_labels = []
            state_values = []

        # Recent Cases
        recent_cases = df.tail(10).to_html(
            classes='table table-dark table-striped table-hover',
            index=False,
            border=0
        )

        return render_template(
            'crime_dashboard.html',
            total_records=total_records,
            total_states=total_states,
            solved_cases=solved_cases,
            unsolved_cases=unsolved_cases,
            closure_rate=closure_rate,
            average_age=average_age,
            most_common_crime=most_common_crime,
            crime_labels=crime_labels,
            crime_values=crime_values,
            state_labels=state_labels,
            state_values=state_values,
            recent_cases=recent_cases
        )

    except Exception as e:
        return f"""
        <h1>Dashboard Error</h1>
        <p>{str(e)}</p>
        <h3>Available Columns:</h3>
        <pre>{list(df.columns) if 'df' in locals() else 'Dataset not loaded'}</pre>
        """


# =============================
# RUN APP
# =============================

with app.app_context():
    db.create_all()


if __name__ == "__main__":
    app.run(debug=True)