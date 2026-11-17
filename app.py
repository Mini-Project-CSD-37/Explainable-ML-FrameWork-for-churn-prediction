import base64
import io
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

from flask import Flask, render_template, request
from matplotlib.patches import Circle, Wedge

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_rfc.pkl")

app = Flask(__name__)

# Continue using the existing trained Random Forest model.
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)


def checkbox(name):
    return 1 if request.form.get(name) == "1" else 0


def make_gauge(probability):
    """Create the probability gauge."""
    probability = float(np.clip(probability, 0, 1))
    fig, ax = plt.subplots(figsize=(5.2, 2.8), dpi=150)

    segments = [
        (135, 180, "#65B889", "LOW"),
        (90, 135, "#70A9D8", "MEDIUM"),
        (45, 90, "#F2C66D", "HIGH"),
        (0, 45, "#E98585", "EXTREME"),
    ]

    for start, end, color, label in segments:
        ax.add_patch(Wedge((0, 0), 1, start, end, width=0.22,
                           facecolor=color, edgecolor="white", linewidth=2))
        mid = np.deg2rad((start + end) / 2)
        ax.text(0.78 * np.cos(mid), 0.78 * np.sin(mid), label,
                ha="center", va="center", fontsize=8, fontweight="bold",
                color="#263449")

    angle = np.deg2rad(180 - probability * 180)
    ax.annotate("", xy=(0.78 * np.cos(angle), 0.78 * np.sin(angle)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#17243A", lw=3,
                                mutation_scale=17))
    ax.add_patch(Circle((0, 0), 0.045, color="#17243A"))
    ax.text(0, -0.13, f"{probability:.1%}", ha="center", va="center",
            fontsize=17, fontweight="bold", color="#17243A")

    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-0.22, 1.12)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.3)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", transparent=True, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


FEATURE_LABELS = {
    "gender": "Gender",
    "SeniorCitizen": "Senior citizen",
    "Partner": "Has a partner",
    "Dependents": "Has dependents",
    "tenure": "Tenure",
    "PhoneService": "Phone service",
    "MultipleLines": "Multiple lines",
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
    "PaperlessBilling": "Paperless billing",
    "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges",
    "InternetService_Fiber optic": "Fiber optic internet",
    "InternetService_No": "No internet service",
    "Contract_One year": "One-year contract",
    "Contract_Two year": "Two-year contract",
    "PaymentMethod_Credit card (automatic)": "Automatic credit card",
    "PaymentMethod_Electronic check": "Electronic check",
    "PaymentMethod_Mailed check": "Mailed check",
}


def get_shap_row(shap_values):
    """
    Extract class-1 (Churn = Yes) SHAP values.
    Supports the list format used by older SHAP versions and
    the ndarray formats used by newer SHAP versions.
    """
    if isinstance(shap_values, list):
        if len(shap_values) > 1:
            values = np.asarray(shap_values[1])
        else:
            values = np.asarray(shap_values[0])
        return values[0] if values.ndim > 1 else values

    values = np.asarray(shap_values)

    if values.ndim == 3:
        # Common shape: samples x features x classes.
        if values.shape[-1] > 1:
            return values[0, :, 1]
        return values[0, :, 0]

    if values.ndim == 2:
        return values[0]

    return values


def feature_value_text(name, value, contract_value=None):
    """Convert encoded model input values into readable customer values."""
    binary_yes_no = {
        "SeniorCitizen", "Partner", "Dependents", "PhoneService",
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "PaperlessBilling",
        "InternetService_Fiber optic", "InternetService_No",
        "Contract_One year", "Contract_Two year",
        "PaymentMethod_Credit card (automatic)",
        "PaymentMethod_Electronic check", "PaymentMethod_Mailed check",
    }

    if name == "gender":
        return "Female" if int(value) == 1 else "Male"
    if name in ("Contract_One year", "Contract_Two year") and contract_value == "0":
        return "No (month-to-month selected)"
    if name == "tenure":
        return f"{int(value)} months"
    if name in ("MonthlyCharges", "TotalCharges"):
        return f"{float(value):,.2f}"
    if name in binary_yes_no:
        return "Yes" if int(round(float(value))) == 1 else "No"
    return str(value)


def explain_feature(name, value, direction, contract_value=None):
    """
    Give a short contextual explanation for a feature.
    Direction always comes from the actual SHAP value, not these rules.
    """
    positive = direction == "increase"

    if name in ("Contract_One year", "Contract_Two year") and contract_value == "0":
        return ("The selected plan is month-to-month. It does not require a one- or "
                "two-year commitment, so the customer has more flexibility to leave "
                "at a future renewal. The direction shown here still comes from this "
                "feature's actual SHAP contribution.")
    if name == "Contract_One year" and int(value) == 1:
        return ("A one-year contract represents a longer commitment than a "
                "month-to-month plan. The model learned how this contract type "
                "relates to churn in the training data.")
    if name == "Contract_Two year" and int(value) == 1:
        return ("A two-year contract represents a longer commitment. The model "
                "learned how this contract type relates to churn in the training data.")
    if name == "Contract_One year" and int(value) == 0:
        return ("The customer does not have the one-year contract category. "
                "This contribution is interpreted together with the selected "
                "contract type and the other customer details.")
    if name == "Contract_Two year" and int(value) == 0:
        return ("The customer does not have the two-year contract category. "
                "This contribution is interpreted together with the selected "
                "contract type and the other customer details.")
    if name == "tenure":
        if float(value) <= 12:
            return ("The customer has a relatively short service history. "
                    "Early-stage customers may behave differently from long-term customers.")
        if float(value) >= 48:
            return ("The customer has been with the service for at least four years. "
                    "The model uses this service history when estimating churn.")
        return ("The customer's time with the service contributes to the model's "
                "assessment of churn risk.")
    if name == "MonthlyCharges":
        return (f"The monthly charge is {float(value):.2f}. The model uses the "
                "customer's recurring cost alongside contract and service details.")
    if name == "TotalCharges":
        return (f"The recorded total charges are {float(value):.2f}. This reflects "
                "the customer's accumulated charges as represented in the input.")
    if name == "PaymentMethod_Electronic check" and int(value) == 1:
        return ("The customer pays by electronic check. The model learned an "
                "association between this payment category and churn in its training data.")
    if name == "InternetService_Fiber optic" and int(value) == 1:
        return ("The customer uses fiber-optic internet. This service category "
                "contributes according to patterns learned by the model.")
    if name == "InternetService_No" and int(value) == 1:
        return ("The customer does not have internet service. This category "
                "contributes according to patterns learned by the model.")
    if name == "OnlineSecurity" and int(value) == 0:
        return ("The customer does not have online security. The model evaluates "
                "this alongside the other subscribed services.")
    if name == "TechSupport" and int(value) == 0:
        return ("The customer does not have tech support. The model evaluates "
                "this alongside the other subscribed services.")
    if name == "PaperlessBilling" and int(value) == 1:
        return ("Paperless billing is enabled. This is one of the service and "
                "billing attributes considered by the model.")

    if positive:
        return ("For this customer, this feature pushes the model's output "
                "towards a higher churn probability.")
    return ("For this customer, this feature pushes the model's output "
            "towards a lower churn probability.")


def make_retention_recommendations(features, columns, explanation_items, contract_value):
    """
    Produce practical, non-causal retention ideas.
    Model-signal suggestions are only generated from features with positive
    SHAP contributions. Month-to-month is a profile observation because it
    is the one-hot reference category, not an independently encoded feature.
    """
    recommendations = []
    positive_items = [
        item for item in explanation_items
        if item["direction"] == "increase"
    ]

    positive_names = {item["name"] for item in positive_items}

    # These are possible actions for a business to consider, not guarantees.
    if "Electronic check" in positive_names:
        recommendations.append({
            "title": "Review payment convenience",
            "detail": "Consider asking whether the customer would prefer a simpler "
                      "payment option, such as an available automatic payment method.",
            "reason": "Electronic check is pushing this model prediction towards higher churn.",
            "basis": "Model signal"
        })

    if "Monthly charges" in positive_names:
        recommendations.append({
            "title": "Review plan value",
            "detail": "Check whether the current plan still matches the customer's "
                      "needs and explain any suitable lower-cost or better-value options.",
            "reason": "Monthly charges are pushing this model prediction towards higher churn.",
            "basis": "Model signal"
        })

    if "Tenure" in positive_names:
        recommendations.append({
            "title": "Offer a proactive check-in",
            "detail": "A customer check-in can help identify unresolved issues and "
                      "make sure the customer understands the services available to them.",
            "reason": "Tenure is pushing this model prediction towards higher churn.",
            "basis": "Model signal"
        })

    if "Fiber optic internet" in positive_names:
        recommendations.append({
            "title": "Check service experience",
            "detail": "Ask whether the customer is experiencing connection, speed or "
                      "reliability problems before suggesting a plan change.",
            "reason": "Fiber-optic internet is pushing this model prediction towards higher churn.",
            "basis": "Model signal"
        })

    if "Online security" in positive_names or "Tech support" in positive_names:
        recommendations.append({
            "title": "Review support needs",
            "detail": "Check whether the customer needs help with security or technical "
                      "issues, and explain relevant support options if appropriate.",
            "reason": "A support-related feature is pushing this model prediction towards higher churn.",
            "basis": "Model signal"
        })

    # Contract = 0 is the omitted one-hot reference category in this model.
    # It is a profile observation, not an independent SHAP driver.
    if contract_value == "0":
        recommendations.append({
            "title": "Discuss contract options",
            "detail": "The customer is currently on a month-to-month plan. If it suits "
                      "their needs, explain the benefits and terms of longer-term options "
                      "without assuming they should switch.",
            "reason": "Profile observation: month-to-month is the selected contract type; "
                      "this is not presented as an independent SHAP contribution.",
            "basis": "Customer profile"
        })

    if not recommendations:
        recommendations.append({
            "title": "Start with a customer check-in",
            "detail": "Ask about the customer's recent experience and whether anything "
                      "could be improved. Avoid offering a discount or plan change without "
                      "understanding the customer's needs.",
            "reason": "No mapped positive SHAP driver was found among the displayed features.",
            "basis": "General suggestion"
        })

    return recommendations[:4]


def make_shap_explanation(features, columns, contract_value=None):
    """
    Build a larger SHAP contribution chart and concise explanations.
    Positive SHAP values increase the model output for Churn = Yes;
    negative values decrease it.
    """
    explainer = shap.TreeExplainer(model)
    raw_values = explainer.shap_values(features)
    shap_row = np.asarray(get_shap_row(raw_values), dtype=float).reshape(-1)

    if len(shap_row) != len(columns):
        raise ValueError(
            f"SHAP returned {len(shap_row)} values for {len(columns)} features."
        )

    order = np.argsort(np.abs(shap_row))[-10:]
    order = order[np.argsort(np.abs(shap_row[order]))]

    labels = [FEATURE_LABELS.get(columns[i], columns[i]) for i in order]
    values = shap_row[order]
    colors = ["#E45B64" if value > 0 else "#3986D7" for value in values]

    fig_height = max(5.0, 0.52 * len(order) + 1.8)
    fig, ax = plt.subplots(figsize=(11.5, fig_height), dpi=170)
    bars = ax.barh(labels, values, color=colors, height=0.62)

    ax.axvline(0, color="#7B8796", linewidth=1)
    ax.set_title("What influenced this churn prediction?",
                 loc="left", fontsize=15, fontweight="bold", pad=18,
                 color="#17243A")
    ax.set_xlabel("SHAP contribution to the model output for churn",
                  fontsize=10, labelpad=10, color="#66758A")
    ax.tick_params(axis="y", labelsize=10, colors="#34445B", length=0, pad=9)
    ax.tick_params(axis="x", labelsize=9, colors="#66758A")
    ax.grid(axis="x", linestyle="--", alpha=0.22)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#DCE3EC")

    max_abs = max(float(np.max(np.abs(values))) if len(values) else 0, 0.01)
    ax.set_xlim(-max_abs * 1.28, max_abs * 1.28)

    for bar, value in zip(bars, values):
        offset = max_abs * 0.035
        if value >= 0:
            ax.text(value + offset, bar.get_y() + bar.get_height() / 2,
                    f"+{value:.3f}", va="center", ha="left",
                    fontsize=8, color="#A83C45")
        else:
            ax.text(value - offset, bar.get_y() + bar.get_height() / 2,
                    f"{value:.3f}", va="center", ha="right",
                    fontsize=8, color="#2867A8")

    fig.tight_layout(pad=1.8)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight",
                facecolor="white", dpi=170)
    plt.close(fig)
    chart = base64.b64encode(buffer.getvalue()).decode("utf-8")

    explanation_items = []
    ranked = sorted(range(len(shap_row)),
                    key=lambda index: abs(shap_row[index]), reverse=True)[:8]

    for index in ranked:
        shap_value = float(shap_row[index])
        if abs(shap_value) < 1e-12:
            continue

        name = columns[index]
        raw_value = features[0][index]
        direction = "increase" if shap_value > 0 else "decrease"

        explanation_items.append({
            "name": FEATURE_LABELS.get(name, name),
            "value": feature_value_text(name, raw_value, contract_value),
            "direction": direction,
            "direction_label": (
                "Pushes churn higher" if shap_value > 0
                else "Pushes churn lower"
            ),
            "detail": explain_feature(name, raw_value, direction, contract_value),
            "impact": abs(shap_value),
        })

    return chart, explanation_items


@app.route("/")
def home():
    return render_template("index.html", form_data={}, result=None)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        gender = 0 if request.form.get("gender") == "0" else 1

        senior_citizen = checkbox("SeniorCitizen")
        partner = checkbox("Partner")
        dependents = checkbox("Dependents")
        paperless_billing = checkbox("PaperlessBilling")

        monthly_charges = float(request.form.get("MonthlyCharges", ""))
        tenure = int(request.form.get("Tenure", ""))

        if not np.isfinite(monthly_charges) or monthly_charges < 0:
            raise ValueError("Enter a valid non-negative monthly charge.")
        if tenure < 0 or tenure > 72:
            raise ValueError("Tenure must be between 0 and 72 months.")

        # Retains the feature construction used in the original project.
        total_charges = monthly_charges * tenure

        phone_service = checkbox("PhoneService")
        multiple_lines = checkbox("MultipleLines") if phone_service else 0

        internet_service = request.form.get("InternetService", "0")
        internet_no = 1 if internet_service == "0" else 0
        internet_fiber = 1 if internet_service == "2" else 0
        has_internet = internet_no == 0

        online_security = checkbox("OnlineSecurity") if has_internet else 0
        online_backup = checkbox("OnlineBackup") if has_internet else 0
        device_protection = checkbox("DeviceProtection") if has_internet else 0
        tech_support = checkbox("TechSupport") if has_internet else 0
        streaming_tv = checkbox("StreamingTV") if has_internet else 0
        streaming_movies = checkbox("StreamingMovies") if has_internet else 0

        contract = request.form.get("Contract", "0")
        contract_one_year = 1 if contract == "1" else 0
        contract_two_year = 1 if contract == "2" else 0

        payment = request.form.get("PaymentMethod", "0")
        payment_credit_card = 1 if payment == "1" else 0
        payment_electronic_check = 1 if payment == "2" else 0
        payment_mailed_check = 1 if payment == "3" else 0

        features = [
            gender, senior_citizen, partner, dependents, tenure,
            phone_service, multiple_lines, online_security, online_backup,
            device_protection, tech_support, streaming_tv, streaming_movies,
            paperless_billing, monthly_charges, total_charges,
            internet_fiber, internet_no, contract_one_year, contract_two_year,
            payment_credit_card, payment_electronic_check, payment_mailed_check
        ]

        columns = [
            "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
            "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
            "PaperlessBilling", "MonthlyCharges", "TotalCharges",
            "InternetService_Fiber optic", "InternetService_No",
            "Contract_One year", "Contract_Two year",
            "PaymentMethod_Credit card (automatic)",
            "PaymentMethod_Electronic check", "PaymentMethod_Mailed check"
        ]

        feature_array = np.asarray([features], dtype=float)
        probability = float(model.predict_proba(feature_array)[0, 1])
        gauge_image = make_gauge(probability)

        shap_image = None
        explanation_items = []
        retention_recommendations = []
        explanation_error = None

        try:
            shap_image, explanation_items = make_shap_explanation(
                feature_array, columns, contract_value=contract
            )
            retention_recommendations = make_retention_recommendations(
                feature_array, columns, explanation_items, contract
            )
        except Exception as exc:
            # Keep the prediction available even if SHAP rendering fails.
            explanation_error = str(exc)
            app.logger.exception("Could not create SHAP explanation")

        return render_template(
            "index.html",
            form_data=request.form,
            result=probability,
            gauge_image=gauge_image,
            shap_image=shap_image,
            explanation_items=explanation_items,
            retention_recommendations=retention_recommendations,
            explanation_error=explanation_error
        )

    except (ValueError, TypeError) as exc:
        return render_template(
            "index.html",
            form_data=request.form,
            error=str(exc),
            result=None
        ), 400


if __name__ == "__main__":
    app.run(debug=True)
