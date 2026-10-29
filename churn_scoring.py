
import pickle
import pandas as pd
import shap


class ChurnScorer:
    def __init__(self, model_path="churn_model_bundle.pkl"):
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)

        self.model = bundle["model"]
        self.threshold = bundle["threshold"]
        self.final_features = bundle["final_features"]
        self.dummy_groups = bundle["dummy_groups"]
        self.category_values = bundle["category_values"]
        self.binary_features = bundle["binary_features"]
        self.what_if_features = bundle["what_if_features"]
        self.tenure_max = bundle["tenure_max"]

        self.explainer = shap.Explainer(self.model)

    def score(self, customer_x):
        customer_x = customer_x[self.final_features].astype(float)

        probability = self.model.predict_proba(customer_x)[0, 1]
        prediction = "Churn" if probability >= self.threshold else "No Churn"

        shap_values = self.explainer(customer_x)

        driver_table = pd.DataFrame({
            "Feature": self.final_features,
            "Customer Value": customer_x.iloc[0].values,
            "SHAP Impact": shap_values.values[0]
        })

        risk_drivers = (
            driver_table[driver_table["SHAP Impact"] > 0]
            .nlargest(3, "SHAP Impact")
            .reset_index(drop=True)
        )

        protective_drivers = (
            driver_table[driver_table["SHAP Impact"] < 0]
            .nsmallest(3, "SHAP Impact")
            .reset_index(drop=True)
        )

        return {
            "probability": probability,
            "prediction": prediction,
            "threshold": self.threshold,
            "shap_values": shap_values,
            "risk_drivers": risk_drivers,
            "protective_drivers": protective_drivers
        }

    def apply_scenario(self, customer_x, feature_name, value):
        scenario = customer_x.copy()

        if feature_name == "tenure":
            tenure_value = float(value)
            scenario.loc[:, "tenure"] = tenure_value

            if "IsNewCustomer" in self.final_features:
                scenario.loc[:, "IsNewCustomer"] = int(tenure_value <= 6)

            return scenario

        if feature_name in self.dummy_groups:
            group_cols = self.dummy_groups[feature_name]
            scenario.loc[:, group_cols] = 0

            expected_col = f"{feature_name}_{value}"

            if expected_col in group_cols:
                scenario.loc[:, expected_col] = 1

            return scenario

        if feature_name in self.binary_features:
            scenario.loc[:, feature_name] = int(value)
            return scenario

        raise ValueError(f"Unsupported feature: {feature_name}")
