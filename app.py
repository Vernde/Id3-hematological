
import math
import io
import re
from collections import Counter

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="CBC Classification System",
    page_icon="🩸",
    layout="wide",
)

FEATURES = [
    "WBC", "LYMp", "NEUTp", "LYMn", "NEUTn",
    "RBC", "HGB", "HCT", "MCV", "MCH",
    "MCHC", "PLT", "PDW", "PCT",
]

# Dataset-label grouping for the proposed display classes. This is a
# prototype mapping, not a clinically validated risk score.
RISK_CLASSES = {
    "Healthy": {
        "class": "Class 1 — Low Risk",
        "level": "Low",
        "description": (
            "The dataset label is Healthy. This does not establish that "
            "a person is clinically healthy."
        ),
    },
    "Normocytic hypochromic anemia": {
        "class": "Class 2 — Moderate Risk",
        "level": "Moderate",
        "description": "The predicted dataset category is an anemia label and requires professional evaluation.",
    },
    "Iron deficiency anemia": {
        "class": "Class 2 — Moderate Risk",
        "level": "Moderate",
        "description": "The predicted dataset category is an anemia label and requires professional evaluation.",
    },
    "Other microcytic anemia": {
        "class": "Class 2 — Moderate Risk",
        "level": "Moderate",
        "description": "The predicted dataset category is an anemia label and requires professional evaluation.",
    },
    "Normocytic normochromic anemia": {
        "class": "Class 2 — Moderate Risk",
        "level": "Moderate",
        "description": "The predicted dataset category is an anemia label and requires professional evaluation.",
    },
    "Macrocytic anemia": {
        "class": "Class 2 — Moderate Risk",
        "level": "Moderate",
        "description": "The predicted dataset category is an anemia label and requires professional evaluation.",
    },
    "Leukemia": {
        "class": "Class 3 — High Risk",
        "level": "High",
        "description": "The predicted dataset category warrants prompt assessment by a qualified professional.",
    },
    "Thrombocytopenia": {
        "class": "Class 3 — High Risk",
        "level": "High",
        "description": "The predicted dataset category warrants prompt assessment by a qualified professional.",
    },
    "Leukemia with thrombocytopenia": {
        "class": "Class 3 — High Risk",
        "level": "High",
        "description": "The predicted dataset category warrants prompt assessment by a qualified professional.",
    },
}


# --------------------------------------------------
# NUMERICAL ID3 ALGORITHM
# --------------------------------------------------

def entropy(labels):
    counts = Counter(labels)
    total = len(labels)

    return -sum(
        (count / total) * math.log2(count / total)
        for count in counts.values()
    )


def information_gain(data, feature, target, threshold):
    left = data[data[feature] <= threshold]
    right = data[data[feature] > threshold]

    if left.empty or right.empty:
        return 0.0

    parent_entropy = entropy(data[target])

    weighted_entropy = (
        (len(left) / len(data)) * entropy(left[target])
        + (len(right) / len(data)) * entropy(right[target])
    )

    return parent_entropy - weighted_entropy


def best_split(data, features, target):
    best_split_result = None
    best_gain = 0.0

    for feature in features:
        values = sorted(data[feature].unique())

        for i in range(len(values) - 1):
            threshold = (values[i] + values[i + 1]) / 2

            gain = information_gain(
                data, feature, target, threshold
            )

            if gain > best_gain:
                best_gain = gain
                best_split_result = (feature, threshold)

    return best_split_result


def build_tree(data, features, target, depth=0, max_depth=10):
    if data.empty:
        return None

    if data[target].nunique() == 1:
        return data[target].iloc[0]

    majority_class = data[target].mode().iloc[0]

    if depth >= max_depth or not features:
        return majority_class

    split = best_split(data, features, target)

    if split is None:
        return majority_class

    feature, threshold = split

    left = data[data[feature] <= threshold]
    right = data[data[feature] > threshold]

    if left.empty or right.empty:
        return majority_class

    return {
        "feature": feature,
        "threshold": threshold,
        "default": majority_class,
        "left": build_tree(
            left, features, target, depth + 1, max_depth
        ),
        "right": build_tree(
            right, features, target, depth + 1, max_depth
        ),
    }


def predict(tree, record):
    if tree is None:
        raise ValueError("The decision tree could not make a prediction.")

    while isinstance(tree, dict):
        feature = tree["feature"]

        if feature not in record:
            raise ValueError(f"Missing input feature: {feature}")

        if record[feature] <= tree["threshold"]:
            tree = tree["left"]
        else:
            tree = tree["right"]

    return tree


def leaf_training_agreement(tree, record, training_data, target, prediction):
    """Return the matching-label share among training rows at this leaf."""
    leaf_data = training_data
    while isinstance(tree, dict):
        feature = tree["feature"]
        if record[feature] <= tree["threshold"]:
            leaf_data = leaf_data[leaf_data[feature] <= tree["threshold"]]
            tree = tree["left"]
        else:
            leaf_data = leaf_data[leaf_data[feature] > tree["threshold"]]
            tree = tree["right"]

    if leaf_data.empty:
        return 0.0
    return float((leaf_data[target] == prediction).mean())


REPORT_LABELS = {
    "WBC": ("WHITE BLOOD CELLS", "WBC"),
    "LYMp": ("LYMPHOCYTE PERCENT", "LYMPHOCYTES %", "LYMPHOCYTE %", "LYMPHOCYTE", "LYMP %", "LYMP"),
    "NEUTp": ("NEUTROPHIL PERCENT", "NEUTROPHILS %", "NEUTROPHIL %", "NEUTROPHIL", "NEUT %", "NEUT"),
    "LYMn": ("ABSOLUTE LYMPHOCYTE", "LYMPHOCYTES #", "LYMPHOCYTE #", "LYMPHOCYTE COUNT", "LYM #", "LYM#"),
    "NEUTn": ("ABSOLUTE NEUTROPHIL", "NEUTROPHILS #", "NEUTROPHIL #", "NEUTROPHIL COUNT", "NEUT #", "NEUT#"),
    "RBC": ("RED BLOOD CELLS", "RBC"),
    "HGB": ("HEMOGLOBIN", "HGB", "HB"),
    "HCT": ("HEMATOCRIT", "HCT", "PCV"),
    "MCV": ("MCV",),
    "MCH": ("MCH",),
    "MCHC": ("MCHC",),
    "PLT": ("PLATELETS", "PLATELET", "PLT"),
    "PDW": ("PDW",),
    "PCT": ("PCT",),
}


def extract_cbc_values(report_text):
    """Read CBC feature values from labeled report lines when possible."""
    extracted = {}
    numeric_pattern = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)")

    # Longer labels first so, for example, MCHC is not interpreted as MCH.
    label_patterns = [
        (feature, re.compile(rf"(?<![A-Z]){re.escape(label)}(?![A-Z])"))
        for feature, labels in REPORT_LABELS.items()
        for label in sorted(labels, key=len, reverse=True)
    ]
    label_patterns.sort(key=lambda item: len(item[1].pattern), reverse=True)

    for line in report_text.upper().splitlines():
        for feature, pattern in label_patterns:
            match = pattern.search(line)
            if not match or feature in extracted:
                continue

            suffix = line[match.end():]
            for value_match in numeric_pattern.finditer(suffix):
                # Ignore the 10 in units such as 10^9/L before trying the result.
                following_text = suffix[value_match.end():].lstrip()
                if following_text.startswith(("^", "**", "/")):
                    continue
                extracted[feature] = float(value_match.group())
                break

    return extracted


def read_report(uploaded_file):
    """Extract report text from an image using Tesseract or a text PDF."""
    file_bytes = uploaded_file.getvalue()
    suffix = uploaded_file.name.rsplit(".", 1)[-1].lower()

    if suffix == "pdf":
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise ValueError("PDF support requires the pypdf package.") from error

        reader = PdfReader(io.BytesIO(file_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if not text.strip():
            raise ValueError(
                "This PDF has no selectable text. Upload a clear image "
                "of the report instead."
            )
        return text

    try:
        from PIL import Image
        import pytesseract
    except ImportError as error:
        raise ValueError(
            "Image reading requires Pillow and pytesseract. Install the "
            "project requirements, then install the Tesseract OCR application."
        ) from error

    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return pytesseract.image_to_string(image)
    except Exception as error:
        raise ValueError(
            "Could not read this image. Check that Tesseract OCR is installed "
            "and upload a clear, well-lit report image."
        ) from error


# --------------------------------------------------
# LOAD DATASET AND TRAIN BASELINE MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    data = pd.read_csv("diagnosed_cbc_screened.csv")
    data.columns = data.columns.astype(str).str.strip()

    required_columns = FEATURES + ["Diagnosis"]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing dataset columns: {missing_columns}"
        )

    data = data[required_columns].copy()

    for feature in FEATURES:
        data[feature] = pd.to_numeric(
            data[feature], errors="coerce"
        )

    data = data.dropna()

    if data["Diagnosis"].nunique() < 2:
        raise ValueError(
            "The dataset needs at least two diagnosis classes."
        )

    train_data, test_data = train_test_split(
        data,
        test_size=0.25,
        random_state=42,
        stratify=data["Diagnosis"],
    )

    tree = build_tree(
        train_data,
        FEATURES,
        "Diagnosis",
        max_depth=10,
    )

    test_predictions = [
        predict(tree, row)
        for row in test_data[FEATURES].to_dict("records")
    ]
    test_accuracy = sum(
        actual == predicted
        for actual, predicted in zip(test_data["Diagnosis"], test_predictions)
    ) / len(test_data)

    feature_ranges = {
        feature: (float(data[feature].min()), float(data[feature].max()))
        for feature in FEATURES
    }

    return tree, train_data, len(data), len(test_data), test_accuracy, feature_ranges


# --------------------------------------------------
# USER INTERFACE
# --------------------------------------------------

st.title("CBC-Based Hematological Classification")
st.caption("Numerical ID3 Baseline | Academic Research Prototype")

st.warning(
    "RESEARCH PROTOTYPE ONLY: This system has not been clinically "
    "validated. Risk classes are a preliminary grouping of dataset "
    "labels, not a clinically validated risk score. Do not use its "
    "predictions to diagnose patients or guide medical treatment."
)

try:
    tree, training_data, dataset_count, testing_count, test_accuracy, feature_ranges = load_model()

except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.info(
        "Check that diagnosed_cbc_screened.csv is located in "
        "the same folder as app.py."
    )
    st.stop()


with st.expander("About this system"):
    st.write(
        "This prototype uses a numerical ID3-style decision tree "
        "to classify CBC measurements into a category found in "
        "the research dataset."
    )

    st.write(f"Dataset records loaded: {dataset_count}")
    st.write(f"Records used to train the tree: {len(training_data)}")
    st.write(f"Records used for held-out evaluation: {testing_count}")
    st.write(f"Input features: {len(FEATURES)}")

with st.expander("Proposed hematological risk classes"):
    st.write(
        "Risk class is assigned from the predicted dataset diagnosis "
        "using the mapping below. It is not calculated from validated "
        "clinical thresholds."
    )
    st.markdown(
        "- **Class 1 — Low Risk:** Healthy\n"
        "- **Class 2 — Moderate Risk:** the five anemia labels in this dataset\n"
        "- **Class 3 — High Risk:** Leukemia, Thrombocytopenia, and "
        "Leukemia with thrombocytopenia"
    )

st.divider()
st.subheader("Enter CBC Measurements")

st.subheader("Upload a CBC report")
st.write(
    "Upload a clear image of a hemogram/CBC report or a PDF with "
    "selectable text. Extracted values will be placed in the fields "
    "below so you can review and correct them."
)
uploaded_report = st.file_uploader(
    "CBC report image or PDF",
    type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "pdf"],
    help="Image OCR requires Tesseract OCR to be installed on the computer running this app.",
)

if uploaded_report is not None and st.button("Read CBC report"):
    try:
        report_text = read_report(uploaded_report)
        extracted_values = extract_cbc_values(report_text)
        if not extracted_values:
            st.error(
                "No CBC measurements could be identified. Use a clearer "
                "image or enter the values manually."
            )
        else:
            st.session_state.report_extracted_values = extracted_values
            for feature in FEATURES:
                st.session_state[f"{feature}_{st.session_state.get('cbc_reset_count', 0)}"] = extracted_values.get(feature, 0.0)
    except Exception as error:
        st.error(f"Could not read the report: {error}")

if st.session_state.get("report_extracted_values"):
    extracted_values = st.session_state.report_extracted_values
    st.write("Review the extracted CBC values below; edit any incorrect or missing values before predicting.")
    st.dataframe(
        pd.DataFrame(
            {
                "Measurement": FEATURES,
                "Extracted value": [extracted_values.get(feature) for feature in FEATURES],
                "Status": ["Read" if feature in extracted_values else "Not found" for feature in FEATURES],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

st.write(
    "Enter the measurements from a CBC record using the "
    "same units and conventions as the dataset."
)

# Initialize inputs for the reset button.
if "cbc_reset_count" not in st.session_state:
    st.session_state.cbc_reset_count = 0

reset_key = st.session_state.cbc_reset_count
for feature in FEATURES:
    st.session_state.setdefault(f"{feature}_{reset_key}", 0.0)

with st.form(key=f"cbc_form_{reset_key}"):

    columns = st.columns(3)
    values = {}

    for index, feature in enumerate(FEATURES):
        with columns[index % 3]:
            values[feature] = st.number_input(
                feature,
                min_value=0.0,
                format="%.4f",
                key=f"{feature}_{reset_key}",
            )

    predict_clicked = st.form_submit_button(
        "Predict Classification",
        type="primary",
        use_container_width=True,
    )


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

if predict_clicked:
    st.divider()
    st.subheader("Prediction Results")

    try:
        if all(value == 0 for value in values.values()):
            raise ValueError(
                "all CBC measurements are 0. Enter the measurements "
                "from a CBC record and try again."
            )

        outside_dataset = [
            f"{feature} ({values[feature]:g}; dataset range "
            f"{minimum:g}–{maximum:g})"
            for feature, (minimum, maximum) in feature_ranges.items()
            if not minimum <= values[feature] <= maximum
        ]
        if outside_dataset:
            raise ValueError(
                "one or more measurements are outside the values "
                "observed in diagnosed_cbc_screened.csv: "
                + ", ".join(outside_dataset)
                + ". Check the units and values, then try again."
            )

        prediction = predict(tree, values)

        st.success(
            f"Predicted dataset class: {prediction}"
        )

        leaf_agreement = leaf_training_agreement(
            tree, values, training_data, "Diagnosis", prediction
        )
        score_columns = st.columns(2)
        score_columns[0].metric("Held-out test accuracy", f"{test_accuracy:.1%}")
        score_columns[1].metric("Leaf agreement estimate", f"{leaf_agreement:.1%}")
        st.caption(
            f"Held-out accuracy uses {testing_count} records from the dataset. "
            "Leaf agreement is the share of training records reaching the same "
            "tree leaf that have this predicted label. It is not a calibrated "
            "probability or a measure of clinical certainty."
        )

        risk = RISK_CLASSES.get(str(prediction))
        if risk:
            st.subheader(risk["class"])
            st.write(f"**Risk level:** {risk['level']}")
            st.write(risk["description"])
        else:
            st.info(
                "No risk class is configured for this dataset label. "
                "Review the label mapping before interpreting the result."
            )

        st.caption(
            "This is the model's predicted dataset category, "
            "not a confirmed medical diagnosis."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")


# --------------------------------------------------
# RESET INPUTS
# --------------------------------------------------

if st.button("Reset Inputs", use_container_width=True):
    st.session_state.cbc_reset_count += 1
    st.session_state.pop("report_extracted_values", None)
    st.rerun()


st.divider()

st.caption(
    "Academic research use only | Numerical ID3 baseline | "
    "Clinical validity has not been established."
)
