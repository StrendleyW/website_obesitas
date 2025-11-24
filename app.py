from flask import Flask, render_template, request
import joblib
import pickle
import pandas as pd

app = Flask(__name__)

# ============= LOAD ARTEFAK =============
model = joblib.load("model_knn.pkl")
encoders = pickle.load(open("encoders.pkl", "rb"))          # dict: col -> LabelEncoder
target_encoder = pickle.load(open("label_target.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
selected_features = pickle.load(open("selected_features.pkl", "rb"))  # list fitur IG terpilih
feature_order = pickle.load(open("features.pkl", "rb"))               # semua fitur X saat training

# Sesuaikan ini dengan yang kamu pakai di training
CAT_COLS = [
    "Gender",
    "family_history_with_overweight",
    "FAVC",
    "CAEC",
    "SMOKE",
    "SCC",
    "CALC",
    "MTRANS",
]


def preprocess_form(form):
    """
    Ambil input dari form → DataFrame 1 baris
    → encode kategori → pilih fitur IG → scale → siap ke model.
    """
    row = []

    # 1) Pastikan semua fitur di feature_order diisi dari form
    for col in feature_order:
        raw = form.get(col)
        if raw is None or raw == "":
            raise ValueError(f"Fitur '{col}' tidak boleh kosong")

        if col in CAT_COLS:
            # kategorikal → pakai encoder
            if col not in encoders:
                raise ValueError(f"Tidak ada encoder untuk kolom '{col}'")

            le = encoders[col]
            if raw not in le.classes_:
                raise ValueError(
                    f"Nilai '{raw}' tidak dikenal untuk kolom '{col}'. "
                    f"Pilihan valid: {list(le.classes_)}"
                )

            encoded_val = le.transform([raw])[0]
            row.append(encoded_val)
        else:
            # numerik
            try:
                val_num = float(raw)
            except ValueError:
                raise ValueError(f"Fitur '{col}' harus berupa angka, dapat: {raw}")
            row.append(val_num)

    # 2) Buat DataFrame dengan urutan kolom yang sama persis
    df_input = pd.DataFrame([row], columns=feature_order)

    # 3) Ambil hanya fitur yang dipilih Information Gain
    df_selected = df_input[selected_features]

    # 4) Scale pakai scaler yang sama (scaler di-fit di X_train_final dengan kolom selected_features)
    scaled_array = scaler.transform(df_selected)
    df_scaled = pd.DataFrame(scaled_array, columns=selected_features)

    return df_scaled


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_label = None
    error_message = None

    if request.method == "POST":
        try:
            X_input = preprocess_form(request.form)
            y_pred = model.predict(X_input)[0]
            obesity_label = target_encoder.inverse_transform([y_pred])[0]
            prediction_label = obesity_label
        except Exception as e:
            error_message = str(e)

    return render_template("index.html", prediction=prediction_label, error=error_message)


if __name__ == "__main__":
    # Untuk development
    app.run(debug=True)
