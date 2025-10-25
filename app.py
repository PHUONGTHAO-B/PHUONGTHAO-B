from flask import Flask, render_template, request
import joblib
import pandas as pd
import sys
import traceback

app = Flask(__name__)

# 🔹 Load duy nhất 1 file mô hình
try:
    model_package = joblib.load('SVM_model.pkl')
except FileNotFoundError as e:
    print(f"Lỗi: Không tìm thấy file mô hình. {str(e)}")
    sys.exit(1)
except Exception as e:
    print("Lỗi khi tải mô hình:", e)
    traceback.print_exc()
    sys.exit(1)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error_message = None
    if request.method == "POST":
        try:
            # Lấy dữ liệu từ form
            N = float(request.form["N"])
            P = float(request.form["P"])
            K = float(request.form["K"])
            temperature = float(request.form["temperature"])
            humidity = float(request.form["humidity"])
            ph = float(request.form["ph"])
            rainfall = float(request.form["rainfall"])

            # Đóng gói dữ liệu vào DataFrame
            input_data = pd.DataFrame(
                [[N, P, K, temperature, humidity, ph, rainfall]],
                columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
            )

            # Dự đoán trực tiếp bằng pipeline bên trong model_package
            pred_index = model_package['pipeline'].predict(input_data)[0]

            # Chuyển nhãn số sang tên cây
            prediction = model_package['label_encoder'].inverse_transform([pred_index])[0]

        except (ValueError, KeyError):
            error_message = "Vui lòng nhập đầy đủ và đúng định dạng các thông số!"
        except Exception as e:
            error_message = f"Lỗi không xác định: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction,
        error_message=error_message,
        accuracy=model_package.get('accuracy', None)
    )

if __name__ == "__main__":
    app.run(debug=True)
