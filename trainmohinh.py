# ================================================================
# 🎯 DỰ ĐOÁN CÂY TRỒNG PHÙ HỢP DỰA TRÊN THÔNG SỐ ĐẤT & MÔI TRƯỜNG
# Sử dụng mô hình Support Vector Machine (SVM)
# ================================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import joblib

# ------------------------------------------------
# 1️⃣ Đọc dữ liệu từ file CSV
# ------------------------------------------------
data = pd.read_csv("Crop_recommendation.csv")

# Hiển thị vài dòng đầu tiên để kiểm tra
print("📂 Dữ liệu mẫu:")
print(data.head())

# ------------------------------------------------
# 2️⃣ Tách đặc trưng (features) và nhãn (label)
# ------------------------------------------------
# 'label' là tên cây trồng cần dự đoán
X = data.drop("label", axis=1)   # Đặc trưng: N, P, K, temperature, humidity, ph, rainfall
y = data["label"]                # Nhãn: loại cây trồng

# ------------------------------------------------
# 3️⃣ Mã hóa nhãn cây trồng thành số (Label Encoding)
# ------------------------------------------------
# Ví dụ: rice -> 0, maize -> 1, chickpea -> 2, ...
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ------------------------------------------------
# 4️⃣ Chia dữ liệu thành tập huấn luyện và kiểm tra
# ------------------------------------------------
# 80% dùng để huấn luyện, 20% để đánh giá mô hình
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# ------------------------------------------------
# 5️⃣ Tạo pipeline gồm 2 bước:
#     - Chuẩn hóa dữ liệu bằng StandardScaler
#     - Huấn luyện mô hình SVM (kernel RBF)
# ------------------------------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),               # Bước 1: chuẩn hóa dữ liệu
    ("svm", SVC(kernel='rbf', C=10, gamma='scale', probability=True))  # Bước 2: mô hình SVM
])

# ------------------------------------------------
# 6️⃣ Huấn luyện mô hình
# ------------------------------------------------
print("\n🚀 Đang huấn luyện mô hình SVM...")
pipeline.fit(X_train, y_train)
print("✅ Huấn luyện hoàn tất!")

# ------------------------------------------------
# 7️⃣ Đánh giá độ chính xác mô hình
# ------------------------------------------------
accuracy = pipeline.score(X_test, y_test)
print(f"🎯 Độ chính xác mô hình trên tập kiểm tra: {accuracy * 100:.2f}%")

# ------------------------------------------------
# 8️⃣ Lưu mô hình và bộ mã hóa vào một file duy nhất
# ------------------------------------------------
# Gộp cả pipeline (chuẩn hóa + SVM) và label_encoder (giải mã tên cây)
model_package = {
    "pipeline": pipeline,
    "label_encoder": label_encoder
}

joblib.dump(model_package, "SVM_model.pkl")

print("\n💾 Đã lưu mô hình vào file: SVM_model.pkl")
print("Bạn có thể nạp lại file này để dự đoán trong Flask hoặc Streamlit.")
