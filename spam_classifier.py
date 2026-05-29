import pandas as pd
import customtkinter as ctk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv("spam.csv", encoding='latin-1')

df = df[['v1', 'v2']]
df.columns = ['label', 'message']

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# =====================================================
# TRAIN MODEL
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    df['message'],
    df['label'],
    test_size=0.2,
    random_state=42
)

vectorizer = TfidfVectorizer()

X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

model = MultinomialNB()

model.fit(X_train_features, y_train)

predictions = model.predict(X_test_features)

accuracy = accuracy_score(y_test, predictions)

# =====================================================
# APP SETTINGS
# =====================================================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# =====================================================
# MAIN WINDOW
# =====================================================

app = ctk.CTk()

app.title("Spam Email Classifier")

# FULL SCREEN FIX
app.state("zoomed")

# Background Color
app.configure(fg_color="#0A0F1C")

# =====================================================
# SCROLLABLE FRAME
# =====================================================

main_frame = ctk.CTkScrollableFrame(
    app,
    fg_color="#0A0F1C"
)

main_frame.pack(fill="both", expand=True)

# =====================================================
# HEADER SECTION
# =====================================================

title = ctk.CTkLabel(
    main_frame,
    text="SPAM EMAIL CLASSIFIER",
    font=("Arial", 42, "bold"),
    text_color="#ff4fd8"
)

title.pack(pady=(30, 10))

subtitle = ctk.CTkLabel(
    main_frame,
    text="Smart ML Model • Detect Spam Instantly",
    font=("Arial", 22),
    text_color="white"
)

subtitle.pack(pady=(0, 30))

# =====================================================
# INPUT FRAME
# =====================================================

input_frame = ctk.CTkFrame(
    main_frame,
    width=1100,
    height=420,
    corner_radius=25,
    fg_color="#071129",
    border_width=3,
    border_color="#3b82f6"
)

input_frame.pack(pady=20, padx=30)

input_title = ctk.CTkLabel(
    input_frame,
    text="ENTER YOUR MESSAGE",
    font=("Arial", 30, "bold"),
    text_color="#3b82f6"
)

input_title.pack(anchor="w", padx=30, pady=(25, 20))

# =====================================================
# TEXT BOX
# =====================================================

text_box = ctk.CTkTextbox(
    input_frame,
    width=950,
    height=200,
    corner_radius=20,
    border_width=3,
    border_color="#ff4fd8",
    font=("Arial", 20),
    text_color="white",
    fg_color="#111111"
)

text_box.pack(padx=25, pady=10)

# =====================================================
# PREDICT FUNCTION
# =====================================================

def predict_message():

    message = text_box.get("1.0", "end").strip()

    if message == "":

        result_label.configure(
            text="⚠ PLEASE ENTER A MESSAGE",
            text_color="yellow"
        )

        confidence_label.configure(text="")

        return

    message_vector = vectorizer.transform([message])

    prediction = model.predict(message_vector)

    probability = model.predict_proba(message_vector)

    confidence = max(probability[0]) * 100

    if prediction[0] == 1:

        result_label.configure(
            text="🚨 SPAM MESSAGE DETECTED",
            text_color="#ff4d4d"
        )

        result_frame.configure(
            border_color="#ff4d4d"
        )

        confidence_label.configure(
            text=f"Threat Score : {confidence:.2f}%",
            text_color="#ff4d4d"
        )

    else:

        result_label.configure(
            text="✅ MESSAGE IS SAFE",
            text_color="#00ff99"
        )

        result_frame.configure(
            border_color="#00ff99"
        )

        confidence_label.configure(
            text=f"Confidence Score : {confidence:.2f}%",
            text_color="#00ff99"
        )

# =====================================================
# PREDICT BUTTON
# =====================================================

predict_button = ctk.CTkButton(
    input_frame,
    text="⚡ PREDICT MESSAGE",
    font=("Arial", 22, "bold"),
    width=320,
    height=65,
    corner_radius=20,
    fg_color="#ff4fd8",
    hover_color="#3b82f6",
    command=predict_message
)

predict_button.pack(pady=30)

# =====================================================
# RESULT FRAME
# =====================================================

result_frame = ctk.CTkFrame(
    main_frame,
    width=1100,
    height=260,
    corner_radius=25,
    fg_color="#071129",
    border_width=3,
    border_color="#00ff99"
)

result_frame.pack(pady=25, padx=30)

result_title = ctk.CTkLabel(
    result_frame,
    text="AI DETECTION RESULT",
    font=("Arial", 30, "bold"),
    text_color="#3b82f6"
)

result_title.pack(pady=(30, 10))

result_label = ctk.CTkLabel(
    result_frame,
    text="Waiting For Prediction...",
    font=("Arial", 34, "bold"),
    text_color="white"
)

result_label.pack(pady=10)

confidence_label = ctk.CTkLabel(
    result_frame,
    text="",
    font=("Arial", 22),
    text_color="#00ff99"
)

confidence_label.pack(pady=10)

# =====================================================
# FOOTER SECTION
# =====================================================

footer_frame = ctk.CTkFrame(
    main_frame,
    fg_color="transparent"
)

footer_frame.pack(pady=40)

dataset_info = ctk.CTkLabel(
    footer_frame,
    text="📂 Dataset: SMS Spam Collection",
    font=("Arial", 18),
    text_color="#3b82f6"
)

dataset_info.grid(row=0, column=0, padx=30)

model_info = ctk.CTkLabel(
    footer_frame,
    text="🧠 Model: Naive Bayes",
    font=("Arial", 18),
    text_color="#ff4fd8"
)

model_info.grid(row=0, column=1, padx=30)

accuracy_info = ctk.CTkLabel(
    footer_frame,
    text=f"📈 Accuracy: {accuracy*100:.2f}%",
    font=("Arial", 18),
    text_color="#00ff99"
)

accuracy_info.grid(row=0, column=2, padx=30)

status_info = ctk.CTkLabel(
    footer_frame,
    text="✅ Status: Model Loaded",
    font=("Arial", 18),
    text_color="white"
)

status_info.grid(row=0, column=3, padx=30)

# =====================================================
# RUN APPLICATION
# =====================================================

app.mainloop()