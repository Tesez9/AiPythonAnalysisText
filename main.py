import csv
from transformers import pipeline

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

analyzer = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME
)

reviews = [
    "The application works perfectly. I use it every day.",
    "The delivery was late, and the package was damaged.",
    "The product is acceptable, but nothing special.",
    "Excellent customer support and fast service!",
    "The software crashes every time I open it.",
    "The headphones have clear sound and comfortable ear cushions.",
    "I am disappointed with the quality of this product.",
    "The website is easy to navigate and looks modern.",
    "The food arrived cold and tasted terrible.",
    "The staff were friendly and helped me solve the problem.",
    "The battery lasts much longer than I expected.",
    "This is the worst purchase I have made this year.",
    "The service was fine, but the price was a little high.",
    "Setup was simple, and everything worked on the first try.",
    "The instructions are confusing and difficult to follow.",
    "I love the design, and the product feels very durable.",
    "The order arrived two days earlier than expected.",
    "The camera quality is poor in low light.",
    "Nothing was wrong with the item, but it did not impress me.",
    "Customer support ignored my messages for several days.",
    "The new update fixed the bugs and improved performance.",
    "The chair is uncomfortable after sitting for an hour.",
    "The subscription is affordable and offers great value.",
    "The app is slow and drains my phone battery.",
    "The hotel room was clean, quiet, and comfortable.",
    "The product stopped working after only one week.",
    "The packaging was neat, and the item matched the description.",
    "The meal was okay, although the portions were small.",
    "I received the wrong item and had to return it.",
    "The course was informative, engaging, and well organized.",
    "The printer works as expected, with no major issues.",
    "I would not recommend this service to anyone.",
    "The screen is bright, sharp, and easy to read.",
    "The repair took longer than promised, but the result is good.",
    "The price is reasonable for the features provided.",
    "The instructions were clear, and installation took ten minutes.",
    "The sound quality is average, and the controls feel cheap.",
    "I had a wonderful experience from start to finish.",
    "The product looks nice, but it does not work properly.",
    "The support team resolved my issue quickly and professionally."
]

def analyze_review(text):
    if not text.strip():
        print("Помилка: текст відгуку не може бути порожнім.")
        return None
    result = analyzer(text)[0]
    return {
        "text": text,
        "label": result["label"],
        "confidence": result["score"]
    }

def translate_label(label):
    if label == "POSITIVE":
        return "ПОЗИТИВНИЙ"
    elif label == "NEGATIVE":
        return "НЕГАТИВНИЙ"
    else:
        return label

def analyze_one():
    print("\n===== ANALYZE ONE REVIEW =====")
    text = input("Введіть текст відгуку: ")
    result = analyze_review(text)
    if result is None:
        return
    print("\n===== RESULT =====")
    print("Текст:", result["text"])
    print("Тональність:", translate_label(result["label"]))
    print("Впевненість моделі:", f"{result['confidence'] * 100:.2f}%")

def analyze_all():
    results = []
    positive = 0
    negative = 0
    print("\n===== REVIEW ANALYSIS =====")
    for i, review in enumerate(reviews, 1):
        result = analyze_review(review)
        if result is None:
            continue
        results.append(result)
        if result["label"] == "POSITIVE":
            positive += 1
        elif result["label"] == "NEGATIVE":
            negative += 1
        print(f"\n{i}. {review}")
        print("Тональність:", translate_label(result["label"]))
        print("Впевненість моделі:", f"{result['confidence'] * 100:.2f}%")
    print("\n================================")
    print("          STATISTICS")
    print("================================")
    print("Загальна кількість:", len(results))
    print("Позитивних:", positive)
    print("Негативних:", negative)
    print("Нейтральних: 0")
    print("Примітка: обрана модель не підтримує нейтральний клас.")
    save_results(results)

def save_results(results):
    with open("results.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "text",
            "sentiment",
            "confidence"
        ])
        for result in results:
            writer.writerow([
                result["text"],
                translate_label(result["label"]),
                f"{result['confidence']:.4f}"
            ])
    print("\nРезультати збережено у файл results.csv")

def show_info():
    print("\n===== MODEL INFO =====")
    print("Назва моделі:", MODEL_NAME)
    print("Підтримувані класи: ПОЗИТИВНИЙ / НЕГАТИВНИЙ")
    print("Нейтральна тональність: модель не підтримує цей клас")

while True:
    print("""
================================
       REVIEW ANALYZER
================================

1. Аналіз одного відгуку
2. Аналіз усіх відгуків
3. Інформація про модель
0. Вихід
""")
    choice = input("Оберіть пункт: ")
    if choice == "1":
        analyze_one()
    elif choice == "2":
        analyze_all()
    elif choice == "3":
        show_info()
    elif choice == "0":
        print("Програму завершено.")
        break
    else:
        print("Невірний пункт меню.")
