# 📚 NLP (Doğal Dil İşleme) Uygulamalı Notlarım

Bu notlar, Miuul Veri Bilimi Eğitim Programı kapsamında aldığım eğitim sürecinde hazırladığım uygulamalı çalışmaları ve kendi ek açıklamalarımı içermektedir. Metin verileri üzerinde temel NLP işlemlerinden başlayarak görselleştirme, duygu analizi ve makine öğrenmesi modelleme süreçlerine kadar geniş bir kapsam sunmaktadır

---

## 1. Veri Yükleme ve Ayarlar

```python
from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Veri setini oku
df = pd.read_csv("C:/Users/sudea/PycharmProjects/NLP/amazon_reviews.csv", sep=",")
df.head()
```

---

## 2. Veri Ön İşleme (Text Preprocessing)

### 2.1 Case Folding (Büyük/Küçük Harf Dönüşümü)

```python
df['reviewText'] = df['reviewText'].str.lower()
```

### 2.2 Noktalama İşaretlerinin Temizlenmesi

```python
df['reviewText'] = df['reviewText'].str.replace('[^\w\s]', '')
```

### 2.3 Sayısal İfadelerin Silinmesi

```python
df['reviewText'] = df['reviewText'].str.replace('\d', '')
```

### 2.4 Stop Words Temizliği (Gereksiz kelimelerin çıkarılması)

```python
nltk.download('stopwords')
sw = stopwords.words('english')
df['reviewText'] = df['reviewText'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
```

### 2.5 Rare Words (Nadir Kelimeler) Temizliği

```python
temp_df = pd.Series(' '.join(df['reviewText']).split()).value_counts()
drops = temp_df[temp_df <= 1]
df['reviewText'] = df['reviewText'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))
```

### 2.6 Lemmatization (Kelime Köklerine Ayırma)

```python
nltk.download('wordnet')
df['reviewText'] = df['reviewText'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
```

---

## 3. Metin Görselleştirme

### 3.1 Terim Frekansı (Term Frequency) Hesaplama

```python
tf = df['reviewText'].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns = ["words", "tf"]
```

### 3.2 WordCloud Görselleştirme

```python
text = " ".join(i for i in df.reviewText)

wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
```

---

## 4. Duygu Analizi (Sentiment Analysis)

### 4.1 Sentiment Skoru Hesaplama

```python
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

df['polarity_score'] = df['reviewText'].apply(lambda x: sia.polarity_scores(x)['compound'])
```

### 4.2 Sentiment Etiketleme (Pozitif/Negatif)

```python
df['sentiment_label'] = df['reviewText'].apply(lambda x: "pos" if sia.polarity_scores(x)['compound'] > 0 else "neg")
df['sentiment_label'] = LabelEncoder().fit_transform(df['sentiment_label'])
```

---

## 5. Özellik Mühendisliği (Feature Engineering)

Özellik mühendisliği aşamasında metinleri makine öğrenmesi algoritmalarının anlayabileceği sayısal formatlara çeviriyoruz.

### 5.1 Count Vectors

Kelimelerin dokümanlardaki ham frekanslarına göre bir temsil oluşturur.

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X_count = vectorizer.fit_transform(df['reviewText'])
```

### 5.2 N-gram Tabanlı Count Vectors

İkili veya üçlü kelime öbekleri (n-gramlar) oluşturarak daha anlamlı özellikler çıkarır.

```python
vectorizer_ngram = CountVectorizer(ngram_range=(2, 3))
X_ngram = vectorizer_ngram.fit_transform(df['reviewText'])
```

### 5.3 TF-IDF (Term Frequency - Inverse Document Frequency)

TF-IDF yöntemi, hem bir kelimenin dokümanda ne kadar sık geçtiğini hem de o kelimenin koleksiyondaki tüm dokümanlar arasında ne kadar ayırt edici olduğunu dikkate alır.

Formüller:

\[ TF(t) = \frac{\text{Belirli dokümanda t kelimesinin frekansı}}{\text{Dokümandaki toplam kelime sayısı}} \]

\[ IDF(t) = \log\left(\frac{1 + \text{Toplam doküman sayısı}}{1 + \text{t kelimesini içeren doküman sayısı}}\right) \]

TF-IDF skoru, TF ve IDF değerlerinin çarpımıdır.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(df['reviewText'])
```

N-gram tabanlı TF-IDF vektörizasyonu:

```python
tfidf_ngram_vectorizer = TfidfVectorizer(ngram_range=(2, 3))
X_tfidf_ngram = tfidf_ngram_vectorizer.fit_transform(df['reviewText'])
```

---

## 6. Modelleme

### 6.1 Lojistik Regresyon ile Duygu Sınıflandırması

```python
log_model = LogisticRegression().fit(X_tfidf, df['sentiment_label'])
cv_score = cross_val_score(log_model, X_tfidf, df['sentiment_label'], scoring='accuracy', cv=5).mean()
print(f"Logistic Regression CV Accuracy: {cv_score:.4f}")
```

Yeni yorum tahmini:

```python
new_review = pd.Series("this product is great")
new_review_tfidf = tfidf_vectorizer.transform(new_review)
print(log_model.predict(new_review_tfidf))
```

### 6.2 Random Forest ile Sınıflandırma ve Hiperparametre Optimizasyonu

```python
rf_model = RandomForestClassifier(random_state=17)

rf_params = {
    "max_depth": [8, None],
    "max_features": [7, "auto"],
    "min_samples_split": [2, 5, 8],
    "n_estimators": [100, 200]
}

rf_cv_model = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X_tfidf, df['sentiment_label'])

print(f"Best RF Parameters: {rf_cv_model.best_params_}")
```

---

# 🎯 Notlar:
- NLP projelerinde veri ön işleme adımları model başarımı için kritik öneme sahiptir.
- Özellik mühendisliğinde hem kelime bazlı hem de karakter bazlı temsiller kullanılmalıdır.
- TF-IDF gibi ağırlıklı yöntemler, nadir ama önemli kelimeleri öne çıkarmada etkilidir.
- Modelleme aşamasında öncelikle basit modeller (Lojistik Regresyon gibi) denenmeli, ardından daha karmaşık modellere (Random Forest vb.) geçilmelidir.

---

İlerleyen süreçte bu yapıyı daha da geliştirerek LSTM, BERT gibi derin öğrenme tabanlı modellerle desteklemeyi planlıyorum. 🚀

---