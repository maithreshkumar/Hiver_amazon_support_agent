from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class SimpleBaseline:
    def fit(self, texts: list[str], labels: list[str]) -> "SimpleBaseline":
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True
        )
        matrix = self.vectorizer.fit_transform(texts)
        self.classifier = LogisticRegression(max_iter=1000, class_weight="balanced").fit(matrix, labels)
        return self

    def predict_intent(self, text: str) -> dict[str, object]:
        probabilities = self.classifier.predict_proba(self.vectorizer.transform([text]))[0]
        index = int(np.argmax(probabilities))
        return {
            "intent": str(self.classifier.classes_[index]),
            "confidence": float(probabilities[index]),
        }

