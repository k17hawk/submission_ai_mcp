# core/clause_ratings.py
"""
Load the ACORD 2-5 Star Clause Pairs Excel and predict rating for a new clause.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List
import re

# Optional: use a simple similarity or ML model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsRegressor


class ClauseRatings:
    def __init__(self, excel_path: Path):
        self.excel_path = excel_path
        self.df: Optional[pd.DataFrame] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.model: Optional[KNeighborsRegressor] = None
        self._load_and_prepare()

    def _load_and_prepare(self):
        """Load and flatten the Excel into a DataFrame with columns: clause_text, category, rating."""
        # The Excel has a complex multi-header format. We'll flatten it:
        # First, read without headers to capture the structure.
        df_raw = pd.read_excel(self.excel_path, header=None)

        # Row 0 contains category names (e.g., "cap on liability") and "Rating" strings.
        # Row 1 contains actual clause texts (starting at column 0) and ratings in subsequent columns.
        # We'll iterate over columns to extract pairs.
        records = []
        num_rows = len(df_raw)
        # Identify category columns: row 0 has a string not equal to "Rating"
        categories = []
        rating_cols = []
        for col in range(df_raw.shape[1]):
            val = df_raw.iloc[0, col]
            if pd.isna(val):
                continue
            if str(val).strip() == "Rating":
                rating_cols.append(col)
            else:
                categories.append((str(val).strip(), col))

        # Now for each row from row 2 onward (since row 0=headers, row 1=first clause? Actually row 1 contains first clause text and maybe ratings)
        # Based on your snippet, row index 1 seems to be clause texts mixed with ratings.
        # Safest: start from row 1, and for each row, clause is in column 0, then each category's rating is at category_col+1.
        for row_idx in range(1, num_rows):
            clause = df_raw.iloc[row_idx, 0]
            if pd.isna(clause):
                continue
            clause = str(clause).strip()
            for cat_name, cat_col in categories:
                rating_col = cat_col + 1
                if rating_col < df_raw.shape[1]:
                    rating_val = df_raw.iloc[row_idx, rating_col]
                    if pd.notna(rating_val):
                        try:
                            rating = float(rating_val)
                        except:
                            rating = None
                        if rating is not None:
                            records.append({
                                'clause_text': clause,
                                'category': cat_name,
                                'rating': rating
                            })
        self.df = pd.DataFrame(records)

    def build_prediction_model(self):
        """Train a simple k-NN regressor on TF-IDF features of clause texts (per category? or global)."""
        if self.df is None or self.df.empty:
            raise ValueError("No data loaded.")
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
        X = self.vectorizer.fit_transform(self.df['clause_text'])
        y = self.df['rating']
        self.model = KNeighborsRegressor(n_neighbors=3, weights='distance')
        self.model.fit(X, y)

    def predict_rating(self, clause_text: str, category: Optional[str] = None) -> float:
        """
        Predict rating (1-5) for a given clause text.
        If category is provided, we might filter training data to that category.
        For simplicity, we use all training clauses.
        """
        if self.model is None:
            self.build_prediction_model()
        # Optionally filter by category (if you want per-category models)
        # For now, use the global model.
        X_query = self.vectorizer.transform([clause_text])
        pred = self.model.predict(X_query)[0]
        # Clip to 1-5 range and round to half-star? Return as float.
        pred = max(1.0, min(5.0, pred))
        return round(pred, 1)

    def get_rating_distribution(self) -> pd.Series:
        return self.df['rating'].value_counts().sort_index()

    def get_clauses_by_category(self, category: str) -> pd.DataFrame:
        return self.df[self.df['category'] == category]