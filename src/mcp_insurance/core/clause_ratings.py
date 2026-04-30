# core/clause_ratings.py
"""
Load the ACORD 2-5 Star Clause Pairs Excel and predict rating for a new clause.

Excel structure:
  Row 0: category name (e.g. "cap on liability"), "Rating", next category, "Rating", ...
  Row 1: sub-headers (e.g. "«worse»", "«better»", "Clauses") — SKIP THIS ROW
  Row 2+: clause text in category column, numeric rating in the adjacent Rating column
"""
import logging
import pandas as pd
from pathlib import Path
from typing import Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsRegressor

logger = logging.getLogger(__name__)


class ClauseRatings:
    def __init__(self, excel_path: Path):
        self.excel_path = excel_path
        self.df: Optional[pd.DataFrame] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.model: Optional[KNeighborsRegressor] = None
        self._load_and_prepare()

    def _load_and_prepare(self):
        """
        Load and flatten the Excel into a DataFrame with columns:
            clause_text, category, rating
        
        Excel layout (0-indexed):
          Row 0  → category name | "Rating" | category name | "Rating" | ...
          Row 1  → sub-headers like «worse», «better», Clauses  ← SKIP
          Row 2+ → clause text | numeric rating | clause text | numeric rating | ...
        """
        df_raw = pd.read_excel(self.excel_path, header=None)
        logger.info(f"Excel shape: {df_raw.shape}")
        logger.info(f"Row 0 (headers): {df_raw.iloc[0].tolist()}")
        logger.info(f"Row 1 (sub-headers): {df_raw.iloc[1].tolist()}")

        # --- Step 1: Find (category_name, clause_col, rating_col) triples from row 0 ---
        column_groups = []  # list of (category_name, clause_col, rating_col)
        col = 0
        while col < df_raw.shape[1]:
            val = df_raw.iloc[0, col]
            if pd.isna(val):
                col += 1
                continue
            val_str = str(val).strip()
            if val_str == "Rating":
                # Orphan Rating col — skip
                col += 1
                continue
            # This is a category name; next non-empty col should be its Rating col
            rating_col = col + 1
            while rating_col < df_raw.shape[1]:
                next_val = df_raw.iloc[0, rating_col]
                if pd.notna(next_val) and str(next_val).strip() == "Rating":
                    break
                rating_col += 1
            if rating_col < df_raw.shape[1]:
                column_groups.append((val_str, col, rating_col))
                logger.info(f"Category '{val_str}': clause_col={col}, rating_col={rating_col}")
            col += 1

        # --- Step 2: Extract records starting from row 2 (skip row 0=headers, row 1=sub-headers) ---
        records = []
        for row_idx in range(2, len(df_raw)):
            for cat_name, clause_col, rating_col in column_groups:
                clause = df_raw.iloc[row_idx, clause_col]
                rating_val = df_raw.iloc[row_idx, rating_col]

                if pd.isna(clause) or pd.isna(rating_val):
                    continue

                clause_str = str(clause).strip()
                if not clause_str:
                    continue

                try:
                    rating = float(rating_val)
                except (ValueError, TypeError):
                    continue

                if 1.0 <= rating <= 5.0:
                    records.append({
                        'clause_text': clause_str,
                        'category': cat_name,
                        'rating': rating
                    })

        self.df = pd.DataFrame(records)
        logger.info(f"Loaded {len(self.df)} clause-rating pairs across {self.df['category'].nunique() if not self.df.empty else 0} categories")
        if not self.df.empty:
            logger.info(f"Categories found: {self.df['category'].unique().tolist()}")

    def build_prediction_model(self):
        """Train a k-NN regressor on TF-IDF features of all clause texts."""
        if self.df is None or self.df.empty:
            raise ValueError("No data loaded. Check Excel path and structure.")
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        X = self.vectorizer.fit_transform(self.df['clause_text'])
        y = self.df['rating']
        self.model = KNeighborsRegressor(n_neighbors=min(5, len(self.df)), weights='distance')
        self.model.fit(X, y)
        logger.info(f"Prediction model built on {len(self.df)} samples")

    def predict_rating(self, clause_text: str, category: Optional[str] = None) -> float:
        """
        Predict rating (1.0–5.0) for a given clause text.
        If category is provided, trains a local model on just that category's data.
        """
        if category:
            subset = self.df[self.df['category'] == category]
            if len(subset) >= 3:
                vec = TfidfVectorizer(stop_words='english', max_features=500)
                X = vec.fit_transform(subset['clause_text'])
                y = subset['rating']
                mdl = KNeighborsRegressor(n_neighbors=min(3, len(subset)), weights='distance')
                mdl.fit(X, y)
                pred = mdl.predict(vec.transform([clause_text]))[0]
                return round(max(1.0, min(5.0, float(pred))), 1)

        # Fall back to global model
        if self.model is None:
            self.build_prediction_model()
        X_query = self.vectorizer.transform([clause_text])
        pred = self.model.predict(X_query)[0]
        return round(max(1.0, min(5.0, float(pred))), 1)

    def get_rating_distribution(self) -> pd.Series:
        return self.df['rating'].value_counts().sort_index()

    def get_clauses_by_category(self, category: str) -> pd.DataFrame:
        return self.df[self.df['category'] == category]

    def list_categories(self):
        """Return all available category names."""
        if self.df is None or self.df.empty:
            return []
        return sorted(self.df['category'].unique().tolist())