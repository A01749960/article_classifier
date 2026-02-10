from __future__ import annotations

import json
import random
import re
from typing import Dict, List
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

SEED = 7
random.seed(SEED)
rng = np.random.default_rng(SEED)

st.set_page_config(page_title="Topic Classifier", page_icon="🧠", layout="centered")

st.title("Topic Classifier")
st.write(
    "Classifier workflow using Wikipedia text. "
)


@st.cache_data(show_spinner=False)
def wikipedia_plaintext(title: str, *, user_agent: str = "mathematica-replica") -> str:
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "titles": title,
    }
    url = "https://en.wikipedia.org/w/api.php?" + urlencode(params)
    req = Request(url, headers={"User-Agent": user_agent})
    with urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    page = next(iter(data["query"]["pages"].values()))
    return page.get("extract", "")


try:
    import nltk

    from nltk.tokenize import sent_tokenize

    def _regex_split(text: str) -> List[str]:
        chunks = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in chunks if s.strip()]

    def split_sentences(text: str) -> List[str]:
        try:
            return [s.strip() for s in sent_tokenize(text) if s.strip()]
        except LookupError:
            # Some NLTK versions require 'punkt_tab' in addition to 'punkt'.
            try:
                nltk.download("punkt", quiet=True)
                nltk.download("punkt_tab", quiet=True)
            except Exception:
                return _regex_split(text)
            try:
                return [s.strip() for s in sent_tokenize(text) if s.strip()]
            except Exception:
                return _regex_split(text)

except Exception:

    def split_sentences(text: str) -> List[str]:
        chunks = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in chunks if s.strip()]


@st.cache_data(show_spinner=False)
def text_sentences(title: str) -> List[str]:
    return split_sentences(wikipedia_plaintext(title))


@st.cache_resource(show_spinner=False)
def build_model():
    physics = text_sentences("Physics")
    biology = text_sentences("Biology")
    math = text_sentences("Mathematics")

    topicdataset = (
        [(s, "Phyics") for s in physics]
        + [(s, "Biology") for s in biology]
        + [(s, "Mathematics") for s in math]
    )

    X_train, y_train = zip(*topicdataset)
    model = make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2),
        MultinomialNB(),
    )
    model.fit(X_train, y_train)
    return model


with st.spinner("Training classifier from Wikipedia articles..."):
    try:
        model = build_model()
    except Exception as exc:
        st.error("Failed to build the model. Check your internet connection.")
        st.exception(exc)
        st.stop()


st.subheader("Classify text")
text = st.text_area("Text", value="The world is made of atoms", height=100)

if st.button("Classify"):
    pred = model.predict([text])[0]
    probs = model.predict_proba([text])[0]
    prob_map = dict(zip(model.classes_, probs))

    st.write(f"**Class:** {pred}")
    st.dataframe(prob_map)
    st.bar_chart(prob_map)

