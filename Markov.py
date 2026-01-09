from pyspark.sql import functions as F
import re
import random
from collections import defaultdict, Counter
import math

df_raw = (
    spark.read
    .option("header", False)
    .option("inferSchema", True)
    .option("sep", "\t")
    .csv("dbfs:/databricks-datasets/sms_spam_collection/data-001/")
)

df_raw.printSchema()
df_raw.show(5, truncate=False)

df_raw = df_raw.select(
    F.col("_c0").alias("label"),
    F.col("_c1").alias("message")
)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", " URL ", text)
    text = re.sub(r"\d+", " NUMBER ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

clean_udf = F.udf(clean_text)

df = (
    df_raw
    .withColumn("clean", clean_udf("message"))
    .withColumn("tokens", F.split("clean", " "))
)

df.select("label", "tokens").show(5, truncate=False)

tokens = (
    df
    .filter(F.col("label") == "spam")
    .select(F.explode("tokens").alias("word"))
    .select("word")
    .collect()
)

tokens = [row.word for row in tokens]

tokens.append(tokens[0])

len(tokens), tokens[:20]

def build_markov_model(tokens, k):
    model = defaultdict(Counter)
    
    for i in range(len(tokens) - k):
        state = tuple(tokens[i:i+k])
        next_word = tokens[i+k]
        model[state][next_word] += 1
    
    prob_model = {}
    for state, counter in model.items():
        total = sum(counter.values())
        prob_model[state] = {
            word: count / total for word, count in counter.items()
        }
    return prob_model

model_k1 = build_markov_model(tokens, 1)
model_k2 = build_markov_model(tokens, 2)
model_k3 = build_markov_model(tokens, 3)

len(model_k1), len(model_k2), len(model_k3)

def generate_text(model, k, length=500):
    state = random.choice(list(model.keys()))
    output = list(state)
    
    for _ in range(length):
        next_words = model.get(state)
        if not next_words:
            state = random.choice(list(model.keys()))
            continue
        
        words, probs = zip(*next_words.items())
        next_word = random.choices(words, probs)[0]
        output.append(next_word)
        state = tuple(output[-k:])
    
    return " ".join(output)

text_k1 = generate_text(model_k1, 1)
text_k2 = generate_text(model_k2, 2)
text_k3 = generate_text(model_k3, 3)

print("===== k = 1 =====\n", text_k1[:700])
print("\n===== k = 2 =====\n", text_k2[:700])
print("\n===== k = 3 =====\n", text_k3[:700])

def hapax_ratio(text):
    words = text.split()
    freq = Counter(words)
    hapax = sum(1 for w in freq if freq[w] == 1)
    return hapax / len(freq)

hapax_k1 = hapax_ratio(text_k1)
hapax_k2 = hapax_ratio(text_k2)
hapax_k3 = hapax_ratio(text_k3)

hapax_k1, hapax_k2, hapax_k3

def bigram_distribution(tokens):
    bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
    freq = Counter(bigrams)
    total = sum(freq.values())
    return {k: v/total for k, v in freq.items()}

def total_variation(p, q):
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in keys)

# oryginalny rozkład
orig_bigrams = bigram_distribution(tokens)

tv_k1 = total_variation(orig_bigrams, bigram_distribution(text_k1.split()))
tv_k2 = total_variation(orig_bigrams, bigram_distribution(text_k2.split()))
tv_k3 = total_variation(orig_bigrams, bigram_distribution(text_k3.split()))

tv_k1, tv_k2, tv_k3

def special_token_ratio(text, token):
    words = text.split()
    return words.count(token) / len(words)

stats = {
    "k": [1, 2, 3],
    "hapax": [hapax_k1, hapax_k2, hapax_k3],
    "TV_bigram": [tv_k1, tv_k2, tv_k3],
    "URL_ratio": [
        special_token_ratio(text_k1, "URL"),
        special_token_ratio(text_k2, "URL"),
        special_token_ratio(text_k3, "URL")
    ],
    "NUMBER_ratio": [
        special_token_ratio(text_k1, "NUMBER"),
        special_token_ratio(text_k2, "NUMBER"),
        special_token_ratio(text_k3, "NUMBER")
    ]
}

stats

from pyspark.ml.feature import Tokenizer, CountVectorizer, StringIndexer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline

df_cls = df_raw.select("label", "message")

label_indexer = StringIndexer(
    inputCol="label",
    outputCol="label_idx"
)

tokenizer = Tokenizer(
    inputCol="message",
    outputCol="words"
)

vectorizer = CountVectorizer(
    inputCol="words",
    outputCol="features",
    minDF=5
)

nb = NaiveBayes(
    featuresCol="features",
    labelCol="label_idx",
    modelType="multinomial"
)

pipeline = Pipeline(stages=[
    label_indexer,
    tokenizer,
    vectorizer,
    nb
])

train_df, test_df = df_cls.randomSplit([0.8, 0.2], seed=42)

nb_model = pipeline.fit(train_df)

pred_test = nb_model.transform(test_df)
pred_test.select("label", "prediction").show(10)

generated_texts = [
    ("markov_k1", text_k1),
    ("markov_k2", text_k2),
    ("markov_k3", text_k3)
]

df_generated = spark.createDataFrame(
    generated_texts,
    ["source", "message"]
)

pred_generated = nb_model.transform(df_generated)

pred_generated.select(
    "source",
    "prediction",
    "probability"
).show(truncate=False)
