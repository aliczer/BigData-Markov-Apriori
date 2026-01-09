from pyspark.sql import functions as F
import re

df_raw = (
    spark.read
    .option("header", False)
    .option("inferSchema", True)
    .option("sep", "\t")
    .csv("dbfs:/databricks-datasets/sms_spam_collection/data-001/")
)

df_raw = df_raw.select(
    F.col("_c0").alias("label"),
    F.col("_c1").alias("message")
)

df_spam = df_raw.filter(F.col("label") == "spam")

def clean_text_apriori(text):
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

clean_apriori_udf = F.udf(clean_text_apriori)

df_tx = (
    df_spam
    .withColumn("clean", clean_apriori_udf("message"))
    .withColumn(
        "items",
        F.sort_array(
            F.array_distinct(
                F.split("clean", " ")
            )
        )
    )
)

df_tx = (
    df_tx
    .withColumn("HAS_NUMBER",
        F.when(F.col("message").rlike(r"\d"), F.lit("HAS_NUMBER")).otherwise(None)
    )
    .withColumn("HAS_URL",
        F.when(F.col("message").rlike(r"http"), F.lit("HAS_URL")).otherwise(None)
    )
    .withColumn("HAS_CURRENCY",
        F.when(F.col("message").rlike(r"£|\$|€"), F.lit("HAS_CURRENCY")).otherwise(None)
    )
)

df_tx = (
    df_tx
    .withColumn(
        "items",
        F.sort_array(
            F.array_distinct(
                F.concat("items", F.array("HAS_NUMBER", "HAS_URL", "HAS_CURRENCY"))
            )
        )
    )
    .withColumn("items", F.expr("filter(items, x -> x is not null)"))
    .withColumn("tid", F.monotonically_increasing_id())
    .select("tid", "items")
)

stopwords = [
    "a","the","to","is","you","your","of","for","on","in","at",
    "or","and","be","we","u","i","me","my","it","this","that"
]

df_tx = df_tx.withColumn(
    "items",
    F.expr("filter(items, x -> length(x) > 1)")
)

df_tx = df_tx.withColumn(
    "items",
    F.expr(
        f"filter(items, x -> NOT array_contains(array({','.join([f'\"{w}\"' for w in stopwords])}), x))"
    )
)

n_tx = df_tx.count()

freq_1 = (
    df_tx
    .select(F.explode("items").alias("item"))
    .groupBy("item")
    .count()
    .withColumn("support", F.col("count") / F.lit(n_tx))
    .filter(F.col("support") >= 0.005)
    .orderBy(F.desc("support"))
)

freq_1.show(20, truncate=False)

pairs = (
    df_tx
    .select("tid", F.explode("items").alias("item1"))
    .join(
        df_tx.select("tid", F.explode("items").alias("item2")),
        on="tid"
    )
    .filter(F.col("item1") < F.col("item2"))
)


freq_1_01 = (
    df_tx
    .select(F.explode("items").alias("item"))
    .groupBy("item")
    .count()
    .withColumn("support", F.col("count") / F.lit(n_tx))
    .filter(F.col("support") >= 0.01)
    .orderBy(F.desc("support"))
)

freq_1_01.show(20, truncate=False)

freq_2 = (
    pairs
    .groupBy("item1", "item2")
    .count()
    .withColumn("support", F.col("count") / F.lit(n_tx))
    .filter(F.col("support") >= 0.005)
    .orderBy(F.desc("support"))
)

freq_2.show(20, truncate=False)

freq_2_01 = (
    pairs
    .groupBy("item1", "item2")
    .count()
    .withColumn("support", F.col("count") / F.lit(n_tx))
    .filter(F.col("support") >= 0.01)
    .orderBy(F.desc("support"))
)

freq_2_01.show(20, truncate=False)

rules = (
    freq_2
    .join(
        freq_1.select(
            F.col("item").alias("A"),
            F.col("support").alias("support_A")
        ),
        freq_2.item1 == F.col("A")
    )
    .join(
        freq_1.select(
            F.col("item").alias("B"),
            F.col("support").alias("support_B")
        ),
        freq_2.item2 == F.col("B")
    )
    .withColumn("confidence", F.col("support") / F.col("support_A"))
    .withColumn("lift", F.col("confidence") / F.col("support_B"))
    .select(
        F.col("item1").alias("antecedent"),
        F.col("item2").alias("consequent"),
        "support",
        "confidence",
        "lift"
    )
    .orderBy(F.desc("lift"))
)

rules.show(20, truncate=False)

rules_final = (
    rules
    .filter(F.col("support") >= 0.02)   
    .filter(F.col("confidence") >= 0.6)
    .orderBy(F.desc("lift"))
)

rules_final.show(20, truncate=False)


df_tx.select(F.explode("items").alias("item")) \
    .groupBy("item") \
    .count() \
    .orderBy(F.desc("count")) \
    .show(20, truncate=False)
