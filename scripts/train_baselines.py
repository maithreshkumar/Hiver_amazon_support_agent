from pathlib import Path

import joblib
import pandas as pd
import yaml

from hiver_support.baselines.simple import SimpleBaseline
from hiver_support.baselines.trivial import TrivialBaseline
from hiver_support.intents.taxonomy import load_approved_taxonomy

config = yaml.safe_load(Path("configs/training.yaml").read_text(encoding="utf-8"))
load_approved_taxonomy(config["taxonomy_path"])
data = pd.read_parquet(config["weak_labels_path"])
output = Path(config["baseline_dir"])
output.mkdir(parents=True, exist_ok=True)
labels = data["predicted_intent"].astype(str).tolist()
texts = data["customer_text"].astype(str).tolist()
joblib.dump(TrivialBaseline().fit(labels), output / "trivial.joblib")
joblib.dump(SimpleBaseline().fit(texts, labels), output / "simple.joblib")
print(f"Saved baselines to {output}")
