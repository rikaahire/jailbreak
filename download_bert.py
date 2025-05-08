from transformers import AutoTokenizer, AutoModelForSequenceClassification

save_path = "./bert-base-cased"

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased")

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)