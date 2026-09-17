from datasets import load_dataset 
data_set = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

print(data_set["train"].column_names)
print(data_set["train"][0])