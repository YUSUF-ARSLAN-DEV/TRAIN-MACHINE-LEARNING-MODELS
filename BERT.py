from datasets import load_dataset 
from transformers import AutoTokenizer , AutoModelForSequenceClassification 
from BERT_SUPPORT import split_train_val ,  extract_and_process_data , create_mapping 


data_set = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

# the data set has 2 dictionary inside 
# the train data set and the test data set 

raw_test = data_set["test"]
raw_train, raw_val = split_train_val(data_set["train"], 20)
label_mapping = create_mapping(raw_train)

train_tokenized, train_labels,  train_raw = extract_and_process_data(raw_train, label_mapping)
val_tokenized,   val_labels,    val_raw   = extract_and_process_data(raw_val,   label_mapping)
test_tokenized,  test_labels,   test_raw  = extract_and_process_data(raw_test,  label_mapping)