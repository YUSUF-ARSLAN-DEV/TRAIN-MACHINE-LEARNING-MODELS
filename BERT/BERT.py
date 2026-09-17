from datasets import load_dataset  , Dataset
from transformers import AutoTokenizer , AutoModelForSequenceClassification 
from BERT.TEXT_ANALYSIS.BERT_SUPPORT import split_train_val_test ,  extract_and_process_data , create_mapping 


data_set = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

# the data set has 2 dictionary inside 
# the train data set and the test data set 


raw_train , raw_val , raw_test = split_train_val_test(data_set["train"])
label_mapping = create_mapping(raw_train)

train_tokenized, train_labels,  train_raw_instructions = extract_and_process_data(raw_train, label_mapping)
val_tokenized,   val_labels,    val_raw_instructoins   = extract_and_process_data(raw_val,   label_mapping)
test_tokenized,  test_labels,   test_raw_instructions  = extract_and_process_data(raw_test,  label_mapping)

training_ds = Dataset.from_dict(
    {
        "input_ids": train_tokenized["input_id"],
        "attention_mask" : train_tokenized["attention_mask"], 
        "labels" : train_labels 
    }    
)