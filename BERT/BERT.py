from datasets import load_dataset  , Dataset  
from BERT_SUPPORT import split_train_val_test ,  extract_and_process_data , create_mapping , TrainingArguments , BERT_TOKENIZER ,collator  , compute_metrics
from transformers import  AutoModelForSequenceClassification   , Trainer , pipeline 
from config import MODEL_NAME ,split_values , OUTPUT_DIR # where the weights are saved 
data_set = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

# the data set has 2 dictionary inside 
# the train data set and the test data set 


raw_train , raw_val , raw_test = split_train_val_test(data_set["train"],split_values)
label_mapping = create_mapping(raw_train)

train_tokenized, train_labels,  train_raw_instructions = extract_and_process_data(raw_train, label_mapping)
val_tokenized,   val_labels,    val_raw_instructoins   = extract_and_process_data(raw_val,   label_mapping)
test_tokenized,  test_labels,   test_raw_instructions  = extract_and_process_data(raw_test,  label_mapping)

training_ds = Dataset.from_dict(
    {
        "input_ids": train_tokenized["input_ids"],
        "attention_mask" : train_tokenized["attention_mask"], 
        "labels" : train_labels 
    }    
)

val_ds = Dataset.from_dict( 
    {
    "input_ids" : val_tokenized["input_ids"] , 
    "attention_mask" : val_tokenized["attention_mask"] , 
    "labels" : val_labels 
    }
)


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME , 
    num_labels = len(label_mapping) 
) 

trainer = Trainer (
    model = model , 
    args = TrainingArguments , 
    train_dataset = training_ds , 
    eval_dataset = val_ds , 
    data_collator = collator , 
    compute_metrics = compute_metrics
) 
#trainer.train() 

# manually testing the pipeline 


clf = pipeline("text-classification" , model = OUTPUT_DIR + "/checkpoint-3528" , tokenizer = BERT_TOKENIZER ) 
result = clf("I want to cancel my super very super long order that was not cancelled last week even though I bought it ")
print(result)

