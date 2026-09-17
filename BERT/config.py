from transformers import AutoTokenizer , DataCollatorWithPadding 
BERT_TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")
collator = DataCollatorWithPadding(tokenizer = BERT_TOKENIZER )  
# The collator being a dynamic padder so that each batch gets padded with the most minimal number of tokens possible 
MODEL_NAME = "bert-base-uncased"
NUM_EPOCHS = 3 # how many times the model will see the entire data set 
BATCH_SIZE = 16 # how many  n pieces are we going to split the train_data set 
LEARNING_RATE = 0.00002
split_values = { "train" : 70  , "val" : 15 , "test" : 15 } 