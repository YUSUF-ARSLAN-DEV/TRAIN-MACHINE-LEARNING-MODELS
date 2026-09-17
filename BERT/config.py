from transformers import AutoTokenizer , DataCollatorWithPadding 

# The collator being a dynamic padder so that each batch gets padded with the most minimal number of tokens possible 
MODEL_NAME = "bert-base-uncased"
NUM_EPOCHS = 3 # how many times the model will see the entire data set 
BATCH_SIZE = 16 # how many  n pieces are we going to split the train_data set 
LEARNING_RATE = 0.00002
split_values = { "train" : 70  , "val" : 15 , "test" : 15 } 
OUTPUT_DIR = r"C:\Users\aonli\Desktop\INTERNSHIP PROJECTS\Independent Learning\TRAIN MACHINE LEARNING MODELS\BERT\bert_intent_model" 
