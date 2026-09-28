from numpy import random  
from config import  MODEL_NAME , NUM_EPOCHS , BATCH_SIZE  , LEARNING_RATE , OUTPUT_DIR , SAMPLE_TEST 
from transformers import  TrainingArguments , AutoTokenizer , AutoModelForSequenceClassification  , DataCollatorWithPadding ,pipeline
import numpy as np 
from sklearn.metrics import accuracy_score , f1_score 
import re   



def split_train_val_test(raw_all,  split_values ):
    # splitting the raw into train , val and test 
    train_percentage = split_values["train"]
    val_percentage = split_values["val"]
    test_percentage = split_values["test"]
    
    # shuffling the indices list 
    # n of elements 
    n_elements  = len(raw_all) 
    shuffled  = random.permutation(n_elements) # this shuffles the list of indices 

    # c here stands for check point 
    c_train = int(n_elements*(train_percentage/100))
    c_val   = c_train + int(n_elements*(val_percentage/100))
  
    train_indices = shuffled[:c_train ]
    val_indices = shuffled[c_train :c_val ]
    test_indices = shuffled[c_val:]

    raw_train = raw_all.select(train_indices)
    raw_val =  raw_all.select(val_indices)  
    raw_test = raw_all.select(test_indices)  
    return raw_train , raw_val , raw_test 
  

def create_mapping(raw_train):
    intents_unique = sorted(set(raw_train["intent"])) # in asceing order 
    label_mapping = {}
    for i in range (0,len(intents_unique)):
        label_mapping[intents_unique[i]] = i 

    return label_mapping 




def extract_and_process_data(data ,label_mapping): # Takes this bittext customer support data set and extracts the require data set in adition to tokenizing the daat 
    
    list_of_intent_labels_for_extracted_set =[ k["intent"] for k in data ]
    intent_labels_numerical = []
    raw_instruction_list = [k["instruction"] for k in data]
    tokenized_data = BERT_TOKENIZER(raw_instruction_list , padding=False , truncation = True ) 
    intent_labels_numerical = [label_mapping[k] for k in list_of_intent_labels_for_extracted_set]

    return tokenized_data , intent_labels_numerical , raw_instruction_list 

TrainingArguments = TrainingArguments (
    output_dir = OUTPUT_DIR , 
    num_train_epochs = NUM_EPOCHS , 
    per_device_train_batch_size = BATCH_SIZE, 
    per_device_eval_batch_size = BATCH_SIZE , 
    learning_rate = LEARNING_RATE , 
    eval_strategy = "epoch" , 
    save_strategy = "epoch", 
    use_cpu = False 
 ) 

BERT_TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")
collator = DataCollatorWithPadding(tokenizer = BERT_TOKENIZER )  

def compute_metrics(evaluation_prediction) :
    logits ,labels = evaluation_prediction  
    preds = np.argmax(logits,axis=-1)
    return {
        "accuracy":accuracy_score(labels,preds) , # compares the predictions we get 
        "f1_macro" : f1_score(labels,preds,average="macro") 
    } 


def clf_accuracy_test() :
    test_inputs = SAMPLE_TEST
    correct = 0 
    wrong = 0 
    confidence = 0 
    for test in test_inputs  : 
        clf = pipeline("text-classification" , model = OUTPUT_DIR  , tokenizer = BERT_TOKENIZER , device = 0  ) 
        user_instruction=test[0]
        result = clf(user_instruction)
        print(result)
        valid   = test[1]==result[0]["label"]
        confidence += result[0]["score"]
        if valid  == 1 :
            correct +=1 
        else :
            wrong +=1 

    print(f"This is the model's Real life accuracy:{correct/(correct+wrong)}")
    print(f"The Average confidence of the model in each prediction is:{confidence/len(test_inputs)}")

def skeleton(s):
    s = re.sub(r"\{\{.*?\}\}", "", s.lower())
    return re.sub(r"[^a-z ]", "", s).strip()


