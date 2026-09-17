from numpy import random 
from config import BERT_TOKENIZER



def split_train_val(raw_train,  split_values = { "train" : 70  , "val" : 15 , "test" : 15 } ):
    # splitting the raw into train , val and test 
    train_percentage = split_values["train"]
    val_percentage = split_values["val"]
    test_percentage = split_values["test"]
    
    # shuffling the indices list 
    # n of elements 
    n_elements  = len(raw_train) 
    shuffled  = random.permutation(n_elements) # this shuffles the list of indices 

    # c here stands for check point 
    c_train = int(n_elements*train_percentage)
    c_val   = c_train + int(n_elements*val_percentage)
    c_test = c_val + int(n_elements*test_percentage)
    train_indices = shuffled[:c_train ]
    val_indices = shuffled[c_train :c_val ]
    test_indices = shuffled[c_val :c_test -1]

    

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
