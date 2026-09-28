from transformers import AutoTokenizer , DataCollatorWithPadding 

# The collator being a dynamic padder so that each batch gets padded with the most minimal number of tokens possible 
MODEL_NAME = "bert-base-uncased"
NUM_EPOCHS = 3 # how many times the model will see the entire data set 
BATCH_SIZE = 50 # how many  n pieces are we going to split the train_data set 
LEARNING_RATE = 0.00002
split_values = { "train" : 70  , "val" : 15 , "test" : 15 } 
OUTPUT_DIR = r"C:\Users\aonli\Desktop\INTERNSHIP PROJECTS\Independent Learning\TRAIN MACHINE LEARNING MODELS\BERT\bert_intent_model" 


SAMPLE_TEST = [
    # --- delete_account (clear) ---
    ("I want to close my account for good", "delete_account"),
    ("delete all my data and profile", "delete_account"),
    ("remove my account permanently", "delete_account"),
    # --- create_account ---
    ("how do I open an account", "create_account"),
    ("I want to register for a new account", "create_account"),
    # --- edit_account ---
    ("I need to update my profile details", "edit_account"),
    ("change the name on my account", "edit_account"),
    # --- recover_password ---
    ("I can't get into my account, password is gone", "recover_password"),
    ("my login stopped working", "recover_password"),
    ("account got locked out", "recover_password"),
    # --- registration_problems ---
    ("signing up isn't working for me", "registration_problems"),
    ("I keep getting an error when I register", "registration_problems"),
    # --- switch_account ---
    ("I want to use a different account", "switch_account"),
    ("how do I log into another profile", "switch_account"),
    # --- contact_human_agent ---
    ("put me through to a real human", "contact_human_agent"),
    ("I need to speak with a manager", "contact_human_agent"),
    ("connect me to an agent right now", "contact_human_agent"),
    # --- contact_customer_service ---
    ("I need help from customer service", "contact_customer_service"),
    ("can someone from support assist me", "contact_customer_service"),
    # --- delivery_period ---
    ("how long does delivery take", "delivery_period"),
    ("when will my parcel arrive", "delivery_period"),
    ("my package is late", "delivery_period"),
    # --- delivery_options ---
    ("what shipping options do you have", "delivery_options"),
    ("do you offer express shipping", "delivery_options"),
    # --- review ---
    ("I'd like to share some feedback", "review"),
    ("let me leave a review", "review"),
    # --- complaint ---
    ("I want to file a formal complaint", "complaint"),
    ("the product arrived damaged, I'm furious", "complaint"),
    # --- check_invoice ---
    ("show me my bill from last month", "check_invoice"),
    ("how much was I charged", "check_invoice"),
    # --- get_invoice ---
    ("I need a copy of my invoice", "get_invoice"),
    ("send me my invoice by email", "get_invoice"),
    # --- place_order ---
    ("I want to buy something new", "place_order"),
    ("help me place an order", "place_order"),
    # --- change_order ---
    ("I need to change my order details", "change_order"),
    ("modify my purchase before it ships", "change_order"),
    # --- track_order ---
    ("where is my order right now", "track_order"),
    ("I never got my parcel", "track_order"),
    # --- cancel_order ---
    ("cancel my purchase immediately", "cancel_order"),
    ("I want to stop my order", "cancel_order"),
    # --- check_payment_methods ---
    ("what payment methods do you accept", "check_payment_methods"),
    ("can I pay with PayPal", "check_payment_methods"),
    # --- payment_issue ---
    ("my card was declined", "payment_issue"),
    ("I was charged twice for one item", "payment_issue"),
    # --- check_refund_policy ---
    ("what's your refund policy", "check_refund_policy"),
    ("how many days do I have to return this", "check_refund_policy"),
    # --- get_refund ---
    ("I want my money back", "get_refund"),
    # --- track_refund ---
    ("where is my refund", "track_refund"),
    # --- check_cancellation_fee ---
    ("how much is the cancellation fee", "check_cancellation_fee"),
    # --- change_shipping_address ---
    ("I moved, update my delivery address", "change_shipping_address"),
    # --- set_up_shipping_address ---
    ("set up a new shipping address for me", "set_up_shipping_address"),
    # --- newsletter_subscription ---
    ("stop emailing me your newsletter", "newsletter_subscription"),
]