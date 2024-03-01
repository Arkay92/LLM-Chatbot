import os
import re
import threading
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Define global paths and variables
model_path = "chatbot_model_secure"
train_file = "chatbot_dataset_secure.txt"
batch_size = 500  # Adjust based on your requirements
retraining_in_progress = False

def get_user_consent():
    consent_text = """
    Welcome to the chatbot. We value your privacy and comply with data protection regulations. 
    We collect conversation data to improve our services. Your data will be anonymized and securely stored. 
    You have the right to access or delete your data at any time. Do you consent to these terms? (yes/no): """
    consent = input(consent_text).lower()
    return consent == 'yes'

def anonymize_data(text):
    text = re.sub(r'\b[A-Z][a-z]*\b', '[NAME]', text)  # Attempt to anonymize names
    text = re.sub(r'[\w\.-]+@[\w\.-]+', '[EMAIL]', text)  # Anonymize email addresses
    return text

def append_to_dataset(user_input, bot_response):
    global retraining_in_progress
    user_input_anonymized = anonymize_data(user_input)
    bot_response_anonymized = anonymize_data(bot_response)

    with open(train_file, "a", encoding="utf-8") as f:
        f.write(f"User: {user_input_anonymized}\nBot: {bot_response_anonymized}\n")

    # Trigger retraining based on the size of the dataset
    if os.path.getsize(train_file) // batch_size > 0 and not retraining_in_progress:
        trigger_retraining()

def trigger_retraining():
    global retraining_in_progress
    retraining_in_progress = True
    print("Initiating retraining process in the background...")
    retraining_thread = threading.Thread(target=secure_train_model, args=())
    retraining_thread.start()

def secure_train_model():
    print("Training with new data...")
    model_name = 'distilgpt2'
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    train_dataset = TextDataset(tokenizer=tokenizer, file_path=train_file, block_size=128)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=model_path,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        warmup_steps=500,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    global retraining_in_progress
    retraining_in_progress = False
    print("Retraining completed.")

def handle_user_requests():
    request = input("Do you want to access or delete your data? (access/delete/no): ").lower()
    if request == 'access':
        if os.path.exists(train_file):
            with open(train_file, "r", encoding="utf-8") as f:
                print(f.read())
        else:
            print("No data available.")
    elif request == 'delete':
        if os.path.exists(train_file):
            os.remove(train_file)
            print("Your data has been deleted.")
        else:
            print("No data to delete.")

def main():
    if not get_user_consent():
        print("Consent not given. Exiting.")
        return

    if os.path.exists(model_path):
        model = GPT2LMHeadModel.from_pretrained(model_path)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    else:
        print("Initializing new model...")
        model = GPT2LMHeadModel.from_pretrained('distilgpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')

    print("Chatbot is ready to talk! Type 'quit' to stop.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        response = generate_response(model, tokenizer, user_input)

        print("Bot:", response)
        append_to_dataset(user_input, response)

    handle_user_requests()

def generate_response(model, tokenizer, user_input):
    # Encode the user input and add end of string token
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Generate a sequence of tokens from the model based on the input
    chat_history_ids = model.generate(
        input_ids, 
        max_length=1000, 
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,  # Prevents the model from repeating the same n-grams
        do_sample=True,  # Enable sampling to generate more diverse responses
        top_k=50,  # The number of highest probability vocabulary tokens to keep for top-k-filtering
        top_p=0.95,  # If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation
        temperature=0.7  # Controls the randomness of predictions by scaling the logits before applying softmax
    )

    # Decode the generated tokens to a string and skip the special tokens
    response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    main()
