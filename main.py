import os, re, threading, language_tool_python
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load a pre-trained sentence transformer model for coherency checks
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

model_path = "chatbot_model_secure"
train_file = "chatbot_dataset_secure.txt"
batch_size = 500
retraining_in_progress = False
context_buffer = []  # Buffer to store conversation history

# Initialize the LanguageTool object
tool = language_tool_python.LanguageTool('en-GB')

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

    # Update context buffer with the latest exchange
    update_context_buffer(user_input, bot_response)

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
    gpt_model = GPT2LMHeadModel.from_pretrained('distilgpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')

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
        model=gpt_model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()
    gpt_model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    global retraining_in_progress
    retraining_in_progress = False
    print("Retraining completed.")
    chat_with_user()

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

def chat_with_user():
    if os.path.exists(model_path):
        gpt_model = GPT2LMHeadModel.from_pretrained(model_path)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    else:
        print("Initializing new model...")
        gpt_model = GPT2LMHeadModel.from_pretrained('distilgpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')

    print("Chatbot is ready to talk! Type 'quit' to stop.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        response = generate_response(gpt_model, tokenizer, user_input)  # Pass gpt_model here
        print("Bot:", response)
        append_to_dataset(user_input, response)

def main():
    if not get_user_consent():
        print("Consent not given. Exiting.")
        return

    chat_with_user()  # Start chatting with the user
    handle_user_requests()

def generate_response(model, tokenizer, user_input):
    global context_buffer
    # Update context buffer and include it in the input for generating responses
    context = " ".join(context_buffer[-5:])  # Use the last 5 exchanges as context
    input_with_context = context + " " + user_input  # Combine context with the latest user input

    input_ids = tokenizer.encode(input_with_context + tokenizer.eos_token, return_tensors='pt')
    tokenizer.padding_side = 'left'

    chat_history_ids = model.generate(
        input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        # Add beam_search or other advanced decoding techniques here if needed
    )

    response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    post_processed_response = post_process_response(response)  # Implement this function to refine the response
    return post_processed_response

def update_context_buffer(user_input, bot_response):
    # Update the global context_buffer with the latest exchange
    global context_buffer
    context_buffer.extend([f"User: {user_input}", f"Bot: {bot_response}"])

def check_coherency(response, context):
    # Combine the context and response to check coherency
    combined_text = context + " " + response

    # Get embeddings for the combined text and the original context
    combined_embedding = sentence_model.encode([combined_text])  # Use sentence_model here
    context_embedding = sentence_model.encode([context])  # Use sentence_model here

    # Calculate cosine similarity between the combined text and original context embeddings
    similarity = cosine_similarity([combined_embedding], [context_embedding])[0][0]

    # If the similarity is below a certain threshold, the response might be considered incoherent
    threshold = 0.7  # This is an arbitrary threshold and might need adjustment
    if similarity < threshold:
        print("The response might be incoherent with the context.")
        # Here you might handle incoherent responses, e.g., by generating a new response

    return response  # Return the original or a modified response based on your coherency check logic

def correct_grammar(text):
    # Check the text for grammar issues
    matches = tool.check(text)
    corrected_text = language_tool_python.utils.correct(text, matches)
    return corrected_text

def post_process_response(response):
    # Trim the response to a maximum length, if necessary
    max_length = 280
    if len(response) > max_length:
        response = response[:max_length].rsplit(' ', 1)[0] + "..."

    # Correct basic punctuation spacing issues
    response = re.sub(r'\s+([,.!?"])', r'\1', response)

    # Advanced grammar correction using LanguageTool
    response = correct_grammar(response)

    # Filter out inappropriate content
    blacklist = ["badword1", "badword2"]
    for word in blacklist:
        response = response.replace(word, "[REDACTED]")

    response = check_coherency(response)

    return response.strip()

if __name__ == "__main__":
    main()
