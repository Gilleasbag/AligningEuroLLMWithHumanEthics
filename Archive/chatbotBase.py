from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Load the model and tokenizer
model_id = "utter-project/EuroLLM-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Use GPU if available
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model.to(device)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def chat_with_bot():
    print("Chatbot is ready! Type 'exit' to end the chat.\n")

    while True:
        # Get user input
        user_input = input("You (English): ")

        # Exit condition
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Format the input as per the model's expected format
        text = f"{user_input}"

        # Tokenize the input
        inputs = tokenizer(text, return_tensors="pt").to(device)

        # Generate a response
        outputs = model.generate(
            **inputs,
            max_new_tokens=25,
        )

        # Decode and print the chatbot's response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    chat_with_bot()
