from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model and tokenizer
model_id = "/fs/nas/eikthyrnir0/gpeterson/Fine_Tuning/ft_temp_lr3e-05_bs1_ep4"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def chat_with_bot():
    print("Chatbot is ready! Type 'exit' to end the chat.\n")

    while True:
        # Get user input
        user_input = input("You: ")
        
        # Exit condition
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Format the input as per the model's expected format
        text = f"<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{user_input}\n<|im_end|>\n<|im_start|>assistant\n"

        # Tokenize the input
        inputs = tokenizer(text, return_tensors="pt").to(device)

        # Generate a response
        outputs = model.generate(**inputs, max_new_tokens=300)

        # Decode and print the chatbot's response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Bot: {response}")

if __name__ == "__main__":
    chat_with_bot()
