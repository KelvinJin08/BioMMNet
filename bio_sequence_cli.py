import argparse
import torch
from transformers import pipeline
from models.progen.sample import sample, truncate
from models.progen.modeling_progen import ProGenForCausalLM
from tokenizers import Tokenizer
from torch.optim import Adam

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# protein sequence model configuration
sequence_batch_size = 16
num_epochs = 2000
max_length = 512
top_p = 0.9
temp = 1.5
context = "1"  # eg.："MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGH"
model_checkpoint_path = './checkpoints/progen2-small'  # 模型checkpoint的路径
tokenizer_file = 'tokenizer.json'  # Tokenizer文件的路径
learning_rate = 0.00001

# initialize protein sequence model
protein_gen = ProGenForCausalLM.from_pretrained(model_checkpoint_path).to(device)
tokenizer = Tokenizer.from_file(tokenizer_file)
optimizer = Adam(protein_gen.parameters(), lr=learning_rate)

pad_token_id = tokenizer.encode('<|pad|>').ids[0]  # pad token id


# A simplified function to generate a protein sequence
def generate_protein_sequence(length):
    # This is a simplified example. In a real application, this would be more complex.
    generated_sequences = sample(
        device=device,
        model=protein_gen,
        tokenizer=tokenizer,
        context=context,
        max_length=length,
        num_return_sequences=1,  # 每次生成一个序列
        top_p=top_p,
        temp=temp,
        pad_token_id=pad_token_id
    )
    sequences = [truncate(generated_sequence[1:], terminals=['1', '2']) for generated_sequence in generated_sequences]
    return sequences
    # return 'M' + 'A' * (length - 1)


# A simplified function to generate an RNA sequence
def generate_rna_sequence(length):
    # This is a simplified example. In a real application, this would be more complex.

    return 'AUCG' * (length // 4) + 'AUCG'[:length % 4]


# Function to understand natural language commands using a language model
def understand_command(command):
    # Using the transformers pipeline for natural language understanding
    nlp = pipeline('text-generation', model='gpt2')  # Assuming GPT-2 for demonstration
    response = nlp(command, max_length=50)  # Limiting the length of the generated text
    return response[0]['generated_text']


# Main function to parse command-line input and process the request
def main():
    parser = argparse.ArgumentParser(description="Generate biological sequences from natural language commands.")
    parser.add_argument('command', type=str, help="The natural language command for generating a biological sequence.")
    args = parser.parse_args()

    # Understanding the command
    command = args.command
    understood_text = understand_command(command)
    print(f"Understood: {understood_text}")

    # Based on the understood command, call the respective generation function
    # The logic here needs to be customized based on the structure of the understood command
    # The following is a simplified example:
    if "protein" in understood_text.lower():
        length = 1024  # The length should be extracted from the understood text
        sequence = generate_protein_sequence(length)
        print(f"Generated protein sequence: {sequence}")
    elif "rna" in understood_text.lower():
        length = 1024  # The length should be extracted from the understood text
        sequence = generate_rna_sequence(length)
        print(f"Generated RNA sequence: {sequence}")
    else:
        print("Could not understand the command or it's not implemented.")


if __name__ == "__main__":
    main()
