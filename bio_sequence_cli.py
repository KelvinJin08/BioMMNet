import argparse
import torch
from transformers import pipeline
from models.progen.sample import sample, truncate
from models.progen.modeling_progen import ProGenForCausalLM
from tokenizers import Tokenizer
from torch.optim import Adam
import re

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# protein sequence model configuration
top_p = 0.9
temp = 1.5
num_return_sequences = 1
context = "1"  # eg.："MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGH"
model_checkpoint_path = './checkpoints/progen2-small'  # 模型checkpoint的路径
tokenizer_file = 'tokenizer.json'  # Tokenizer文件的路径

# initialize protein sequence model
protein_gen = ProGenForCausalLM.from_pretrained(model_checkpoint_path).to(device)
tokenizer = Tokenizer.from_file(tokenizer_file)

pad_token_id = tokenizer.encode('<|pad|>').ids[0]  # pad token id


# A simplified function to generate a protein sequence
def generate_protein_sequence(context, length, num_return_sequences, top_p, temp):
    # This is a simplified example. In a real application, this would be more complex.
    generated_sequences = sample(
        device=device,
        model=protein_gen,
        tokenizer=tokenizer,
        context=context,
        max_length=length,
        num_return_sequences=num_return_sequences,
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

def understand_command(command):
    nlp = pipeline('text-generation', model='gpt2')
    response = nlp(command, max_length=50)
    return response[0]['generated_text']

def extract_parameters(request):
    length_match = re.search(r"length (?:is )?(\d+)|(\d+) length", request, re.IGNORECASE)
    length = int(length_match.group(1) if length_match.group(1) else length_match.group(2)) if length_match else 64

    temp_match = re.search(r"temperature (?:is )?([\d\.]+)|([\d\.]+) temperature", request, re.IGNORECASE)
    temperature = float(temp_match.group(1) if temp_match.group(1) else temp_match.group(2)) if temp_match else 1.0

    return length, temperature

def main():
    parser = argparse.ArgumentParser(description="Generate biological sequences from natural language commands.")
    parser.add_argument('command', type=str, help="The natural language command for generating a biological sequence.")
    args = parser.parse_args()

    understood_text = understand_command(args.command)
    print(f"Understood: {understood_text}")

    length, temp = extract_parameters(understood_text)
    if "protein" in understood_text.lower():
        sequence = generate_protein_sequence(context, length, num_return_sequences, top_p, temp)
        print(f"Generated protein sequence: {sequence}")
    elif "rna" in understood_text.lower():
        sequence = generate_rna_sequence(length)
        print(f"Generated RNA sequence: {sequence}")
    else:
        print("Could not understand the command or it's not implemented.")

if __name__ == "__main__":
    main()
