import gpt_2_simple as gpt2
from collections import Counter
import re

# Initialize the model
sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess, run_name='run1')

def get_model_output(prompt):
    generated_text = gpt2.generate(sess, prefix=prompt, length=50, return_as_list=True)[0]
    return generated_text

def extract_categories(output):
    # Extract categories from the model's output
    match = re.search(r'Category: \[(.*?)\]', output)
    categories = match.group(1).split(",") if match else []
    # Strip spaces and remove quotes
    categories = [cat.strip().replace("'", "") for cat in categories]
    return categories

def main():
    category_counts = Counter()

    with open('prompts_preprocessed.txt', 'r') as file:
        for line in file:
            prompt = line.strip()
            if prompt:  # Check if the line is not empty
                model_output = get_model_output(prompt)
                categories = extract_categories(model_output)
                category_counts.update(categories)

    print("Categories and their counts:", category_counts)

main()