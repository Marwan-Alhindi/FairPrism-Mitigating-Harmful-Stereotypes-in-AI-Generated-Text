import gpt_2_simple as gpt2
import re

# Start a TensorFlow session
sess = gpt2.start_tf_sess()

# Load the trained model. Replace 'run1' with the name of your checkpoint folder if it's different.
gpt2.load_gpt2(sess, run_name='run1')

# Take user input from the console
prompt_question = input("Enter your prompt: ")

# Generate a response using the user's prompt with limited token generation
generated_text = gpt2.generate(sess, prefix=prompt_question, length=30, return_as_list=True)[0]  # Adjust the length as needed

# Extract the content after "Category:" and before the next newline
match = re.search(r'Category: (.*?)\n', generated_text)
answer = match.group(1).strip() if match else "No match found"

# Print the answer
print("Extracted Answer:", answer)


