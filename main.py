from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig

model_name = "google/flan-t5-base"

tokennizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
config = GenerationConfig(max_new_tokens=200)

#input = tokennizer(text, return_tensors="pt")

while True:
    input_text = input("What do you want to ask about? --->")

    if input_text == "quit":
        break

    #Create tensor tokens
    tokens = tokennizer(input_text, return_tensors="pt")
    outputs = model.generate(**tokens, generation_config=config)
    print(tokennizer.batch_decode(outputs, skip_special_tokens=True))

print("Thanks You Good Bye!")
