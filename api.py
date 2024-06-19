from flask import Flask, request, jsonify
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
import torch

app = Flask(__name__)

# Parameters
batch_size = 96  # May not directly apply to this pipeline
n_epochs = 2      # Not used in this context (relevant for training)
base_LM_model = "deepset/roberta-base-squad2"  # Using the QA model
max_seq_len = 384  # Adjusting to fit model's limit (often 512)
doc_stride = 128
max_query_length = 64

# Load the model and tokenizer
model = AutoModelForQuestionAnswering.from_pretrained(base_LM_model)
tokenizer = AutoTokenizer.from_pretrained(base_LM_model)

# Create the pipeline with custom parameters
nlp = pipeline(
    "question-answering",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
    max_seq_len=max_seq_len,  # Limit input sequence length
    doc_stride=doc_stride,
    max_answer_length=1000,  # Allow longer answers
)

@app.route('/answer', methods=['POST'])
def answer_question():
    data = request.get_json()

    if "question" not in data:
        return jsonify({"error": "Please provide 'question' in the request body"}), 400

    question = data["question"]
    context = data.get("context", "")  

    # Use the pipeline (with or without context)
    result = nlp(question=question, context=context)
    answer = result['answer']

    # Add confidence information
    return jsonify({"answer": answer, "score": result['score']})

if __name__ == '__main__':
    app.run(debug=True)
