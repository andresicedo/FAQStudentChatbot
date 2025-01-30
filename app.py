from transformers import pipeline
import gradio as gr  # Import Gradio for the interface

# Load a text-generation model
chatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium")

# Load the classification model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Customize the bot's knowledge base with predefined responses
faq_responses = {
    "study tips": "Here are some study tips: 1) Break your study sessions into 25-minute chunks (Pomodoro Technique). 2) Test yourself frequently. 3) Stay organized using planners or apps like Notion or Todoist.",
    "resources for studying": "You can find free study resources on websites like Khan Academy, Coursera, and edX. For research papers, check Google Scholar.",
    "how to focus": "To improve focus, try studying in a quiet place, remove distractions like your phone, and use apps like Forest or Focus@Will.",
    "time management tips": "Start by creating a to-do list each morning. Prioritize tasks using methods like Eisenhower Matrix and allocate specific time blocks for each task.",
    "how to avoid procrastination": "Break tasks into smaller steps, set deadlines, and reward yourself after completing milestones. Tools like Trello can help you stay organized."
}

# Define the chatbot's response function
def faq_chatbot(user_input):
    # Classify the user input by passing the FAQ keywords as labels
    classified_user_input = classifier(user_input, candidate_labels=list(faq_responses.keys()))

    # Get the highest confidence score label, ie. the most likely of the FAQ
    predicted_label = classified_user_input["labels"][0]
    confidence_score = classified_user_input["scores"][0]

    # Confidence threshold (adjust if needed)
    threshold = 0.5

    # If the classification confidence is high, return the corresponding FAQ response
    if confidence_score > threshold:
        return faq_responses[predicted_label]
    
    
    # Check if the user's input matches any FAQ keywords
    # for key, response in faq_responses.items():
    #     if key in user_input.lower():
    #         return response

    # If no FAQ match, use the AI model to generate a response
    conversation = chatbot(user_input, max_length=50, num_return_sequences=1)
    return conversation[0]['generated_text']

# Create the Gradio interface
interface = gr.Interface(
    fn=faq_chatbot,  # The function to handle user input
    inputs=gr.Textbox(lines=2, placeholder="Ask me about studying tips or resources..."),  # Input text box
    outputs="text",  # Output as text
    title="Student FAQ Chatbot",
    description="Ask me for study tips, time management advice, or about resources to help with your studies!"
)

# Launch the chatbot and make it public
interface.launch(share=True)
