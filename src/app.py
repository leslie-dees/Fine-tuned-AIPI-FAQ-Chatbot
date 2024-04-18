from flask import Flask, request, jsonify, send_from_directory
from main import main  # Import your main function from the script provided
import json

app = Flask(__name__, static_url_path='')

# Serve the HTML file on the homepage
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# Process chat requests
@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.json
        conversation = data.get("conversation")
        last_user_message = conversation[-1]["text"] if conversation else ""
        
        # Call your pipeline's main function to get the response
        response = main(last_user_message)
        
        # You can format the response in the structure your front end expects
        return jsonify({"body": response})
    except Exception as e:
        print(e)
        return jsonify({"body": "Sorry, I didn't get that."}), 500

# Add additional routes if necessary

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
