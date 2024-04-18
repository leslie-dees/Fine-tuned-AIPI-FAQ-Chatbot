from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.get_json()
    user_prompt = data['prompt']
    # Replace with logic
    return jsonify({'body': f'Response to your prompt: {user_prompt}'})

if __name__ == '__main__':
    app.run(debug=True, port=8000)