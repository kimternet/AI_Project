from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/support')
def support():
    return render_template('support.html')

@app.route('/music-selection')
def music_selection():
    return render_template('music-selection.html')

@app.route('/difficulty-selection')
def difficulty_selection():
    return render_template('difficulty-selection.html')

if __name__ == '__main__':
    app.run(debug=True)
