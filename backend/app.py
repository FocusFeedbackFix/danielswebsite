from flask import Flask, Response, jsonify
from detection import detect_caps, gen_frames

app = Flask(__name__)

@app.route('/video_feed')
def video_feed():
    # Returns live video feed (MJPEG stream)
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect')
def detect():
    # Returns detected cap numbers as JSON
    caps = detect_caps()
    return jsonify({'caps': caps})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)