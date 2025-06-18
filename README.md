Traffic Close Call
Traffic Close Call is a real-time object detection web application designed to identify and alert for near-miss eIt uses YOLO (You Only Look Once) object detection to analyze proximity between vehicles, pedestrians, anFeatures
- Real-time Object Detection
 Detects cars, trucks, bikes, and pedestrians in video frames using a YOLO model.
- Close Call Detection
 Calculates distance between detected objects and flags events where the distance is below a safe thresho- Web Interface (Flask)
 Interactive interface to upload traffic videos and visualize results.
- Video Frame Processing
 Extracts frames from input video, applies detection, and generates output with bounding boxes and alerts.
Technologies Used
- Frontend: HTML, CSS (Flask templates)
- Backend: Python, Flask
- Object Detection: YOLOv5, OpenCV
- Libraries & Tools: NumPy, ImageIO
Project Structure
Traffic-Close-Call/
■■■ app.py # Main Flask app
■■■ detection.py # Custom object detection & distance logic
■■■ model.py # Model loading and configuration
■■■ utils.py # Helper functions for frame processing
■■■ templates/ # HTML templates for web UI
■■■ static/ # Static files (CSS, images, JS)
■■■ vid2.mp4 # Sample input video
■■■ output/ # Detected frame results
How to Run Locally
1. Clone the repository
 git clone https://github.com/GaganKhandale/Traffic-Close-Call.git
 cd Traffic-Close-Call
2. Install dependencies
 pip install -r requirements.txt
3. Run the Flask app
 python app.py
4. Open in browser
5. Go to http://127.0.0.1:5000/ and upload a video to test detection.
Sample Output
- Frames with bounding boxes and distance alerts.
- Logs of potential close calls with object class info.
Learning & Impact
- Built a complete AI pipeline: detection → logic → web deployment.
- Applied YOLO in real-time applications.
- Explored safety applications in smart city systems.
Acknowledgments
- YOLOv5 pre-trained weights by Ultralytics
- Flask documentation and OpenCV tutorials
