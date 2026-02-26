# Hand Gesture Recognition

✨ Hand Gesture Recognition is an AI-powered hand gesture recognition system trained using YOLOv8 on custom datasets collected and annotated via Roboflow. It can recognize key hand gestures with decent accuracy in real-time.

## Project Structure

- `sig_venv/`: Virtual environment directory
- `static/`: Static files directory
- `templates/`: HTML templates directory
- `.gitignore`: Git ignore file
- `app.py`: Main application file
- `best.pt`: Best trained model weights
- `requirements.txt`: Python dependencies file

## Features

- Real-time hand gesture recognition
- Trained on custom datasets using YOLOv8
- Utilizes Roboflow for dataset collection and annotation

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Azohajutt/Hand-Gesture-Recognition-System
   cd signoraai
   ```

2. Set up the virtual environment:
   ```bash
   python -m venv sig_venv
   source sig_venv/bin/activate  # On Windows use `sig_venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python app.py
   ```

## Usage

- Ensure your webcam is connected and accessible.
- Run the app and point your hand gestures towards the camera.
- The system will detect and classify gestures in real-time.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

[MIT License](LICENSE) (or specify your preferred license)
