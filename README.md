Face Recognition System using OpenCV

This project is a simple **face recognition system** using Python and OpenCV. It allows you to:
- Capture face images using your webcam
- Train a face recognizer model (LBPH)
- Recognize and label faces in real-time

Project Structure
├── start.py # Collect face images from webcam
├── classifier.py # Train the model using collected images
├── name.py # Run real-time face recognition
├── haarcascade_frontalface_default.xml # Face detection model (from OpenCV)
├── classifier.xml # Trained face recognizer (auto generated)
├── data/ # Folder where face images will be saved

Requirements

- Python 3.x
- OpenCV (`pip install opencv-python opencv-contrib-python`)
- NumPy
- Pillow (`pip install pillow`)

---

How to Run

### Step 1: Collect Face Images
Run this command to start capturing face images:

```bash
python start.py
It will open your webcam.
Make sure your face is visible.
100 images will be saved in the data/ folder.
Press Enter key to stop earlier.

Step 2: Train the Model
Once you have collected images, train the model:
python classifier.py
This creates a file called classifier.xml which contains the trained model.

Step 3: Start Face Recognition
Now start real-time face recognition:
python name.py
It will open your webcam and show your face with a name if recognized.
If not recognized, it shows "unknown".
Automatically closes after 2 minutes or when you close the window.

