import cv2
import matplotlib.pyplot as plt

# Load the face detection cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open webcam
cap = cv2.VideoCapture(0)

plt.ion()  # Turn on interactive mode in matplotlib

# Create a figure for the matplotlib display
fig, ax = plt.subplots()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Convert BGR (OpenCV's format) to RGB (matplotlib's format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the image using matplotlib
    ax.clear()
    ax.imshow(rgb_frame)
    ax.axis('off')  # Hide the axes
    plt.draw()
    plt.pause(0.001)  # Pause to allow the plot to update

    # Check for 'q' key press using matplotlib's event handling
    if plt.waitforbuttonpress(timeout=0.001) and plt.get_current_fig_manager().canvas.keypress == 'q':
        break

cap.release()
plt.close()  # Close the matplotlib window
