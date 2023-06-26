import cv2
import os

# Directory path to save the images
#save_directory = "C:/Users/ririk/Desktop/20230406_TEMP/DLIP_extra_project/raw_datasets/scissor"

# For test
save_directory = "C:/Users/ririk/Desktop/20230406_TEMP/DLIP_extra_project/rock_scissor_paper/test/rock"

# Create the directory if it doesn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Open the webcam
cap = cv2.VideoCapture(0)

# Image counter
image_counter = 1

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Show the frame
    cv2.imshow("Webcam", frame)

    # Press "s" key to save the current frame
    if cv2.waitKey(1) & 0xFF == ord("s"):
        # Set the file name as a number
        filename = str(image_counter) + ".jpg"
        filepath = os.path.join(save_directory, filename)

        # Save the image
        cv2.imwrite(filepath, frame)
        print(f"Saved image: {filename}")

        # Increase the image counter
        image_counter += 1

    # Press "q" key to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam
cap.release()
cv2.destroyAllWindows()
