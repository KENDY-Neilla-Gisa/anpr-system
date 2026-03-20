"""
Camera Module
Simple utility to verify camera connectivity and display feed.
"""
import cv2


def test_camera(device_index=0):
    """
    Open camera and display video feed for verification.
    Press 'q' to exit the preview window.
    """
    capture = cv2.VideoCapture(device_index)

    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera at index {device_index}")

    print(f"Camera opened successfully at index {device_index}")
    print("Press 'q' to exit")

    while True:
        success, frame = capture.read()
        if not success:
            print("Failed to capture frame")
            break

        cv2.imshow("Camera Feed - Press 'q' to exit", frame)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()
    print("Camera test completed")


def main():
    test_camera()


if __name__ == "__main__":
    main()
