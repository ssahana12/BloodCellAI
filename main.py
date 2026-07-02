from camera import get_frame
import cv2

def main():
    print("Starting BloodCellAI... Press Q to quit")

    while True:
        frame = get_frame()

        if frame is None:
            print("No frame from camera")
            break

        cv2.imshow("BloodCellAI - Live Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
