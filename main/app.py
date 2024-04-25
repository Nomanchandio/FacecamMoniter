import cv2

def detect_anomalies(frame, fgbg):
    fgmask = fgbg.apply(frame)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return frame

def lambda_handler(event, context):
    fgbg = cv2.createBackgroundSubtractorMOG2()
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return {
            'statusCode': 500,
            'body': 'Could not open camera'
        }
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to grab frame.")
            break
        
        frame = detect_anomalies(frame, fgbg)
        
        cv2.imshow('Camera Feed', frame)
        
        # Wait for 1 millisecond for a key event or window to close
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Camera Feed', cv2.WND_PROP_VISIBLE) < 1:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    return {
        'statusCode': 200,
        'body': 'Processing complete'
    }

if __name__ == '__main__':
    lambda_handler(None, None)