import cv2
import numpy as np

# Simulate 52-card classes
class_names = [f"{v}{s}" for v in ['2','3','4','5','6','7','8','9','10','J','Q','K','A'] for s in ['H','D','C','S']]
from dummy import DummyYOLO  # or paste it inline
model = DummyYOLO(class_names)

cap = cv2.VideoCapture(0)

frame_count = 0

def draw_virtual_blackjack(frame, detections):
    h, w, _ = frame.shape
    table = np.zeros_like(frame)
    table[:] = (0, 100, 0)  # green background

    # Divide detections into player and dealer hands
    player_cards = []
    dealer_cards = []

    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        label = det['label']

        # Use vertical position to assign to player or dealer
        if cy > h // 2:
            player_cards.append((cx, label))
        else:
            dealer_cards.append((cx, label))

    # Sort cards left to right
    player_cards.sort()
    dealer_cards.sort()

    # Draw cards
    def draw_hand(cards, y_pos, label):
        spacing = 100
        start_x = w//2 - len(cards)//2 * spacing
        cv2.putText(table, label, (start_x, y_pos - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
        for i, (_, lbl) in enumerate(cards):
            x = start_x + i * spacing
            cv2.rectangle(table, (x, y_pos), (x+60, y_pos+90), (255,255,255), -1)
            cv2.putText(table, lbl, (x+10, y_pos+50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

    draw_hand(dealer_cards, y_pos=80, label="Dealer")
    draw_hand(player_cards, y_pos=h-180, label="Player")

    return table

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    frame_count += 1
    print(f"📷 Processing frame {frame_count}...")

    results = model.predict(source=frame, conf=0.4, stream=False)
    result = results[0]

    detections = []
    if result.boxes is not None and len(result.boxes) > 0:
        print(f"✅ Detected {len(result.boxes)} objects")
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            x1, y1, x2, y2 = box.tolist()
            class_id = int(cls.item())
            label = model.names[class_id]
            detections.append({"bbox": (x1, y1, x2, y2), "label": label})
    else:
        print("⚠️ No detections")

    virtual_table = draw_virtual_blackjack(frame, detections)

    # Combine original + virtual view
    combined = np.vstack((cv2.resize(frame, (frame.shape[1], 300)),
                          cv2.resize(virtual_table, (frame.shape[1], 300))))

    cv2.imshow("Blackjack Live View", combined)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

