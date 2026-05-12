import cv2
import time
import math
import requests
from pyzbar.pyzbar import decode


class RobotEngine:
    def __init__(self, server_url="http://192.168.1.100:8080"):
        self.server_url = server_url
        self.cmd_id = 0

    def send(self, command, duration_ms=0):
        self.cmd_id += 1
        payload = {"command": command, "duration": duration_ms, "id": self.cmd_id}
        try:
            requests.post(f"{self.server_url}/commands", json=payload, timeout=2)
        except requests.RequestException:
            pass

    def forward(self, ms): self.send("forward", ms)
    def left(self, ms):    self.send("left", ms)
    def right(self, ms):   self.send("right", ms)
    def stop(self):        self.send("stop", 0)



def get_qr_centers(frame):
    result = {}
    for obj in decode(frame):
        text = obj.data.decode("utf-8", errors="ignore").strip()
        pts = obj.polygon
        if len(pts) == 4:
            cx = sum(p.x for p in pts) // 4
            cy = sum(p.y for p in pts) // 4
        else:
            r = obj.rect
            cx = r.left + r.width // 2
            cy = r.top + r.height // 2
        result[text] = (cx, cy)
    return result



def signed_angle(v1, v2):
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    cross = v1[0]*v2[1] - v1[1]*v2[0]
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    angle = math.degrees(math.acos(max(-1, min(1, dot/(n1*n2)))))
    return -angle if cross < 0 else angle


def choose_target(qrs, forward_label, backward_label, target_label):
    if forward_label not in qrs or backward_label not in qrs:
        return None

    if target_label not in qrs:
        return None

    fwd = qrs[forward_label]
    bwd = qrs[backward_label]
    tgt = qrs[target_label]

    mid = ((fwd[0] + bwd[0]) / 2, (fwd[1] + bwd[1]) / 2)
    base_vec = (fwd[0] - bwd[0], fwd[1] - bwd[1])
    vec = (tgt[0] - mid[0], tgt[1] - mid[1])

    dist = math.hypot(*vec)
    angle = signed_angle(base_vec, vec)

    return angle, dist


def choose_camera(max_tested=5):
    available = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
        cap.release()

    if not available:
        print("Камеры не найдены.")
        exit(1)

    print(f"Доступные камеры: {available}")
    choice = input(f"Выберите индекс камеры [{available[0]}]: ").strip()

    try:
        cam_id = int(choice) if choice else available[0]
        if cam_id not in available:
            raise ValueError
    except ValueError:
        cam_id = available[0]

    return cam_id


def main():
    forward_label = input("Forward QR [Forward]: ").strip() or "Forward"
    backward_label = input("Backward QR [Backward]: ").strip() or "Backward"
    target_label = input("Target QR: ").strip()
    home_label = input("Home QR: ").strip()

    cam_id = choose_camera()
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    engine = RobotEngine()

    ANGLE_THR = 5
    TURN_K = 5
    FORWARD_MS = 400
    GOAL_DIST = 50
    MOVE_DELAY = 0.5
    last_move = 0

    print("Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("QR Nav", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        now = time.time()
        if now - last_move < MOVE_DELAY:
            continue

        qrs = get_qr_centers(frame)
        res = choose_target(qrs, forward_label, backward_label, target_label)

        if not res:
            continue

        angle, dist = res

        if dist < GOAL_DIST:
            print(f"\nДостигнута цель: {target_label}")
            engine.stop()

            while True:
                print("\n1 — Новая цель")
                print("2 — Вернуться домой")
                print("q — Выход")
                ch = input("> ").strip().lower()

                if ch == '1':
                    target_label = input("Текст нового QR цели: ").strip()
                    break
                elif ch == '2':
                    target_label = home_label
                    print("Возвращаемся домой...")
                    break
                elif ch == 'q':
                    cap.release()
                    cv2.destroyAllWindows()
                    engine.stop()
                    return

        else:
            if abs(angle) > ANGLE_THR:
                ms = int(abs(angle) * TURN_K)
                if angle > 0:
                    engine.left(ms)
                else:
                    engine.right(ms)
            else:
                engine.forward(FORWARD_MS)

            last_move = now

    cap.release()
    cv2.destroyAllWindows()
    engine.stop()


if __name__ == "__main__":
    main()