import cv2
import numpy as np
from pyzbar.pyzbar import decode
import time
import math
import sys
import requests

try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False
    print("Для лучшего распознавания кодировок установите chardet: pip install chardet")


class FooEngine:
    """Драйвер робота, отправляющий команды на C++ HTTP-сервер."""
    def __init__(self, server_url="http://192.168.1.100:8080"):
        self.server_url = server_url
        self.command_counter = 0

    def _send_command(self, command, duration_ms=0):
        """Отправляет команду на сервер и дожидается её выполнения (блокирующий вызов)."""
        self.command_counter += 1
        cmd_id = self.command_counter
        payload = {
            "command": command,
            "duration": duration_ms,
            "id": cmd_id
        }
        try:
            resp = requests.post(
                f"{self.server_url}/commands",
                json=payload,
                timeout=5
            )
            if resp.status_code != 200:
                print(f"[ERROR] Сервер вернул {resp.status_code}: {resp.text}")
            else:
                print(f"[OK] {command} (id={cmd_id}, duration={duration_ms} ms)")
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Не удалось отправить команду {command}: {e}")
            return

        if duration_ms > 0:
            time.sleep(duration_ms / 1000.0)

    def forward(self, duration_ms):
        self._send_command("forward", duration_ms)

    def left(self, duration_ms):
        self._send_command("left", duration_ms)

    def right(self, duration_ms):
        self._send_command("right", duration_ms)

    def stop(self):
        self._send_command("stop", 0)


def decode_qr_text(data_bytes):
    """Пытается декодировать байты QR-кода в строку."""
    try:
        text = data_bytes.decode('utf-8')
        if text.isprintable() and len(text) > 0:
            return text
    except UnicodeDecodeError:
        pass

    if HAS_CHARDET:
        detected = chardet.detect(data_bytes)
        encoding = detected.get('encoding')
        if encoding:
            try:
                return data_bytes.decode(encoding)
            except UnicodeDecodeError:
                pass

    for enc in ['cp1251', 'cp866', 'koi8-r', 'iso-8859-5']:
        try:
            text = data_bytes.decode(enc)
            if text.isprintable():
                return text
        except UnicodeDecodeError:
            continue

    return data_bytes.hex()


def get_qr_centers_from_frame(frame):
    """Находит QR-коды на кадре и возвращает словарь {текст: центр}."""
    decoded_objects = decode(frame)
    qr_dict = {}
    for obj in decoded_objects:
        data_bytes = obj.data
        text = decode_qr_text(data_bytes).strip()
        points = obj.polygon
        if len(points) == 4:
            xs = [p.x for p in points]
            ys = [p.y for p in points]
            center = (int(np.mean(xs)), int(np.mean(ys)))
        else:
            rect = obj.rect
            center = (rect.left + rect.width // 2, rect.top + rect.height // 2)
        qr_dict[text] = center
    return qr_dict


def transform_coordinates(center, img_height):
    """Пересчитывает координаты из оконных (y вниз) в обычные (y вверх)."""
    x = center[0]
    y_new = img_height - center[1]
    return (x, y_new)


def vector_angle_signed(v1, v2):
    """Возвращает угол от вектора v1 до v2 в градусах со знаком (против часовой > 0)."""
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    norm1 = math.hypot(v1[0], v1[1])
    norm2 = math.hypot(v2[0], v2[1])
    if norm1 == 0 or norm2 == 0:
        return 0.0
    cos_angle = dot / (norm1 * norm2)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)
    if cross < 0:
        angle_deg = -angle_deg
    return angle_deg


def main():
    # ---------- НАСТРАИВАЕМЫЕ ПАРАМЕТРЫ ----------
    print("Настройка QR-меток (оставьте строку пустой для значения по умолчанию)")
    forward_label = input("Текст для 'Forward' [Forward]: ").strip() or "Forward"
    backward_label = input("Текст для 'Backward' [Backward]: ").strip() or "Backward"
    goal_prefix = input("Префикс целевых QR-кодов [Goal]: ").strip() or "Goal"
    print(f"Используем: Forward='{forward_label}', Backward='{backward_label}', префикс целей='{goal_prefix}'\n")

    # Порог расстояния в пикселях, при котором цель считается достигнутой
    DISTANCE_THRESHOLD = 50

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Не удалось открыть камеру.")
        sys.exit(1)

    engine = FooEngine()

    # Константы движения
    ANGLE_THRESHOLD = 5.0
    TURN_FACTOR = 5.0      # мс на градус
    FORWARD_TIME = 500     # мс
    LOOP_DELAY = 0.05

    # Состояния автомата
    WAITING = 'WAITING'
    MOVING = 'MOVING'
    GOAL_REACHED = 'GOAL_REACHED'

    state = WAITING
    current_angle = 0.0
    goal_reached_name = None

    print("Система запущена. Нажмите 'q' в окне камеры для выхода.")
    print("Убедитесь, что C++ сервер запущен.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить кадр.")
            break

        cv2.imshow("QR Navigation", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # --- Конечный автомат ---
        if state == WAITING:
            qr_centers = get_qr_centers_from_frame(frame)
            required = {forward_label, backward_label}
            if required.issubset(qr_centers.keys()):
                img_height = frame.shape[0]
                transformed = {}
                for name, center in qr_centers.items():
                    transformed[name] = transform_coordinates(center, img_height)

                fwd = transformed[forward_label]
                bwd = transformed[backward_label]
                mid = ((fwd[0] + bwd[0]) / 2, (fwd[1] + bwd[1]) / 2)
                base_vector = (fwd[0] - bwd[0], fwd[1] - bwd[1])

                # Ищем цели по заданному префиксу (без учёта регистра)
                goals = {name: coord for name, coord in transformed.items()
                         if name.lower().startswith(goal_prefix.lower())}

                if goals:
                    best_angle = None
                    best_goal_name = None
                    best_dist = None
                    for goal_name, goal_coord in goals.items():
                        to_goal = (goal_coord[0] - mid[0], goal_coord[1] - mid[1])
                        angle = vector_angle_signed(base_vector, to_goal)
                        dist = math.hypot(to_goal[0], to_goal[1])
                        if best_angle is None or abs(angle) < abs(best_angle):
                            best_angle = angle
                            best_goal_name = goal_name
                            best_dist = dist

                    # Проверка, достигли ли цели
                    if best_dist <= DISTANCE_THRESHOLD:
                        print(f"Цель достигнута: {best_goal_name} (расстояние {best_dist:.1f} px)")
                        state = GOAL_REACHED
                        goal_reached_name = best_goal_name
                        engine.stop()
                        continue   # сразу переходим к обработке достижения
                    else:
                        current_angle = best_angle
                        state = MOVING
                else:
                    time.sleep(LOOP_DELAY)
            else:
                time.sleep(LOOP_DELAY)
            continue

        if state == MOVING:
            if abs(current_angle) > ANGLE_THRESHOLD:
                turn_time = int(abs(current_angle) * TURN_FACTOR)
                if current_angle > 0:
                    print(f"Поворот налево на {turn_time} мс (угол {current_angle:.1f}°)")
                    engine.left(turn_time)
                else:
                    print(f"Поворот направо на {turn_time} мс (угол {current_angle:.1f}°)")
                    engine.right(turn_time)
            else:
                print(f"Едем вперёд на {FORWARD_TIME} мс (угол {current_angle:.1f}°)")
                engine.forward(FORWARD_TIME)
            state = WAITING   # после движения снова ищем метки
            continue

        if state == GOAL_REACHED:
            print(f"\nРобот достиг цели: {goal_reached_name}")
            print("Выберите действие:")
            print("1 – Новая цель")
            print("2 – Возврат домой")
            print("3 – Завершить работу")
            choice = input("Ваш выбор: ").strip()
            if choice == '1':
                new_goal = input("Введите текст (или префикс) целевого QR-кода: ").strip()
                if new_goal:
                    goal_prefix = new_goal
                    print(f"Новый префикс цели: '{goal_prefix}'. Ожидание метки...")
                    state = WAITING
                else:
                    print("Пустой ввод – остаёмся на месте. Повторите выбор.")
                continue
            elif choice == '2':
                home_label = input("Введите текст (или префикс) QR-кода дома: ").strip()
                if home_label:
                    goal_prefix = home_label
                    print(f"Возврат домой, префикс: '{goal_prefix}'. Ожидание метки...")
                    state = WAITING
                else:
                    print("Пустой ввод – остаёмся на месте. Повторите выбор.")
                continue
            elif choice == '3' or choice.lower() == 'q':
                print("Завершение работы.")
                break
            else:
                print("Неверный выбор. Попробуйте снова.")
                continue

    cap.release()
    cv2.destroyAllWindows()
    engine.stop()
    print("Завершение работы.")


if __name__ == "__main__":
    main()