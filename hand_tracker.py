"""
Module Hand Tracker: Menggunakan MediaPipe Hands untuk deteksi landmark tangan
dan klasifikasi gestur secara real-time.
"""

import cv2
import numpy as np
from collections import deque

try:
    import mediapipe.python.solutions.hands as mp_hands
    import mediapipe.python.solutions.drawing_utils as mp_drawing
    import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
except Exception:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles


class HandGesture:
    OPEN_PALM = "OPEN_PALM"       # Cosmos / Nebula Bintang
    FIST = "FIST"                 # Hati / Heart Pulse
    PEACE = "PEACE"               # Teks "I LOVE YOU"
    POINTING = "POINTING"         # Planet Saturnus + Cincin 3D
    LOVE_ALYA = "LOVE_ALYA"       # Teks spesial "I LOVE YOU ALYA" (🤟 / 3 Jari)
    NONE = "NONE"


GESTURE_LABELS = {
    HandGesture.OPEN_PALM: "KOSMOS & BINTANG (Tangan Terbuka)",
    HandGesture.FIST:      "JANTUNG HATI (Mengepal)",
    HandGesture.PEACE:     "I LOVE YOU (Dua Jari)",
    HandGesture.POINTING:  "PLANET SATURNUS (Menunjuk)",
    HandGesture.LOVE_ALYA: "I LOVE YOU ALYA (Gestur Cinta)",
    HandGesture.NONE:      "MENUNGGU GESTUR TANGAN..."
}

GESTURE_COLORS = {
    HandGesture.OPEN_PALM: (255, 200, 50),   # BGR: Cyan-Blue glow
    HandGesture.FIST:      (147, 20, 255),   # BGR: Pink-Magenta
    HandGesture.PEACE:     (255, 230, 0),    # BGR: Bright Cyan
    HandGesture.POINTING:  (30, 160, 255),   # BGR: Orange-Gold
    HandGesture.LOVE_ALYA: (220, 80, 255),   # BGR: Radiant Romantic Purple-Pink
    HandGesture.NONE:      (160, 160, 160)   # BGR: Silver Gray
}


class HandTracker:
    def __init__(self, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.history_len = 5
        self.gesture_history = deque(maxlen=self.history_len)
        self.current_gesture = HandGesture.NONE
        self.hand_center = (0.5, 0.5)
        self.hand_present = False

    def _calc_dist(self, p1, p2):
        """Menghitung jarak Euclidean antara 2 titik (x, y)."""
        return np.hypot(p1.x - p2.x, p1.y - p2.y)

    def _is_finger_extended(self, tip_idx, pip_idx, mcp_idx, landmarks, wrist):
        """
        Mengecek apakah jari (Index, Middle, Ring, Pinky) sedang terbuka/terentang.
        Mengombinasikan jarak dari pergelangan tangan (wrist) dan perbandingan vertikal.
        """
        tip = landmarks[tip_idx]
        pip = landmarks[pip_idx]
        mcp = landmarks[mcp_idx]

        dist_tip_wrist = self._calc_dist(tip, wrist)
        dist_pip_wrist = self._calc_dist(pip, wrist)
        dist_mcp_wrist = self._calc_dist(mcp, wrist)

        # Jari terentang jika ujung jari lebih jauh dari pergelangan tangan dibanding sendi PIP
        # dan ujung jari lebih jauh dibanding sendi dasar (MCP)
        return (dist_tip_wrist > dist_pip_wrist * 1.05) and (dist_tip_wrist > dist_mcp_wrist * 1.15)

    def _is_thumb_extended(self, landmarks, wrist):
        """Mengecek apakah jempol terentang atau melipat."""
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        index_mcp = landmarks[5]
        pinky_mcp = landmarks[17]

        # Jarak ujung jempol ke pangkal telapak tangan dan sendi jari telunjuk
        dist_thumb_pinky = self._calc_dist(thumb_tip, pinky_mcp)
        dist_mcp_pinky = self._calc_dist(thumb_mcp, pinky_mcp)
        dist_tip_wrist = self._calc_dist(thumb_tip, wrist)
        dist_ip_wrist = self._calc_dist(thumb_ip, wrist)

        return (dist_thumb_pinky > dist_mcp_pinky * 1.1) and (dist_tip_wrist > dist_ip_wrist * 0.95)

    def classify_gesture(self, landmarks):
        """
        Klasifikasi gestur tangan berdasarkan status 5 jari:
        - OPEN_PALM: 4 atau 5 jari terentang
        - FIST: Semua 4 jari melipat (fist)
        - PEACE: Telunjuk & Jari Tengah naik, Jari Manis & Kelingking melipat
        - POINTING: Telunjuk naik, Jari Tengah, Manis, Kelingking melipat
        """
        wrist = landmarks[0]

        index_up = self._is_finger_extended(8, 6, 5, landmarks, wrist)
        middle_up = self._is_finger_extended(12, 10, 9, landmarks, wrist)
        ring_up = self._is_finger_extended(16, 14, 13, landmarks, wrist)
        pinky_up = self._is_finger_extended(20, 18, 17, landmarks, wrist)
        thumb_up = self._is_thumb_extended(landmarks, wrist)

        extended_count = sum([index_up, middle_up, ring_up, pinky_up])

        # 1. Gestur Cinta 'I LOVE YOU' 🤟 (Telunjuk & Kelingking naik, Tengah & Manis melipat)
        if index_up and pinky_up and not middle_up and not ring_up:
            return HandGesture.LOVE_ALYA

        # 2. Tiga Jari Terbuka (Index, Middle, Ring up)
        if index_up and middle_up and ring_up and not pinky_up:
            return HandGesture.LOVE_ALYA

        # 3. Peace Sign: Hanya Telunjuk & Tengah terbuka
        if index_up and middle_up and not ring_up and not pinky_up:
            return HandGesture.PEACE

        # 4. Pointing: Hanya Telunjuk terbuka
        if index_up and not middle_up and not ring_up and not pinky_up:
            return HandGesture.POINTING

        # 5. Fist: Semua 4 jari melipat rapat
        if extended_count == 0:
            return HandGesture.FIST

        # 6. Open Palm: Minimal 4 jari terbuka
        if extended_count >= 4 or (extended_count == 3 and thumb_up and not (index_up and pinky_up)):
            return HandGesture.OPEN_PALM

        # Default fallback
        if extended_count >= 4:
            return HandGesture.OPEN_PALM
        elif extended_count == 1 and index_up:
            return HandGesture.POINTING

        return HandGesture.NONE

    def process(self, frame):
        """
        Memproses frame gambar dari OpenCV (BGR), mendeteksi tangan,
        menggambar skeleton, dan mengembalikan gestur serta posisi tangan.
        """
        h, w, _ = frame.shape
        # Konversi BGR ke RGB untuk MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        detected_gesture = HandGesture.NONE
        self.hand_present = False

        if results.multi_hand_landmarks:
            self.hand_present = True
            # Ambil tangan pertama yang terdeteksi
            hand_landmarks = results.multi_hand_landmarks[0]

            # Gambar skeleton dengan style MediaPipe
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Hitung titik tengah tangan (rata-rata posisi wrist dan middle_mcp)
            wrist = hand_landmarks.landmark[0]
            middle_mcp = hand_landmarks.landmark[9]
            cx = (wrist.x + middle_mcp.x) / 2.0
            cy = (wrist.y + middle_mcp.y) / 2.0
            self.hand_center = (cx, cy)

            # Gambar titik fokus tracking di telapak tangan
            px, py = int(cx * w), int(cy * h)
            cv2.circle(frame, (px, py), 9, (0, 255, 255), -1)
            cv2.circle(frame, (px, py), 14, (0, 255, 255), 2)

            detected_gesture = self.classify_gesture(hand_landmarks.landmark)

        # Smooth / debounce gestur menggunakan mode suara terbanyak (majority voting)
        self.gesture_history.append(detected_gesture)
        if len(self.gesture_history) > 0:
            # Ambil gestur yang paling sering muncul dalam history
            counts = {}
            for g in self.gesture_history:
                counts[g] = counts.get(g, 0) + 1
            # Cari gestur terbanyak
            most_common = max(counts, key=counts.get)
            if counts[most_common] >= 3:
                self.current_gesture = most_common

        return frame, self.current_gesture, self.hand_center, self.hand_present

    def draw_hud(self, frame, fps=0.0, show_hud=True):
        """
        Menggambar overlay HUD yang ringkas, modern, dan tidak menghalangi wajah/kamera.
        Mendukung toggle 'H' untuk menyembunyikan/menampilkan HUD secara penuh.
        """
        if not show_hud:
            return frame

        h, w, _ = frame.shape
        overlay = frame.copy()

        # 1. Bar Status Ramping di Atas (Hanya tinggi 34px, semi-transparan elegan)
        bar_y1, bar_y2 = 8, 42
        bar_x1, bar_x2 = 10, w - 10
        cv2.rectangle(overlay, (bar_x1, bar_y1), (bar_x2, bar_y2), (18, 18, 22), -1)

        gesture_col = GESTURE_COLORS.get(self.current_gesture, (160, 160, 160))
        cv2.rectangle(overlay, (bar_x1, bar_y1), (bar_x2, bar_y2), gesture_col, 1)

        # Blending transparan halus agar kamera tetap terlihat jelas
        alpha = 0.65
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Indikator Dot status sensor tangan (Hijau: aktif, Abu-abu: standby)
        dot_col = (50, 230, 120) if self.hand_present else (140, 140, 160)
        cv2.circle(frame, (24, 25), 5, dot_col, -1, cv2.LINE_AA)

        # Label status gestur ringkas
        label = GESTURE_LABELS.get(self.current_gesture, "MENUNGGU GESTUR...")
        cv2.putText(frame, label, 
                    (36, 30), cv2.FONT_HERSHEY_DUPLEX, 0.46, gesture_col, 1, cv2.LINE_AA)

        # Status FPS & Sensor di kanan
        status_hand = "TANGAN AKTIF" if self.hand_present else "STANDBY"
        status_txt = f"FPS: {fps:.0f} | {status_hand}"
        (tw, _), _ = cv2.getTextSize(status_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.putText(frame, status_txt, 
                    (w - 20 - tw, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 200, 210), 1, cv2.LINE_AA)

        # 2. Petunjuk Tombol Minimalis di Bawah (Ramping, tinggi 20px)
        bottom_overlay = frame.copy()
        cv2.rectangle(bottom_overlay, (10, h - 28), (w - 10, h - 8), (15, 15, 20), -1)
        cv2.addWeighted(bottom_overlay, 0.6, frame, 0.4, 0, frame)
        guide_text = "[1] Kosmos  [2] Hati  [3] Peace  [4] Saturnus  [5] ALYA  |  'H': Toggle HUD  'Q': Keluar"
        cv2.putText(frame, guide_text, 
                    (18, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 205, 215), 1, cv2.LINE_AA)

        return frame
