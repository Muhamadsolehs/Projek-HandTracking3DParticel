"""
Main Application: Space Gesture Controller & Hand Sensor Monitor
Aplikasi interaktif pengendali sistem partikel luar angkasa menggunakan gestur tangan secara real-time.

Fitur:
- Multi-Window:
  1. 'Hand Sensor Monitor' (OpenCV): Umpan webcam + MediaPipe skeleton + HUD status gestur.
  2. 'Space Gesture Controller' (Pygame): Canvas partikel dinamis 2500+ partikel (Kosmos, Hati, Love You, Saturnus 3D).
- Smooth Physics Interpolation & Color Lerping.
- Sinkronisasi loop FPS tinggi tanpa lagging.
- Tombol keyboard pendukung: [1] Kosmos, [2] Hati, [3] Love You, [4] Saturnus, [Q/ESC] Keluar.
"""

import os
import sys
import time
import cv2
import pygame

# Pastikan lokasi import modular lokal
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hand_tracker import HandTracker, HandGesture
from particle_system import ParticleSystem


def main():
    print("=" * 60)
    print("   SPACE GESTURE CONTROLLER - INTERACTIVE PARTICLE SYSTEM   ")
    print("=" * 60)
    print("[INFO] Menginisialisasi modul kamera dan sistem partikel...")

    # --- 1. Inisialisasi Posisi Jendela ---
    # Posisikan jendela Pygame di sebelah kanan layar
    os.environ['SDL_VIDEO_WINDOW_POS'] = '660,80'

    # --- 2. Inisialisasi Pygame (Space Gesture Controller) ---
    pygame.init()
    pygame.font.init()

    PARTICLE_WIDTH = 960
    PARTICLE_HEIGHT = 720
    screen = pygame.display.set_mode((PARTICLE_WIDTH, PARTICLE_HEIGHT), pygame.DOUBLEBUF)
    pygame.display.set_caption("Space Gesture Controller - Particle Visualizer")
    clock = pygame.time.Clock()

    # Font untuk overlay info di canvas partikel (ringkas & tidak mencolok)
    ui_font_title = pygame.font.SysFont("Trebuchet MS", 14, bold=True)
    ui_font_sub = pygame.font.SysFont("Consolas", 11)

    # Inisialisasi Particle System
    particles = ParticleSystem(width=PARTICLE_WIDTH, height=PARTICLE_HEIGHT, num_particles=2500)

    # --- 3. Inisialisasi OpenCV Webcam (Hand Sensor Monitor) ---
    cap = cv2.VideoCapture(0)
    # Optimasi buffer kamera agar low-latency
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    camera_available = cap.isOpened()
    if not camera_available:
        print("[PERINGATAN] Webcam tidak terdeteksi atau sedang digunakan aplikasi lain.")
        print("[INFO] Mode Simulasi Aktif: Gunakan tombol angka 1, 2, 3, 4, 5 pada keyboard.")

    # Inisialisasi Hand Tracker
    hand_tracker = HandTracker()

    # Nama jendela OpenCV
    cv_win_name = "Hand Sensor Monitor"
    cv2.namedWindow(cv_win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(cv_win_name, 640, 480)
    cv2.moveWindow(cv_win_name, 10, 80)

    # State loop
    running = True
    show_hud = True
    prev_time = time.time()
    fps = 0.0
    last_applied_gesture = HandGesture.OPEN_PALM

    print("[BERHASIL] Kedua jendela siap! Arahkan telapak tangan Anda ke depan webcam.")
    print("  * Tangan Terbuka (Open Palm) : Formasi Kosmos / Nebula")
    print("  * Tangan Mengepal (Fist)      : Formasi Jantung Hati Denyut")
    print("  * Dua Jari (Peace Sign)       : Formasi Teks 'I LOVE YOU'")
    print("  * Satu Jari (Pointing)        : Formasi Planet Saturnus 3D")
    print("  * Gestur Cinta 🤟 / 3 Jari    : Formasi Spesial 'I LOVE YOU ALYA'")
    print("  * Tekan tombol 'H' untuk menyembunyikan/menampilkan HUD.")
    print("  * Tekan tombol 'Q' atau ESC untuk keluar.")

    while running:
        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        # --- A. Event Handling Pygame ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_h:
                    show_hud = not show_hud
                elif event.key == pygame.K_1:
                    particles.set_formation(HandGesture.OPEN_PALM)
                elif event.key == pygame.K_2:
                    particles.set_formation(HandGesture.FIST)
                elif event.key == pygame.K_3:
                    particles.set_formation(HandGesture.PEACE)
                elif event.key == pygame.K_4:
                    particles.set_formation(HandGesture.POINTING)
                elif event.key == pygame.K_5:
                    particles.set_formation(HandGesture.LOVE_ALYA)

        # --- B. Proses Webcam & MediaPipe (Jendela 1) ---
        detected_gesture = HandGesture.NONE
        hand_center = (0.5, 0.5)
        hand_present = False

        if camera_available:
            ret, frame = cap.read()
            if ret and frame is not None:
                # Mirroring frame agar terasa alami seperti cermin
                frame = cv2.flip(frame, 1)

                # Tracking gestur tangan
                frame, detected_gesture, hand_center, hand_present = hand_tracker.process(frame)

                # Gambar overlay HUD ringkas
                frame = hand_tracker.draw_hud(frame, fps=fps, show_hud=show_hud)

                # Tampilkan di jendela OpenCV
                cv2.imshow(cv_win_name, frame)
            else:
                camera_available = False
        else:
            # Placeholder jika tidak ada kamera fisik
            blank_frame = cv2.imread("dummy.png")
            if blank_frame is None:
                import numpy as np
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "WEBCAM OFFLINE / SIMULATION MODE", (50, 200),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 165, 255), 2)
                cv2.putText(blank_frame, "Tekan 1, 2, 3, 4, 5 pada keyboard untuk tes partikel", (50, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.imshow(cv_win_name, blank_frame)

        # --- C. Update Logika Partikel (Jendela 2) ---
        if detected_gesture != HandGesture.NONE and detected_gesture != last_applied_gesture:
            particles.set_formation(detected_gesture)
            last_applied_gesture = detected_gesture

        # Perbarui target koordinat & fisika
        particles.update_targets(hand_center=hand_center, hand_present=hand_present)
        particles.update_physics()

        # Render partikel ke layar
        particles.render(screen)

        # --- D. Overlay HUD di Canvas Partikel ---
        if show_hud:
            gesture_names = {
                HandGesture.OPEN_PALM: ("NEBULA COSMOS", (60, 180, 255)),
                HandGesture.FIST:      ("PULSING HEART", (255, 60, 150)),
                HandGesture.PEACE:     ("I LOVE YOU", (0, 240, 255)),
                HandGesture.POINTING:  ("SATURN 3D ORBIT", (255, 170, 40)),
                HandGesture.LOVE_ALYA: ("I LOVE YOU ALYA", (255, 80, 175)),
            }
            title_text, accent_color = gesture_names.get(particles.current_gesture, ("SPACE CONTROLLER", (200, 200, 200)))

            title_surf = ui_font_title.render(f"MODE: {title_text}", True, accent_color)
            sub_surf = ui_font_sub.render(f"2,500 Partikel | {fps:.0f} FPS | {'Sensor Aktif' if hand_present else 'Standby'}", True, (140, 150, 170))

            hud_badge = pygame.Surface((290, 38), pygame.SRCALPHA)
            hud_badge.fill((8, 8, 14, 150))
            pygame.draw.rect(hud_badge, accent_color, (0, 0, 290, 38), 1, border_radius=6)
            hud_badge.blit(title_surf, (10, 4))
            hud_badge.blit(sub_surf, (10, 21))
            screen.blit(hud_badge, (16, 14))

        # Update tampilan Pygame
        pygame.display.flip()

        # --- E. Polling OpenCV Keyboard Event ---
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            running = False
        elif key in (ord('h'), ord('H')):
            show_hud = not show_hud
        elif key == ord('1'):
            particles.set_formation(HandGesture.OPEN_PALM)
        elif key == ord('2'):
            particles.set_formation(HandGesture.FIST)
        elif key == ord('3'):
            particles.set_formation(HandGesture.PEACE)
        elif key == ord('4'):
            particles.set_formation(HandGesture.POINTING)
        elif key == ord('5'):
            particles.set_formation(HandGesture.LOVE_ALYA)

        # Pertahankan framerate stabil di ~60 FPS
        clock.tick(60)

    # --- Bersihkan Resources ---
    print("\n[INFO] Menutup aplikasi dan membersihkan memori...")
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    print("[SELESAI] Aplikasi ditutup dengan sukses.")


if __name__ == "__main__":
    main()
