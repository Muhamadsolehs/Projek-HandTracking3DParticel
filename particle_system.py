"""
Module Particle System: Simulasi partikel interaktif menggunakan NumPy dan Pygame.
Mendukung transisi morfologi halus (Smooth Physics Lerp) antar 4 formasi:
1. Kosmos / Nebula Bintang (Open Palm)
2. Jantung Hati Berdenyut (Fist)
3. Teks 'I LOVE YOU' (Peace Sign)
4. Planet Saturnus 3D + Cincin Berputar (Pointing)
"""

import math
import numpy as np
import pygame
from hand_tracker import HandGesture


class ParticleSystem:
    def __init__(self, width=960, height=720, num_particles=2500):
        self.width = width
        self.height = height
        self.cx = width / 2.0
        self.cy = height / 2.0
        self.num_particles = num_particles

        # Posisi, kecepatan, dan target koordinat (NumPy Float32)
        self.pos = np.random.uniform(
            low=[0, 0], high=[width, height], size=(num_particles, 2)
        ).astype(np.float32)
        self.vel = np.zeros((num_particles, 2), dtype=np.float32)
        self.targets = np.copy(self.pos)

        # Warna partikel (RGB Float32 untuk smooth color lerp)
        self.colors = np.ones((num_particles, 3), dtype=np.float32) * 200.0
        self.target_colors = np.copy(self.colors)

        # Ukuran partikel (1 sampai 3 pixel)
        self.sizes = np.random.choice([1, 2, 2, 3], size=num_particles).astype(np.int32)

        # State animasi & rotasi
        self.time = 0.0
        self.current_gesture = HandGesture.OPEN_PALM
        self.saturn_rot_angle = 0.0
        self.hand_offset = np.array([0.0, 0.0], dtype=np.float32)

        # Pre-generate / cache struktur bentuk dasar
        self._init_shapes()

        # Inisialisasi awal ke formasi Kosmos
        self.set_formation(HandGesture.OPEN_PALM)

    def _init_shapes(self):
        """Membuat template koordinat untuk setiap bentuk."""
        N = self.num_particles

        # --- 1. KOSMOS / GALAXY SPIRAL ---
        # 3 lengan spiral galaksi + taburan bintang acak
        indices = np.arange(N)
        arm_idx = indices % 3
        arm_offset = arm_idx * (2.0 * np.pi / 3.0)
        radius = np.power(np.random.uniform(0.05, 1.0, N), 0.7) * (min(self.width, self.height) * 0.42)
        spiral_angle = arm_offset + radius * 0.02 + np.random.normal(0, 0.22, N)

        self.cosmos_x = (radius * np.cos(spiral_angle)).astype(np.float32)
        self.cosmos_y = (radius * np.sin(spiral_angle)).astype(np.float32)
        self.cosmos_radius = radius.astype(np.float32)
        self.cosmos_angle = spiral_angle.astype(np.float32)

        # --- 2. HATI 2D/3D (PARAMETRIC HEART) ---
        # Persamaan Parametrik Jantung Hati:
        # x = 16 * sin^3(t)
        # y = -(13 * cos(t) - 5 * cos(2t) - 2 * cos(3t) - cos(4t))
        t = np.random.uniform(0, 2 * np.pi, N)
        r = np.power(np.random.uniform(0.15, 1.0, N), 0.5)  # Distribusi luas merata
        base_hx = 16.0 * (np.sin(t) ** 3)
        base_hy = -(13.0 * np.cos(t) - 5.0 * np.cos(2 * t) - 2.0 * np.cos(3 * t) - np.cos(4 * t))

        scale_heart = min(self.width, self.height) * 0.021
        self.heart_bx = (r * base_hx * scale_heart).astype(np.float32)
        self.heart_by = (r * base_hy * scale_heart).astype(np.float32)

        # --- 3. SATURNUS 3D (BOLA PLANET + PIRINGAN CINCIN) ---
        n_sphere = int(N * 0.38)
        n_rings = N - n_sphere

        # Bola planet (Fibonacci sphere distribution)
        i_sph = np.arange(n_sphere)
        phi_sph = np.arccos(1.0 - 2.0 * (i_sph + 0.5) / n_sphere)
        golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
        theta_sph = 2.0 * np.pi * i_sph / golden_ratio

        r_planet = min(self.width, self.height) * 0.13
        x_sph = r_planet * np.sin(phi_sph) * np.cos(theta_sph)
        y_sph = r_planet * np.sin(phi_sph) * np.sin(theta_sph)
        z_sph = r_planet * np.cos(phi_sph)

        # Piringan Cincin Saturnus
        ring_r = np.random.uniform(r_planet * 1.45, r_planet * 2.85, n_rings)
        ring_theta = np.random.uniform(0, 2 * np.pi, n_rings)
        x_ring = ring_r * np.cos(ring_theta)
        y_ring = np.random.normal(0, 2.5, n_rings)  # Tebal cincin tipis
        z_ring = ring_r * np.sin(ring_theta)

        self.saturn_3d = np.zeros((N, 3), dtype=np.float32)
        self.saturn_3d[:n_sphere, 0] = x_sph
        self.saturn_3d[:n_sphere, 1] = y_sph
        self.saturn_3d[:n_sphere, 2] = z_sph

        self.saturn_3d[n_sphere:, 0] = x_ring
        self.saturn_3d[n_sphere:, 1] = y_ring
        self.saturn_3d[n_sphere:, 2] = z_ring
        self.saturn_is_ring = np.zeros(N, dtype=bool)
        self.saturn_is_ring[n_sphere:] = True

        # --- 4. TEKS 'I LOVE YOU' ---
        # Rasterisasi glyph teks ke koordinat partikel
        self._init_text_coords()

        # --- 5. TEKS 'I LOVE YOU ALYA' + DUAL HEARTS ---
        self._init_text_alya_coords()

    @staticmethod
    def _extract_text_pixels(surface, threshold=120):
        """
        Mengekstrak koordinat piksel (xs, ys) yang berisi teks dari surface Pygame.
        Menggunakan channel alpha agar huruf terdeteksi solid, tidak terbalik/negatif,
        dan koordinat (x, y) sesuai orientasi layar (tidak tertukar / transposing).
        """
        if surface.get_masks()[3] != 0:
            alpha = pygame.surfarray.array_alpha(surface)
            xs, ys = np.where(alpha > threshold)
        else:
            arr_rgb = pygame.surfarray.array3d(surface)
            brightness = arr_rgb.mean(axis=2)
            xs, ys = np.where(brightness > threshold)
        return xs, ys

    def _init_text_coords(self):
        """Meraster teks 'I LOVE YOU' menjadi koordinat partikel berkualitas tinggi."""
        pygame.font.init()
        font = pygame.font.SysFont("Arial", 90, bold=True)
        text_surf = font.render("I LOVE YOU", True, (255, 255, 255))
        tw, th = text_surf.get_size()

        # Ambil titik pixel teks menggunakan alpha mask yang presisi
        xs, ys = self._extract_text_pixels(text_surf)

        if len(xs) > 0:
            # Normalisasi ke tengah teks
            xs = (xs - tw / 2.0).astype(np.float32)
            ys = (ys - th / 2.0).astype(np.float32)

            # Sample secara acak sebanyak N partikel
            sample_idx = np.random.choice(len(xs), size=self.num_particles, replace=True)
            scale_text = min(self.width, self.height) * 0.0016
            # Tambahkan jitter halus agar partikel tidak menumpuk kaku
            jitter_x = np.random.normal(0, 0.7, self.num_particles)
            jitter_y = np.random.normal(0, 0.7, self.num_particles)

            self.text_x = (xs[sample_idx] * scale_text + jitter_x).astype(np.float32)
            self.text_y = (ys[sample_idx] * scale_text + jitter_y).astype(np.float32)
        else:
            # Fallback jika font gagal
            self.text_x = np.random.uniform(-200, 200, self.num_particles).astype(np.float32)
            self.text_y = np.random.uniform(-50, 50, self.num_particles).astype(np.float32)

    def _init_text_alya_coords(self):
        """Meraster teks 'I LOVE YOU ALYA' lengkap dengan siluet hati pendukung."""
        pygame.font.init()
        font_love = pygame.font.SysFont("Arial", 66, bold=True)
        font_alya = pygame.font.SysFont("Arial", 88, bold=True)

        s_love = font_love.render("I LOVE YOU", True, (255, 255, 255))
        s_alya = font_alya.render("ALYA", True, (255, 255, 255))

        w_love, h_love = s_love.get_size()
        w_alya, h_alya = s_alya.get_size()

        # Ambil titik pixel teks menggunakan alpha channel yang presisi (xs: horizontal, ys: vertikal)
        xs_l, ys_l = self._extract_text_pixels(s_love)
        xs_a, ys_a = self._extract_text_pixels(s_alya)

        scale_t = min(self.width, self.height) * 0.0016
        y_center_love = -68.0 * scale_t
        y_center_alya = 55.0 * scale_t

        x_l = (xs_l - w_love / 2.0).astype(np.float32) * scale_t
        y_l = ((ys_l - h_love / 2.0).astype(np.float32) * scale_t) + y_center_love

        x_a = (xs_a - w_alya / 2.0).astype(np.float32) * scale_t
        y_a = ((ys_a - h_alya / 2.0).astype(np.float32) * scale_t) + y_center_alya

        N = self.num_particles
        n_love = int(N * 0.50)
        n_alya = int(N * 0.36)
        n_hearts = N - n_love - n_alya

        # Alokasi partikel
        sample_l = np.random.choice(len(x_l), size=n_love, replace=True)
        sample_a = np.random.choice(len(x_a), size=n_alya, replace=True)

        jit_x = np.random.normal(0, 0.7, N).astype(np.float32)
        jit_y = np.random.normal(0, 0.7, N).astype(np.float32)

        self.alya_x = np.zeros(N, dtype=np.float32)
        self.alya_y = np.zeros(N, dtype=np.float32)
        self.alya_part_type = np.zeros(N, dtype=np.int32)  # 0: I LOVE YOU, 1: ALYA, 2: HEARTS

        self.alya_x[:n_love] = x_l[sample_l] + jit_x[:n_love]
        self.alya_y[:n_love] = y_l[sample_l] + jit_y[:n_love]
        self.alya_part_type[:n_love] = 0

        self.alya_x[n_love:n_love + n_alya] = x_a[sample_a] + jit_x[n_love:n_love + n_alya]
        self.alya_y[n_love:n_love + n_alya] = y_a[sample_a] + jit_y[n_love:n_love + n_alya]
        self.alya_part_type[n_love:n_love + n_alya] = 1

        # Bentuk dua siluet hati kecil di samping kiri dan kanan nama ALYA
        n_h_each = n_hearts // 2
        t_h = np.random.uniform(0, 2 * np.pi, n_hearts)
        r_h = np.power(np.random.uniform(0.1, 1.0, n_hearts), 0.5)
        hx = 16.0 * (np.sin(t_h) ** 3)
        hy = -(13.0 * np.cos(t_h) - 5.0 * np.cos(2 * t_h) - 2.0 * np.cos(3 * t_h) - np.cos(4 * t_h))
        h_scale = min(self.width, self.height) * 0.0035

        # Pusat posisi masing-masing hati kecil
        self.heart_left_cx = -215.0
        self.heart_right_cx = 215.0
        self.heart_cy = y_center_alya

        self.heart_local_x = (r_h * hx * h_scale).astype(np.float32)
        self.heart_local_y = (r_h * hy * h_scale).astype(np.float32)

        # Hati kiri
        self.alya_x[n_love + n_alya : n_love + n_alya + n_h_each] = (
            self.heart_left_cx + self.heart_local_x[:n_h_each] + jit_x[n_love + n_alya : n_love + n_alya + n_h_each]
        )
        self.alya_y[n_love + n_alya : n_love + n_alya + n_h_each] = (
            self.heart_cy + self.heart_local_y[:n_h_each] + jit_y[n_love + n_alya : n_love + n_alya + n_h_each]
        )

        # Hati kanan
        self.alya_x[n_love + n_alya + n_h_each :] = (
            self.heart_right_cx + self.heart_local_x[n_h_each:] + jit_x[n_love + n_alya + n_h_each :]
        )
        self.alya_y[n_love + n_alya + n_h_each :] = (
            self.heart_cy + self.heart_local_y[n_h_each:] + jit_y[n_love + n_alya + n_h_each :]
        )
        self.alya_part_type[n_love + n_alya :] = 2

    def set_formation(self, gesture):
        """Menyetel formasi target dan skema warna berdasarkan gestur tangan."""
        if gesture == HandGesture.NONE:
            return  # Pertahankan formasi sebelumnya

        self.current_gesture = gesture
        N = self.num_particles

        if gesture == HandGesture.OPEN_PALM:
            # Kosmos: Biru Elektrik / Cyan Bintang
            base_col = np.array([45, 175, 255], dtype=np.float32)
            noise_col = np.random.uniform(-25, 40, (N, 3))
            self.target_colors = np.clip(base_col + noise_col, 30, 255)
            # Sedikit partikel bintang putih terang
            bright_stars = np.random.choice(N, size=int(N * 0.15), replace=False)
            self.target_colors[bright_stars] = [220, 245, 255]

        elif gesture == HandGesture.FIST:
            # Hati: Deep Pink / Magenta Rose yang romantis
            base_col = np.array([255, 45, 135], dtype=np.float32)
            noise_col = np.random.uniform(-25, 45, (N, 3))
            self.target_colors = np.clip(base_col + noise_col, 40, 255)
            # Bagian dalam lebih merah terang
            core_idx = np.random.choice(N, size=int(N * 0.25), replace=False)
            self.target_colors[core_idx] = [255, 110, 180]

        elif gesture == HandGesture.PEACE:
            # Teks I LOVE YOU: Biru Muda Neon / Pure Cyan
            base_col = np.array([0, 235, 255], dtype=np.float32)
            noise_col = np.random.uniform(-20, 20, (N, 3))
            self.target_colors = np.clip(base_col + noise_col, 20, 255)
            # Aksen turquoise
            accent_idx = np.random.choice(N, size=int(N * 0.2), replace=False)
            self.target_colors[accent_idx] = [100, 255, 235]

        elif gesture == HandGesture.POINTING:
            # Saturnus: Emas Jingga / Oranye Kosmik
            # Planet = Emas hangat, Cincin = Oranye/Amber cerah
            colors = np.zeros((N, 3), dtype=np.float32)
            sphere_col = np.array([255, 195, 80], dtype=np.float32)
            ring_col = np.array([255, 140, 25], dtype=np.float32)

            colors[~self.saturn_is_ring] = np.clip(sphere_col + np.random.uniform(-20, 20, (np.sum(~self.saturn_is_ring), 3)), 20, 255)
            colors[self.saturn_is_ring] = np.clip(ring_col + np.random.uniform(-30, 30, (np.sum(self.saturn_is_ring), 3)), 20, 255)
            self.target_colors = colors

        elif gesture == HandGesture.LOVE_ALYA:
            # I LOVE YOU ALYA:
            # - Teks I LOVE YOU: Biru Cyan Neon (0, 235, 255)
            # - Teks ALYA: Romantic Rose Pink / Valentine Magenta (255, 55, 160)
            # - Dua Hati Flanking: Hot Pink / Sparkle White (255, 70, 130)
            colors = np.zeros((N, 3), dtype=np.float32)

            mask_love = (self.alya_part_type == 0)
            colors[mask_love] = np.clip(
                np.array([0, 235, 255]) + np.random.uniform(-20, 20, (np.sum(mask_love), 3)), 20, 255
            )

            mask_alya = (self.alya_part_type == 1)
            colors[mask_alya] = np.clip(
                np.array([255, 50, 160]) + np.random.uniform(-25, 35, (np.sum(mask_alya), 3)), 30, 255
            )
            # Kilau putih-pink di beberapa huruf ALYA
            alya_indices = np.where(mask_alya)[0]
            sparkle_idx = np.random.choice(alya_indices, size=int(len(alya_indices) * 0.2), replace=False)
            colors[sparkle_idx] = [255, 235, 250]

            mask_hearts = (self.alya_part_type == 2)
            colors[mask_hearts] = np.clip(
                np.array([255, 45, 115]) + np.random.uniform(-20, 30, (np.sum(mask_hearts), 3)), 30, 255
            )

            self.target_colors = colors

    def update_targets(self, hand_center=(0.5, 0.5), hand_present=False):
        """
        Memperbarui koordinat target partikel secara dinamis:
        - Mengikuti pergerakan tangan webcam (subtle interactive parallax)
        - Denyut jantung (Heartbeat pulse)
        - Rotasi cincin Saturnus 3D
        - Pusaran kosmos melayang lembut
        """
        self.time += 0.03

        # Interaksi posisi tangan: Geser pusat formasi halus
        if hand_present:
            # hand_center bernilai 0..1, petakan ke offset layar (-80 hingga +80)
            target_ox = (hand_center[0] - 0.5) * 160.0
            target_oy = (hand_center[1] - 0.5) * 140.0
            self.hand_offset[0] += (target_ox - self.hand_offset[0]) * 0.08
            self.hand_offset[1] += (target_oy - self.hand_offset[1]) * 0.08
        else:
            self.hand_offset *= 0.94

        cx = self.cx + self.hand_offset[0]
        cy = self.cy + self.hand_offset[1]

        # 1. KOSMOS (Swirling drift)
        if self.current_gesture == HandGesture.OPEN_PALM:
            rot = self.time * 0.25
            cos_r = np.cos(rot + self.cosmos_angle * 0.08)
            sin_r = np.sin(rot + self.cosmos_angle * 0.08)
            # Sedikit drift nafas (breathing)
            pulse = 1.0 + 0.04 * np.sin(self.time * 0.8)
            tx = cx + (self.cosmos_x * cos_r - self.cosmos_y * sin_r) * pulse
            ty = cy + (self.cosmos_x * sin_r + self.cosmos_y * cos_r) * pulse
            self.targets[:, 0] = tx
            self.targets[:, 1] = ty

        # 2. HATI (Pulsing Heartbeat)
        elif self.current_gesture == HandGesture.FIST:
            # Denyut ritmis lub-dub
            t_pulse = self.time * 4.5
            sin_p = np.sin(t_pulse)
            pulse_factor = 1.0 + 0.09 * (sin_p if sin_p > 0 else sin_p * 0.25)
            # Floating lembut ke atas-bawah
            float_y = np.sin(self.time * 1.5) * 6.0

            self.targets[:, 0] = cx + self.heart_bx * pulse_factor
            self.targets[:, 1] = cy + self.heart_by * pulse_factor + float_y

        # 3. TEKS 'I LOVE YOU' (Floating wave)
        elif self.current_gesture == HandGesture.PEACE:
            # Gelombang lembut melintasi teks
            wave_y = np.sin(self.text_x * 0.015 + self.time * 2.5) * 5.0
            float_y = np.sin(self.time * 1.2) * 8.0

            self.targets[:, 0] = cx + self.text_x
            self.targets[:, 1] = cy + self.text_y + wave_y + float_y

        # 4. SATURNUS 3D (Rotasi Matriks 3D)
        elif self.current_gesture == HandGesture.POINTING:
            self.saturn_rot_angle += 0.018
            theta = self.saturn_rot_angle

            # Matriks Rotasi Y (berputar pada poros)
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            x0 = self.saturn_3d[:, 0]
            y0 = self.saturn_3d[:, 1]
            z0 = self.saturn_3d[:, 2]

            # Rotasi mengelilingi sumbu vertikal Y
            x_rot = x0 * cos_t + z0 * sin_t
            y_rot = y0
            z_rot = -x0 * sin_t + z0 * cos_t

            # Kemiringan poros aksial (Axial Tilt) ~27 derajat mengelilingi sumbu X
            tilt = np.radians(26.7)
            cos_tilt = np.cos(tilt)
            sin_tilt = np.sin(tilt)

            y_tilted = y_rot * cos_tilt - z_rot * sin_tilt
            z_tilted = y_rot * sin_tilt + z_rot * cos_tilt

            # Proyeksi 3D ke 2D (sedikit efek perspektif)
            cam_dist = 650.0
            proj = cam_dist / (cam_dist + z_tilted)

            self.targets[:, 0] = cx + x_rot * proj
            self.targets[:, 1] = cy + y_tilted * proj

        # 5. TEKS 'I LOVE YOU ALYA' + MINI HEARTS (Wave on Top, Pulse on Alya & Hearts)
        elif self.current_gesture == HandGesture.LOVE_ALYA:
            # Denyut romantis khusus nama ALYA
            pulse_alya = 1.0 + 0.06 * np.sin(self.time * 4.2)
            float_y = np.sin(self.time * 1.3) * 6.0
            wave_top = np.sin(self.alya_x * 0.015 + self.time * 2.2) * 4.0

            # Bagian I LOVE YOU (type 0)
            mask_love = (self.alya_part_type == 0)
            self.targets[mask_love, 0] = cx + self.alya_x[mask_love]
            self.targets[mask_love, 1] = cy + self.alya_y[mask_love] + wave_top[mask_love] + float_y

            # Bagian ALYA (type 1) berdenyut anggun pada posisinya
            mask_alya = (self.alya_part_type == 1)
            y_alya_rel = self.alya_y[mask_alya] - self.heart_cy
            self.targets[mask_alya, 0] = cx + self.alya_x[mask_alya] * pulse_alya
            self.targets[mask_alya, 1] = cy + self.heart_cy + y_alya_rel * pulse_alya + float_y

            # Bagian Mini Hearts (type 2) berdenyut mandiri di tempat masing-masing
            N = self.num_particles
            n_love = int(N * 0.50)
            n_alya = int(N * 0.36)
            n_hearts = N - n_love - n_alya
            n_h_each = n_hearts // 2
            pulse_h = 1.0 + 0.12 * np.sin(self.time * 5.0)

            # Hati kiri
            idx_hl = slice(n_love + n_alya, n_love + n_alya + n_h_each)
            self.targets[idx_hl, 0] = cx + self.heart_left_cx + self.heart_local_x[:n_h_each] * pulse_h
            self.targets[idx_hl, 1] = cy + self.heart_cy + self.heart_local_y[:n_h_each] * pulse_h + float_y

            # Hati kanan
            idx_hr = slice(n_love + n_alya + n_h_each, N)
            self.targets[idx_hr, 0] = cx + self.heart_right_cx + self.heart_local_x[n_h_each:] * pulse_h
            self.targets[idx_hr, 1] = cy + self.heart_cy + self.heart_local_y[n_h_each:] * pulse_h + float_y

    def update_physics(self):
        """
        Simulasi fisika partikel:
        - Pegas teredam (Damped Spring / Smooth Lerp) menuju target
        - Interpolasi warna halus
        """
        # Gaya dorong menuju target
        diff = self.targets - self.pos
        # Inersia akselerasi halus
        self.vel += diff * 0.075
        # Redaman kecepatan (damping)
        self.vel *= 0.84
        # Integrasi posisi
        self.pos += self.vel

        # Smooth Lerp Warna
        self.colors += (self.target_colors - self.colors) * 0.07

    def render(self, surface):
        """
        Menggambar partikel ke canvas Pygame dengan efek glow dan motion trail.
        """
        # Motion trail: timpa dengan background gelap semi-transparan
        # Background canvas luar angkasa pekat
        surface.fill((4, 4, 10))

        # Batasi koordinat ke dalam layar untuk menghindari crash
        px = np.clip(self.pos[:, 0], 0, self.width - 1).astype(np.int32)
        py = np.clip(self.pos[:, 1], 0, self.height - 1).astype(np.int32)
        cols = np.clip(self.colors, 0, 255).astype(np.int32)

        # Gambar partikel
        # Partikel ukuran 1 digambar lebih cepat, partikel 2-3 dengan circle
        for i in range(self.num_particles):
            color = (cols[i, 0], cols[i, 1], cols[i, 2])
            size = self.sizes[i]
            x, y = px[i], py[i]

            if size == 1:
                surface.set_at((x, y), color)
            else:
                pygame.draw.circle(surface, color, (x, y), size)
