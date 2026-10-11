import numpy as np
import pygame
from hand_tracker import HandTracker, HandGesture
from particle_system import ParticleSystem

print("[TEST 1] Testing ParticleSystem initialization...")
ps = ParticleSystem(width=800, height=600, num_particles=2000)
print(" - Particles initialized:", ps.num_particles)
print(" - Current gesture:", ps.current_gesture)

print("\n[TEST 2] Testing shape transitions...")
for g in [HandGesture.OPEN_PALM, HandGesture.FIST, HandGesture.PEACE, HandGesture.POINTING, HandGesture.LOVE_ALYA]:
    ps.set_formation(g)
    ps.update_targets(hand_center=(0.4, 0.6), hand_present=True)
    ps.update_physics()
    print(f" - Transition to {g}: targets shape={ps.targets.shape}, colors shape={ps.colors.shape}")

print("\n[TEST 3] Testing Pygame rendering surface...")
surf = pygame.Surface((800, 600))
ps.render(surf)
print(" - Render succeeded!")

print("\n[TEST 4] Testing HandTracker initialization...")
ht = HandTracker()
print(" - HandTracker initialized successfully!")

print("\nALL SYSTEM VERIFICATIONS PASSED!")
