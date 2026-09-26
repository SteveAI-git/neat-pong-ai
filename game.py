"""
game.py
--------
Core Pygame engine for the NEAT Pong demo: the Paddle and Ball entities,
plus a rule-based (non-NEAT) opponent paddle that plays with deliberately
imperfect, "human-like" tracking so it is beatable but not trivial.

This module has no NEAT-specific code in it — it is reusable physics/
rendering logic that main.py drives.
"""

import random
import pygame

# --------------------------------------------------------------------------
# Screen / shared constants
# --------------------------------------------------------------------------
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

WHITE = (255, 255, 255)
BLACK = (10, 10, 15)
GREEN = (60, 220, 90)      # NEAT AI paddle
RED = (220, 60, 70)        # Rule-based opponent paddle
GREY = (90, 90, 100)


class Paddle:
    """A vertical paddle confined to the screen's height."""

    WIDTH = 15
    HEIGHT = 100

    def __init__(self, x, y, speed=6, color=WHITE):
        self.x = x
        self.y = y
        self.speed = speed
        self.color = color
        self.rect = pygame.Rect(self.x, self.y, self.WIDTH, self.HEIGHT)

    @property
    def center_y(self):
        return self.y + self.HEIGHT / 2

    def move(self, up=True):
        """Move the paddle one step, clamped to the screen bounds."""
        if up:
            self.y -= self.speed
        else:
            self.y += self.speed

        self.y = max(0, min(self.y, SCREEN_HEIGHT - self.HEIGHT))
        self.rect.y = self.y

    def draw(self, win):
        pygame.draw.rect(win, self.color, self.rect, border_radius=4)


class Ball:
    """The Pong ball: handles movement, wall bounces, and resets."""

    RADIUS = 8
    BASE_SPEED_X = 5
    MAX_SPEED_Y = 5

    def __init__(self, x, y):
        self.origin_x = x
        self.origin_y = y
        self.x = x
        self.y = y
        self.speed_x = self.BASE_SPEED_X * random.choice((1, -1))
        self.speed_y = random.uniform(-3, 3)
        self.rect = pygame.Rect(
            self.x - self.RADIUS, self.y - self.RADIUS,
            self.RADIUS * 2, self.RADIUS * 2
        )

    def move(self):
        self.x += self.speed_x
        self.y += self.speed_y

        # Bounce off the top / bottom walls.
        if self.y - self.RADIUS <= 0:
            self.y = self.RADIUS
            self.speed_y *= -1
        elif self.y + self.RADIUS >= SCREEN_HEIGHT:
            self.y = SCREEN_HEIGHT - self.RADIUS
            self.speed_y *= -1

        self.rect.x = self.x - self.RADIUS
        self.rect.y = self.y - self.RADIUS

    def bounce_off_paddle(self, paddle):
        """Reflect the ball horizontally and add spin based on where on
        the paddle it was hit (classic Pong-style angled returns)."""
        self.speed_x *= -1
        offset = (self.y - paddle.center_y) / (Paddle.HEIGHT / 2)
        self.speed_y = offset * self.MAX_SPEED_Y

    def reset(self):
        """Re-center the ball and serve it in a new random direction."""
        self.x = self.origin_x
        self.y = self.origin_y
        self.speed_x = self.BASE_SPEED_X * random.choice((1, -1))
        self.speed_y = random.uniform(-3, 3)
        self.rect.x = self.x - self.RADIUS
        self.rect.y = self.y - self.RADIUS

    def draw(self, win):
        pygame.draw.circle(win, WHITE, (int(self.x), int(self.y)), self.RADIUS)


def opponent_ai_move(paddle, ball, accuracy=0.65, deadzone=12):
    """
    Rule-based opponent paddle with deliberately human-like imperfection,
    so it is beatable and the NEAT paddle can actually score goals.

    - `accuracy` (~60-70%): the odds the opponent "reacts" at all on any
      given frame, simulating human reaction lag.
    - `deadzone` (+/- px): the opponent only bothers moving once the ball
      is meaningfully off-center from its paddle, simulating imprecise
      human tracking rather than pixel-perfect interception.
    """
    if random.random() > accuracy:
        return  # Missed this frame's reaction — simulates human lag.

    diff = ball.y - paddle.center_y

    if diff > deadzone:
        paddle.move(up=False)
    elif diff < -deadzone:
        paddle.move(up=True)
    # else: within the deadzone, stay put (imperfect tracking).