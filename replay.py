"""
replay.py
---------
Loads a genome pickled by main.py (default: best_genome.pkl) and lets you
watch it play continuously against the rule-based opponent, with no
training involved. Uses the exact same movement logic (decide_ai_direction)
as training, so the replayed behavior matches what the genome actually
earned its fitness doing.

Usage:
    python replay.py                  # uses best_genome.pkl
    python replay.py my_genome.pkl    # or a specific saved file
"""

import os
import pickle
import sys

import neat
import pygame

from game import (
    SCREEN_WIDTH, SCREEN_HEIGHT, WHITE, BLACK, GREEN, RED, GREY,
    Paddle, Ball, opponent_ai_move, decide_ai_direction,
)

AI_PADDLE_X = SCREEN_WIDTH - 30
OPPONENT_PADDLE_X = 15


def get_inputs(paddle, ball):
    return (
        paddle.y / SCREEN_HEIGHT,
        ball.x / SCREEN_WIDTH,
        ball.y / SCREEN_HEIGHT,
        ball.speed_x / 10.0,
        ball.speed_y / 10.0,
    )


def main():
    local_dir = os.path.dirname(__file__)
    genome_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(local_dir, "best_genome.pkl")
    config_path = os.path.join(local_dir, "config-feedforward.txt")

    if not os.path.exists(genome_path):
        print(f"Couldn't find '{genome_path}'.")
        print("Run `python main.py`, let it train a bit, then close the window "
              "(or Ctrl+C) — that saves a genome automatically.")
        return

    config = neat.config.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path,
    )

    with open(genome_path, "rb") as f:
        genome = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    pygame.init()
    win = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(f"NEAT Pong — Replay (trained fitness: {genome.fitness:.1f})")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 22)
    small_font = pygame.font.SysFont("consolas", 16)

    ai_paddle = Paddle(AI_PADDLE_X, SCREEN_HEIGHT // 2 - Paddle.HEIGHT // 2, color=GREEN)
    opp_paddle = Paddle(OPPONENT_PADDLE_X, SCREEN_HEIGHT // 2 - Paddle.HEIGHT // 2, color=RED)
    ball = Ball(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    ai_score = opp_score = 0
    last_direction = "hold"

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        output = net.activate(get_inputs(ai_paddle, ball))[0]
        direction = decide_ai_direction(output, last_direction)
        last_direction = direction
        if direction == "up":
            ai_paddle.move(up=True)
        elif direction == "down":
            ai_paddle.move(up=False)

        opponent_ai_move(opp_paddle, ball)
        ball.move()

        if ball.speed_x > 0 and ball.rect.colliderect(ai_paddle.rect):
            ball.bounce_off_paddle(ai_paddle)
        elif ball.speed_x < 0 and ball.rect.colliderect(opp_paddle.rect):
            ball.bounce_off_paddle(opp_paddle)

        if ball.x < 0:
            ai_score += 1
            ball.reset()
        elif ball.x > SCREEN_WIDTH:
            opp_score += 1
            ball.reset()

        win.fill(BLACK)
        for y in range(0, SCREEN_HEIGHT, 20):
            pygame.draw.rect(win, GREY, (SCREEN_WIDTH // 2 - 1, y, 2, 10))
        ai_paddle.draw(win)
        opp_paddle.draw(win)
        ball.draw(win)

        win.blit(font.render(f"AI: {ai_score}  |  OPP: {opp_score}", True, WHITE), (15, 10))
        win.blit(small_font.render(
            f"Replay mode — trained fitness {genome.fitness:.1f}", True, GREY), (15, 40))

        pygame.display.update()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
