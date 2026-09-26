"""
main.py
-------
NEAT training loop + rendering manager for the Pong AI demo.

Design notes (read this if you're presenting the project):

- Every genome in the population plays its OWN independent game
  (own green AI paddle, own ball, own red rule-based opponent),
  all stepped forward together, one simulation-frame per pygame frame.
- Only ONE game is drawn on screen at a time — the "spotlight" match.
  IMPORTANT: the spotlight is only reassigned when the genome currently
  being watched actually loses (misses the ball). It is deliberately
  NOT re-picked every frame. Re-picking every frame (an earlier version
  of this file did that) causes the camera to flicker between many
  different genomes' games dozens of times a second, which looks like
  random teleporting/lag and makes it impossible to actually watch
  anything learn. Locking the spotlight to one match at a time gives a
  smooth, continuous game you can watch improve, with clean handoffs
  only when a match genuinely ends.
- A genome's own game ends the moment its AI paddle misses the ball.
  Scoring against the opponent does NOT end the game — the rally just
  resets, so a strong paddle keeps racking up fitness and keeps the
  spotlight.
- A generation ends once every genome has lost, or a frame cap is hit
  (protects against a would-be-immortal early paddle stalling training).
- The best fitness of each completed generation is recorded and plotted
  as a small trend line in the corner — the clearest single piece of
  "yes, it's learning" evidence to show an evaluator.
"""

import os
import sys

import neat
import pygame

from game import (
    SCREEN_WIDTH, SCREEN_HEIGHT, WHITE, BLACK, GREEN, RED, GREY,
    Paddle, Ball, opponent_ai_move,
)

pygame.init()
WIN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("NEAT Pong — Real-Time Evolution Demo")
CLOCK = pygame.time.Clock()
FPS = 60

FONT = pygame.font.SysFont("consolas", 22)
SMALL_FONT = pygame.font.SysFont("consolas", 16)
TINY_FONT = pygame.font.SysFont("consolas", 13)

AI_PADDLE_X = SCREEN_WIDTH - 30      # green, right side, NEAT-controlled
OPPONENT_PADDLE_X = 15               # red, left side, rule-based
MAX_FRAMES_PER_GEN = 1800            # ~30s at 60 FPS safety cap

GENERATION = 0
FITNESS_HISTORY = []   # best fitness of each completed generation

GRAPH_RECT = pygame.Rect(SCREEN_WIDTH - 175, SCREEN_HEIGHT - 90, 160, 75)


def get_inputs(paddle, ball):
    """Build the normalized 5-value input vector fed to each genome's net."""
    return (
        paddle.y / SCREEN_HEIGHT,
        ball.x / SCREEN_WIDTH,
        ball.y / SCREEN_HEIGHT,
        ball.speed_x / 10.0,
        ball.speed_y / 10.0,
    )


def draw_fitness_graph():
    """Small trend line, bottom-right, of best fitness per generation.
    Point at this when presenting — it's the clearest evidence of
    learning, independent of how any single on-screen match looks."""
    pygame.draw.rect(WIN, (25, 25, 32), GRAPH_RECT, border_radius=4)
    pygame.draw.rect(WIN, GREY, GRAPH_RECT, width=1, border_radius=4)
    label = TINY_FONT.render("Best fitness / generation", True, GREY)
    WIN.blit(label, (GRAPH_RECT.x + 4, GRAPH_RECT.y - 16))

    if len(FITNESS_HISTORY) < 2:
        return

    recent = FITNESS_HISTORY[-40:]  # last 40 generations fit comfortably
    lo, hi = min(recent), max(recent)
    span = max(hi - lo, 1e-6)

    pad = 6
    plot_w = GRAPH_RECT.width - pad * 2
    plot_h = GRAPH_RECT.height - pad * 2

    points = []
    for i, val in enumerate(recent):
        x = GRAPH_RECT.x + pad + (i / max(len(recent) - 1, 1)) * plot_w
        y = GRAPH_RECT.y + pad + plot_h - ((val - lo) / span) * plot_h
        points.append((x, y))

    pygame.draw.lines(WIN, GREEN, False, points, 2)


def draw_window(ai_paddle, opp_paddle, ball, ai_score, opp_score, contender_label):
    WIN.fill(BLACK)

    # Center dashed line.
    for y in range(0, SCREEN_HEIGHT, 20):
        pygame.draw.rect(WIN, GREY, (SCREEN_WIDTH // 2 - 1, y, 2, 10))

    ai_paddle.draw(WIN)
    opp_paddle.draw(WIN)
    ball.draw(WIN)

    # --- UI overlay: top-left live metrics ---
    gen_text = FONT.render(f"Gen: {GENERATION}", True, WHITE)
    score_text = FONT.render(f"AI: {ai_score}  |  OPP: {opp_score}", True, WHITE)
    hint_text = SMALL_FONT.render("Green = NEAT AI    Red = rule-based opponent", True, GREY)
    contender_text = SMALL_FONT.render(contender_label, True, GREY)

    WIN.blit(gen_text, (15, 10))
    WIN.blit(score_text, (15, 38))
    WIN.blit(hint_text, (15, 66))
    WIN.blit(contender_text, (15, 88))

    draw_fitness_graph()

    pygame.display.update()


def eval_genomes(genomes, config):
    """
    NEAT fitness function, called once per generation by the population.
    Simulates every genome's game in parallel; renders only the current
    spotlight match, switching spotlight only when that match ends.
    """
    global GENERATION
    GENERATION += 1

    # Fixed-size roster — one "player" dict per genome, at a fixed index
    # for the whole generation. Using a fixed list (with an "alive" flag)
    # instead of deleting finished genomes keeps index numbers stable,
    # which is what lets the spotlight logic below reliably track "the
    # same match" frame after frame instead of getting confused by
    # shifting list positions.
    players = []
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        players.append({
            "genome": genome,
            "net": neat.nn.FeedForwardNetwork.create(genome, config),
            "ai_paddle": Paddle(AI_PADDLE_X, SCREEN_HEIGHT // 2 - Paddle.HEIGHT // 2,
                                 color=GREEN),
            "opp_paddle": Paddle(OPPONENT_PADDLE_X, SCREEN_HEIGHT // 2 - Paddle.HEIGHT // 2,
                                  color=RED),
            "ball": Ball(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2),
            "ai_score": 0,
            "opp_score": 0,
            "alive": True,
        })

    spotlight = 0       # index of the player currently drawn on screen
    frame_count = 0

    while frame_count < MAX_FRAMES_PER_GEN:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        frame_count += 1
        any_alive = False

        for p in players:
            if not p["alive"]:
                continue
            any_alive = True

            genome = p["genome"]
            net = p["net"]
            ai_paddle = p["ai_paddle"]
            opp_paddle = p["opp_paddle"]
            ball = p["ball"]

            # --- NEAT-controlled movement for the AI paddle ---
            output = net.activate(get_inputs(ai_paddle, ball))
            if output[0] > 0.1:
                ai_paddle.move(up=True)
            elif output[0] < -0.1:
                ai_paddle.move(up=False)
            # else: hold position.

            # --- Reward frame-by-frame Y-alignment with the ball ---
            alignment_error = abs(ai_paddle.center_y - ball.y) / (SCREEN_HEIGHT / 2)
            genome.fitness += 0.05 * (1.0 - min(alignment_error, 1.0))

            # --- Rule-based opponent movement ---
            opponent_ai_move(opp_paddle, ball)

            # --- Physics step ---
            ball.move()

            if ball.speed_x > 0 and ball.rect.colliderect(ai_paddle.rect):
                ball.bounce_off_paddle(ai_paddle)
                genome.fitness += 100
            elif ball.speed_x < 0 and ball.rect.colliderect(opp_paddle.rect):
                ball.bounce_off_paddle(opp_paddle)

            if ball.x < 0:
                # Ball passed the opponent -> AI scored a goal. Rally
                # resets, this genome's match continues.
                genome.fitness += 200
                p["ai_score"] += 1
                ball.reset()
            elif ball.x > SCREEN_WIDTH:
                # Ball passed the AI paddle -> this genome's match is over.
                genome.fitness -= 50
                p["opp_score"] += 1
                p["alive"] = False

        # Hand off the spotlight ONLY when the watched match just ended —
        # not every frame. This is what keeps the visible game smooth and
        # continuous instead of flickering between genomes.
        if not players[spotlight]["alive"]:
            alive_indices = [i for i, p in enumerate(players) if p["alive"]]
            if alive_indices:
                spotlight = max(alive_indices, key=lambda i: players[i]["genome"].fitness)

        if not any_alive:
            break

        sp = players[spotlight]
        contender_label = (
            f"Watching contender #{spotlight + 1}/{len(players)}"
            f"   (fitness {sp['genome'].fitness:.1f})"
        )
        draw_window(sp["ai_paddle"], sp["opp_paddle"], sp["ball"],
                    sp["ai_score"], sp["opp_score"], contender_label)

        CLOCK.tick(FPS)

    best_fitness = max(p["genome"].fitness for p in players)
    FITNESS_HISTORY.append(best_fitness)


def run(config_path):
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    try:
        # `None` generations = run indefinitely (fitness_threshold is set
        # unreachably high in the config), stop with Ctrl+C or the window's
        # close button.
        winner = population.run(eval_genomes, None)
        print("\nBest genome:\n", winner)
    except KeyboardInterrupt:
        print("\nTraining stopped by user.")
    finally:
        pygame.quit()


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, "config-feedforward.txt")
    run(config_file)
