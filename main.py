"""
main.py
-------
NEAT training loop + rendering manager for the Pong AI demo.

Design notes (read this if you're presenting the project):

- Every genome in the population plays its OWN independent game
  (own green AI paddle, own ball, own red rule-based opponent), all
  stepped forward together. Only ONE game is drawn at a time (the
  "spotlight" match), and the spotlight only changes when the genome
  it's currently showing loses — not every frame — so what you watch
  is one smooth, continuous match at a time.
- SIM_SPEED lets game-logic run faster than real-time while still
  rendering every frame at a steady 60 FPS, so training wall-clock
  time drops without the picture becoming choppy (see SIM_SPEED below).
- Paddle movement goes through game.decide_ai_direction(), which uses
  hysteresis instead of a single hard threshold, so paddle motion
  doesn't vibrate near-zero outputs — see game.py for details.
- The best genome ever seen is tracked via NEAT's own
  `population.best_genome` and pickled to disk on every exit path
  (window close, Ctrl+C, and a periodic autosave), so training can be
  interrupted at any time without losing progress. Replay it with
  `python replay.py`.
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

pygame.init()
WIN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("NEAT Pong — Real-Time Evolution Demo")
CLOCK = pygame.time.Clock()
FPS = 60

# How many game-logic steps run per rendered frame. 1 = real-time Pong
# speed. 2 (default) roughly doubles training throughput while still
# drawing a full 60 frames/sec, so play reads as brisk-but-smooth rather
# than choppy. Push higher (3-4) for faster-but-less-watchable training;
# drop to 1 to slow back down to real-time for a careful walkthrough.
SIM_SPEED = 2

FONT = pygame.font.SysFont("consolas", 22)
SMALL_FONT = pygame.font.SysFont("consolas", 16)
TINY_FONT = pygame.font.SysFont("consolas", 13)

AI_PADDLE_X = SCREEN_WIDTH - 30      # green, right side, NEAT-controlled
OPPONENT_PADDLE_X = 15               # red, left side, rule-based
MAX_FRAMES_PER_GEN = 1800            # ~30s of simulated game-time safety cap

GENERATION = 0
FITNESS_HISTORY = []       # best fitness of each completed generation
AVG_FITNESS_HISTORY = []   # mean fitness of each completed generation
ALL_TIME_BEST = float("-inf")

POPULATION = None  # set in run(); used so any exit path can save progress

BEST_GENOME_PATH = os.path.join(os.path.dirname(__file__), "best_genome.pkl")
GRAPH_RECT = pygame.Rect(SCREEN_WIDTH - 175, SCREEN_HEIGHT - 90, 160, 75)


# ---------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------
def save_best_genome(pop, silent=False):
    """Pickle the best genome NEAT has ever seen so far, if any."""
    if pop is None or getattr(pop, "best_genome", None) is None:
        if not silent:
            print("No best genome to save yet — let at least one generation finish first.")
        return
    with open(BEST_GENOME_PATH, "wb") as f:
        pickle.dump(pop.best_genome, f)
    if not silent:
        print(
            f"\nSaved best genome to '{BEST_GENOME_PATH}' "
            f"(fitness={pop.best_genome.fitness:.2f}). Replay it with: python replay.py"
        )


def quit_and_save():
    save_best_genome(POPULATION)
    pygame.quit()
    sys.exit()


# ---------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------
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
    """Trend lines, bottom-right: best fitness (green) and average
    population fitness (white) per generation. This is the clearest
    single piece of evidence that learning is happening — the average
    line rising shows the WHOLE population improving, not just one
    lucky genome."""
    pygame.draw.rect(WIN, (25, 25, 32), GRAPH_RECT, border_radius=4)
    pygame.draw.rect(WIN, GREY, GRAPH_RECT, width=1, border_radius=4)
    label = TINY_FONT.render("Fitness: best (green) / avg (white)", True, GREY)
    WIN.blit(label, (GRAPH_RECT.x - 5, GRAPH_RECT.y - 16))

    if len(FITNESS_HISTORY) < 2:
        return

    best_recent = FITNESS_HISTORY[-40:]
    avg_recent = AVG_FITNESS_HISTORY[-40:]
    lo = min(min(best_recent), min(avg_recent))
    hi = max(max(best_recent), max(avg_recent))
    span = max(hi - lo, 1e-6)

    pad = 6
    plot_w = GRAPH_RECT.width - pad * 2
    plot_h = GRAPH_RECT.height - pad * 2

    def to_points(series):
        pts = []
        for i, val in enumerate(series):
            x = GRAPH_RECT.x + pad + (i / max(len(series) - 1, 1)) * plot_w
            y = GRAPH_RECT.y + pad + plot_h - ((val - lo) / span) * plot_h
            pts.append((x, y))
        return pts

    pygame.draw.lines(WIN, WHITE, False, to_points(avg_recent), 1)
    pygame.draw.lines(WIN, GREEN, False, to_points(best_recent), 2)


def draw_window(ai_paddle, opp_paddle, ball, ai_score, opp_score,
                 contender_label, avg_fitness, is_new_record):
    WIN.fill(BLACK)

    for y in range(0, SCREEN_HEIGHT, 20):
        pygame.draw.rect(WIN, GREY, (SCREEN_WIDTH // 2 - 1, y, 2, 10))

    ai_paddle.draw(WIN)
    opp_paddle.draw(WIN)
    ball.draw(WIN)

    # --- UI overlay: top-left live metrics ---
    gen_text = FONT.render(f"Gen: {GENERATION}", True, WHITE)
    score_text = FONT.render(f"AI: {ai_score}  |  OPP: {opp_score}", True, WHITE)
    hint_text = SMALL_FONT.render("Green = NEAT AI    Red = rule-based opponent", True, GREY)
    contender_text = SMALL_FONT.render(
        f"{contender_label}   |   Gen avg fitness so far: {avg_fitness:.1f}", True, GREY
    )

    WIN.blit(gen_text, (15, 10))
    WIN.blit(score_text, (15, 38))
    WIN.blit(hint_text, (15, 66))
    WIN.blit(contender_text, (15, 88))

    if is_new_record:
        record_text = FONT.render("★ NEW ALL-TIME BEST!", True, GREEN)
        WIN.blit(record_text, (SCREEN_WIDTH // 2 - record_text.get_width() // 2, 10))

    draw_fitness_graph()
    pygame.display.update()


def draw_generation_banner(gen, best_fitness, avg_fitness, hold_frames=45):
    """Brief, clearly-visible pause between generations so a viewer can
    see each generation's result land before the next one starts,
    instead of it being an instant, easy-to-miss jump."""
    for _ in range(hold_frames):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_and_save()

        WIN.fill(BLACK)
        title = FONT.render(f"Generation {gen} complete", True, WHITE)
        stats = SMALL_FONT.render(
            f"Best fitness: {best_fitness:.1f}   |   Avg fitness: {avg_fitness:.1f}",
            True, GREEN,
        )
        sub = SMALL_FONT.render("Breeding next generation...", True, GREY)

        WIN.blit(title, title.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 20)))
        WIN.blit(stats, stats.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 15)))
        WIN.blit(sub, sub.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 45)))

        pygame.display.update()
        CLOCK.tick(FPS)


# ---------------------------------------------------------------------
# NEAT fitness function / simulation
# ---------------------------------------------------------------------
def eval_genomes(genomes, config):
    """
    Called once per generation by the NEAT population. Simulates every
    genome's game in parallel; renders only the current spotlight match,
    switching spotlight only when that match ends.
    """
    global GENERATION, ALL_TIME_BEST
    GENERATION += 1

    # Fixed-size roster: one "player" dict per genome, at a fixed index
    # for the whole generation, so the spotlight logic can reliably track
    # "the same match" across frames instead of index positions shifting.
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
            "last_direction": "hold",
        })

    spotlight = 0       # index of the player currently drawn on screen
    frame_count = 0
    generation_over = False

    while not generation_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_and_save()

        # --- Simulate SIM_SPEED game-logic steps for every rendered frame ---
        for _ in range(SIM_SPEED):
            if frame_count >= MAX_FRAMES_PER_GEN:
                generation_over = True
                break

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

                # --- NEAT-controlled, hysteresis-smoothed paddle movement ---
                output = net.activate(get_inputs(ai_paddle, ball))[0]
                direction = decide_ai_direction(output, p["last_direction"])
                p["last_direction"] = direction
                if direction == "up":
                    ai_paddle.move(up=True)
                elif direction == "down":
                    ai_paddle.move(up=False)
                # "hold": no movement this step.

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
                    # Ball passed the opponent -> AI scored. Rally resets,
                    # this genome's match continues.
                    genome.fitness += 200
                    p["ai_score"] += 1
                    ball.reset()
                elif ball.x > SCREEN_WIDTH:
                    # Ball passed the AI paddle -> this genome's match ends.
                    genome.fitness -= 50
                    p["opp_score"] += 1
                    p["alive"] = False

            # Hand off the spotlight ONLY when the watched match just
            # ended — not every step — so the visible game stays smooth.
            if not players[spotlight]["alive"]:
                alive_indices = [i for i, pl in enumerate(players) if pl["alive"]]
                if alive_indices:
                    spotlight = max(alive_indices, key=lambda i: players[i]["genome"].fitness)

            if not any_alive:
                generation_over = True
                break

        # --- Render once per outer loop, showing the latest state ---
        sp = players[spotlight]
        avg_fitness_live = sum(p["genome"].fitness for p in players) / len(players)
        is_new_record = sp["genome"].fitness > ALL_TIME_BEST
        contender_label = (
            f"Watching contender #{spotlight + 1}/{len(players)}"
            f" (fitness {sp['genome'].fitness:.1f})"
        )
        draw_window(sp["ai_paddle"], sp["opp_paddle"], sp["ball"],
                    sp["ai_score"], sp["opp_score"], contender_label,
                    avg_fitness_live, is_new_record)

        CLOCK.tick(FPS)

    best_fitness = max(p["genome"].fitness for p in players)
    avg_fitness = sum(p["genome"].fitness for p in players) / len(players)
    FITNESS_HISTORY.append(best_fitness)
    AVG_FITNESS_HISTORY.append(avg_fitness)
    if best_fitness > ALL_TIME_BEST:
        ALL_TIME_BEST = best_fitness

    # Periodic safety-net autosave, independent of how the run eventually
    # ends — cheap insurance against a crash mid-training.
    if GENERATION % 10 == 0:
        save_best_genome(POPULATION, silent=True)

    draw_generation_banner(GENERATION, best_fitness, avg_fitness)


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
def run(config_path):
    global POPULATION

    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    population = neat.Population(config)
    POPULATION = population
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    try:
        # `None` generations = run indefinitely (fitness_threshold is set
        # unreachably high in the config); stop with Ctrl+C or the
        # window's close button — both paths save progress (see above).
        winner = population.run(eval_genomes, None)
        save_best_genome(population)
        print("\nBest genome:\n", winner)
    except KeyboardInterrupt:
        print("\nTraining stopped by user.")
        save_best_genome(population)
    finally:
        pygame.quit()


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, "config-feedforward.txt")
    run(config_file)
