"""
Build a diverse set of 720p text-to-video prompts (caption set).
The text is only used as the condition for manifold-reward finetuning; no paired videos are
needed, so the prompts just need to be "diverse and natural".
Combinatorial generation + dedup + deterministic shuffle, ~300 prompts, written to captions.txt.
"""
import os
import argparse
import hashlib

# Coherent scene cores (subject + action + setting already bound, semantically natural),
# then overlay lighting / shot / style variations.
SCENES = [
    "a professional chef tosses vegetables in a flaming wok in a busy kitchen",
    "an elderly fisherman casts his net from a wooden boat on a calm sea",
    "a little boy in a red jacket splashes through puddles on a rainy street",
    "a ballerina spins gracefully on a dimly lit stage",
    "a street musician plays the violin on a cobblestone square",
    "a golden retriever runs across a sunlit grassy meadow",
    "a black cat stretches and yawns on a windowsill",
    "a galloping horse kicks up dust across an open plain",
    "a flock of birds wheels across an orange sunset sky",
    "a school of tropical fish drifts over a vibrant coral reef",
    "a butterfly slowly opens its wings on a blooming flower",
    "a red fox steps carefully through fresh snow in a forest",
    "a vintage sports car speeds along a coastal highway",
    "a sailboat glides across a glittering bay at dawn",
    "a hot air balloon rises over a patchwork of green fields",
    "a steam train winds through a misty mountain valley",
    "a waterfall cascades down mossy rocks into a clear pool",
    "cherry blossoms fall gently in a quiet temple garden",
    "a bonfire crackles and sends sparks into the night sky",
    "ocean waves crash against jagged coastal cliffs",
    "a barista pours steamed milk into a latte in a cozy cafe",
    "a potter shapes wet clay spinning on a wheel",
    "a blacksmith hammers a glowing piece of metal on an anvil",
    "a couple dances slowly under string lights in a courtyard",
    "a lone hiker climbs a rocky ridge above the clouds",
    "a surfer rides the curl of a large breaking wave",
    "a figure skater glides and spins across an ice rink",
    "a glassblower shapes a glowing bubble of molten glass",
    "a painter brushes bold strokes onto a large canvas",
    "a young woman with long hair walks through an autumn park with falling leaves",
    "steaming dumplings are lifted from a bamboo steamer in a night market",
    "raindrops ripple across the surface of a quiet pond",
    "a hummingbird hovers and sips from a bright red flower",
    "a chef plates a delicate dessert with tweezers in a fine kitchen",
    "a kite dances in the wind above a wide sandy beach",
    "a baker pulls a tray of golden bread from a stone oven",
    "a deer grazes at the edge of a foggy forest clearing",
    "a city skyline lights up as dusk falls over the river",
    "a child blows a stream of soap bubbles in a garden",
    "a violinist performs under a single spotlight in a grand hall",
]
LIGHT = [
    "golden hour sunlight", "soft morning light", "dramatic rim lighting", "warm candlelight",
    "cool blue twilight", "harsh midday sun", "colorful neon glow", "gentle overcast light",
    "flickering firelight", "moonlight",
]
CAMERA = [
    "slow dolly-in close-up", "sweeping aerial drone shot", "smooth tracking shot",
    "handheld follow shot", "slow-motion close-up", "wide establishing shot",
    "orbiting camera move", "static locked-off shot", "tilt-up reveal", "shallow depth of field macro",
]
STYLE = [
    "cinematic, highly detailed, photorealistic", "rich textures, sharp focus, film grain",
    "vivid colors, crisp detail, documentary style", "natural lighting, lifelike detail",
    "ultra-detailed, realistic motion blur", "high dynamic range, fine micro-textures",
]

TEMPLATES = [
    "{scene}, lit by {light}. {cam}. {style}.",
    "{scene}. Bathed in {light}, captured with a {cam}. {style}.",
    "{cam}: {scene}, under {light}. {style}.",
]


def gen(n, seed=0):
    # deterministic pseudo-random (hash-based), no global state from the random module
    out = []
    seen = set()
    i = 0
    while len(out) < n and i < n * 50:
        h = int(hashlib.md5(f"{seed}-{i}".encode()).hexdigest(), 16)
        i += 1
        t = TEMPLATES[h % len(TEMPLATES)]
        s = t.format(
            scene=SCENES[(h // 7) % len(SCENES)],
            light=LIGHT[(h // 19) % len(LIGHT)],
            cam=CAMERA[(h // 23) % len(CAMERA)],
            style=STYLE[(h // 29) % len(STYLE)],
        )
        s = s[0].upper() + s[1:]
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/ -> repo root
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default=os.path.join(repo_root, "data", "captions.txt"),
                    help="output caption file; default is <repo>/data/captions.txt regardless of cwd")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    caps = gen(args.n, args.seed)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)   # create parent dir(s) so --out can point anywhere
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(caps) + "\n")
    print(f"wrote {len(caps)} captions -> {args.out}")
    for c in caps[:5]:
        print(" •", c)
