from PIL import Image, ImageDraw, ImageOps

from main_rl_env import RL_ENV


def plot_state_space(d, states):
    size = 10
    print(states[8], states[9])
    d.rectangle([(states[8], states[9]), (states[8] + size, states[9] + size)], fill=(0, 255, 0, 0))
    d.rectangle([(states[10], states[11]), (states[10] + size, states[11] + size)], fill=(255, 0, 0, 0))


width  = 200
height = 200
img  = Image.new(mode="RGB", size=(width, height), color=(202, 202, 202))
draw = ImageDraw.Draw(img)

import time
import psutil

for _ in range(0, 10):
    env    = RL_ENV()
    state  = env.state_space_funct()
    plot_state_space(draw, state)
    img.show()
    time.sleep(2.0)

    for proc in psutil.process_iter():
        if proc.name() == "Image Viewer":
            proc.kill()






