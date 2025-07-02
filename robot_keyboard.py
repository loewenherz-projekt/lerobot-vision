"""Minimal keyboard controller for manual robot debugging."""

import time
import logging
from pynput import keyboard
from lerobot_vision import web_interface

logging.basicConfig(level=logging.INFO)


class KeyboardController:
    def __init__(self, step: float = 5.0) -> None:
        self.running = True
        self.step = step

    def on_press(self, key: keyboard.Key) -> bool:
        if key == keyboard.Key.esc:
            self.running = False
            return False
        pos = web_interface.robot.get_positions()
        try:
            if key == keyboard.Key.up:
                pos[0] += self.step
            elif key == keyboard.Key.down:
                pos[0] -= self.step
            elif key == keyboard.Key.right:
                pos[1] += self.step
            elif key == keyboard.Key.left:
                pos[1] -= self.step
            else:
                return True
            web_interface.robot.move(pos)
        except Exception as exc:  # pragma: no cover - runtime
            logging.error("Keyboard control failed: %s", exc)
        return True

    def run(self) -> None:
        with keyboard.Listener(on_press=self.on_press) as _:
            while self.running:
                time.sleep(0.1)


def main() -> None:
    KeyboardController().run()


if __name__ == "__main__":
    main()
