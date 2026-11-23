"""Playwright browser environment: execute agent actions and return screenshots."""

import io
from pathlib import Path
from typing import Self
from urllib.parse import urlparse

from PIL import Image, ImageChops
from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

from holo_playground.agent.actions import ClickArgs, ScrollArgs, WriteArgs
from holo_playground.locate import COORD_SCALE, Screenshot

SETTLE_S = 0.5                  # pause before and after waiting for the network
NETWORK_IDLE_TIMEOUT_MS = 5000  # some pages never reach network idle: don't wait forever
SCROLL_FRACTION = 0.75          # one scroll = 3/4 of the viewport height
# Change detection counts pixels that really changed, at full resolution. A mean over the whole
# screenshot would hide a typed word (a few hundred pixels out of a million).
PIXEL_DIFF = 32                 # a pixel "changed" if its gray level moved by more than this (0-255)
MIN_CHANGED_PIXELS = 64         # fewer changed pixels = unchanged (a blinking caret is ~30 px)
# True if the focused element accepts text.
FOCUS_IS_EDITABLE_JS = """() => {
    const e = document.activeElement;
    return !!e && (e.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(e.tagName));
}"""


class Browser:
    """One Chromium page, seen and driven the way the model sees it.

    Screenshots are viewport-only and taken at device_scale_factor=1, so screenshot pixels,
    CSS pixels and mouse coordinates are the same thing.
    """

    def __init__(self, allowed_domains: set[str], width: int = 1280, height: int = 800,
                headless: bool = True) -> None:
        self.allowed_domains = allowed_domains
        self.width, self.height = width, height
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=headless)
        self._context = self._browser.new_context(
            viewport={"width": width, "height": height},
            device_scale_factor=1,
            locale="en-US",
        )
        self.page = self._context.new_page()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._browser.close()
        self._playwright.stop()

    def reset(self, url: str) -> None:
        """Open `url` at the start of a task."""
        self.page.goto(url, wait_until="domcontentloaded")
        self._settle()

    def screenshot(self, path: Path) -> Screenshot:
        """Save what the model will see (viewport only, not full_page) and load it."""
        self.page.screenshot(path=path)
        return Screenshot.load(path)

    def execute(self, tool_call: ClickArgs | WriteArgs | ScrollArgs) -> str:
        """Run one action and describe its effect in a short sentence for <tool_output>.

        Examples: "clicked (606, 32); url changed to /models", "page unchanged",
        "blocked: off-domain (github.com), went back".
        """
        before = self._snapshot()
        url_before = self.page.url
        n_pages = len(self._context.pages)

        match tool_call:
            case ClickArgs(x=x, y=y):
                px, py = self.to_pixels(x, y)
                self.page.mouse.click(px, py)
                done = f"clicked ({px:.0f}, {py:.0f})"
            case WriteArgs(content=content, press_enter=press_enter):
                # Typing with no text field focused would go nowhere, or trigger the site's
                # keyboard shortcuts: refuse and tell the model what to do instead.
                if not self.page.evaluate(FOCUS_IS_EDITABLE_JS):
                    return "nothing typed: no text field has focus, click one first"
                self.page.keyboard.type(content)
                if press_enter:
                    self.page.keyboard.press("Enter")
                done = f"typed {content!r}" + (" and pressed Enter" if press_enter else "")
            case ScrollArgs(direction=direction):
                # The wheel scrolls the element under the mouse: put it in the middle of the page.
                self.page.mouse.move(self.width / 2, self.height / 2)
                dy = SCROLL_FRACTION * self.height
                self.page.mouse.wheel(0, dy if direction == "down" else -dy)
                done = f"scrolled {direction}"
            case _:
                raise ValueError(f"not a browser action: {tool_call!r}")

        self._settle()
        new_tab = self._follow_new_tab(n_pages)
        if new_tab:
            done += "; opened a new tab"

        if not self._allowed(self.page.url):
            host = urlparse(self.page.url).hostname
            if new_tab:
                self.page.close()
                self.page = self._context.pages[-1]
                return f"blocked: off-domain ({host}), closed the new tab"
            self.page.go_back(wait_until="domcontentloaded")
            self._settle()
            return f"blocked: off-domain ({host}), went back"

        if self.page.url != url_before:
            return f"{done}; url changed to {self._short_url(self.page.url)}"
        if self._unchanged(before, self._snapshot()):
            return f"{done}; page unchanged"
        return done

    def to_pixels(self, x: int, y: int) -> tuple[float, float]:
        """Model coordinates in [0, COORD_SCALE] -> viewport pixels."""
        return x / COORD_SCALE * self.width, y / COORD_SCALE * self.height

    def _settle(self) -> None:
        """Let a navigation start, wait for the network to calm down, let the page render."""
        self.page.wait_for_timeout(SETTLE_S * 1000)
        try:
            self.page.wait_for_load_state("networkidle", timeout=NETWORK_IDLE_TIMEOUT_MS)
        except PlaywrightTimeout:
            pass  # pages with polling or streaming never go idle: carry on after the timeout
        self.page.wait_for_timeout(SETTLE_S * 1000)

    def _follow_new_tab(self, n_pages_before: int) -> bool:
        """If the last action opened a tab, continue in it. Return True if it did."""
        if len(self._context.pages) <= n_pages_before:
            return False
        self.page = self._context.pages[-1]
        self.page.set_viewport_size({"width": self.width, "height": self.height})
        self._settle()
        return True

    def _allowed(self, url: str) -> bool:
        """True if the url's host is an allowed domain or one of its subdomains."""
        host = urlparse(url).hostname or ""
        # endswith("." + d), not endswith(d): "evilhuggingface.co" must not pass for "huggingface.co"
        return any(host == d or host.endswith("." + d) for d in self.allowed_domains)

    def _short_url(self, url: str) -> str:
        """Path and query only: off-domain urls are blocked before we get here."""
        parts = urlparse(url)
        return (parts.path or "/") + (f"?{parts.query}" if parts.query else "")

    def _snapshot(self) -> Image.Image:
        """Grayscale screenshot, for change detection only."""
        return Image.open(io.BytesIO(self.page.screenshot())).convert("L")

    def _unchanged(self, before: Image.Image, after: Image.Image) -> bool:
        """True if the page looks the same (ignores a blinking caret or a small animation)."""
        changed = ImageChops.difference(before, after).point(lambda v: 255 if v > PIXEL_DIFF else 0) # type: ignore
        return changed.histogram()[255] < MIN_CHANGED_PIXELS


if __name__ == "__main__":
    shots = Path("runs/browser-check")
    shots.mkdir(parents=True, exist_ok=True)
    with Browser({"huggingface.co"}) as browser:
        browser.reset("https://huggingface.co/Hcompany")
        browser.screenshot(shots / "0-start.png")
        checks = [  # the HF header is not sticky: search before scrolling
            ("click GitHub link", ClickArgs(tool_name="click", element="hcompai GitHub link", x=348, y=204)),
            ("click Models", ClickArgs(tool_name="click", element="Models link", x=473, y=40)),
            ("click empty area", ClickArgs(tool_name="click", element="empty area", x=5, y=990)),
            ("click search box", ClickArgs(tool_name="click", element="search box", x=250, y=40)),
            ("type a query", WriteArgs(tool_name="write", content="Holo", press_enter=True)),
            ("scroll down", ScrollArgs(tool_name="scroll", direction="down")),
        ]
        for i, (name, action) in enumerate(checks, start=1):
            print(f"{name:<18} -> {browser.execute(action)}")
            browser.screenshot(shots / f"{i}-{name.replace(' ', '-')}.png")
