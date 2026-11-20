"""Browser environment.

Unit tests call the pure helpers on a Browser built without Chromium. Integration tests drive a
real headless Chromium on a small page served locally, so they need neither the internet nor a
model. The viewport is 1000x1000 px, so model coordinates (0-1000) equal pixels.
"""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from PIL import Image, ImageDraw

from holo_playground.agent.actions import ClickArgs, ScrollArgs, WriteArgs
from holo_playground.agent.browser import Browser

# ---------------------------------------------------------------- unit tests, no Chromium


@pytest.fixture
def bare():
    """A Browser with its attributes but no Playwright behind it."""
    b = object.__new__(Browser)
    b.allowed_domains, b.width, b.height = {"huggingface.co"}, 1280, 800
    return b


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [(0, 0, (0, 0)), (1000, 1000, (1280, 800)), (500, 500, (640, 400)), (473, 40, (605.44, 32))],
)
def test_to_pixels(bare, x, y, expected):
    assert bare._to_pixels(x, y) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("url", "allowed"),
    [
        ("https://huggingface.co/models", True),
        ("https://www.huggingface.co/", True),  # subdomain
        ("https://hub.huggingface.co/x", True),
        ("https://evilhuggingface.co/", False),  # suffix without a dot: a different site
        ("https://huggingface.co.evil.com/", False),
        ("https://github.com/hcompai", False),
        ("about:blank", False),
        ("chrome-error://chromewebdata/", False),  # failed navigation
    ],
)
def test_allowed(bare, url, allowed):
    assert bare._allowed(url) is allowed


def test_short_url_keeps_path_and_query(bare):
    assert bare._short_url("https://huggingface.co/models?sort=trending") == "/models?sort=trending"
    assert bare._short_url("https://huggingface.co") == "/"


def page_image(caret: bool = False, word: bool = False, panel: bool = False) -> Image.Image:
    """A fake 1280x800 grayscale snapshot."""
    img = Image.new("L", (1280, 800), 255)
    draw = ImageDraw.Draw(img)
    draw.rectangle((80, 80, 1200, 160), fill=40)  # a header
    draw.rectangle((100, 300, 500, 340), outline=120)  # an empty text field
    if caret:
        draw.line((106, 310, 106, 330), fill=0)  # a blinking text caret: 21 px
    if word:
        draw.rectangle((106, 312, 146, 328), fill=30)  # a typed word, ~40x16 px
    if panel:
        draw.rectangle((160, 320, 1120, 720), fill=90)  # a dropdown or a new section
    return img


def test_unchanged_for_identical_pages(bare):
    assert bare._unchanged(page_image(), page_image())


def test_unchanged_ignores_a_blinking_caret(bare):
    assert bare._unchanged(page_image(), page_image(caret=True))


def test_changed_when_a_word_is_typed(bare):
    # Regression: a mean over the whole screenshot called this "unchanged".
    assert not bare._unchanged(page_image(), page_image(word=True))


def test_changed_when_a_panel_opens(bare):
    assert not bare._unchanged(page_image(), page_image(panel=True))


# ---------------------------------------------------------------- integration, real Chromium

HOME = """<!doctype html><html><body style="margin:0;height:3000px;font:16px sans-serif">
<a href="/next" style="position:absolute;left:100px;top:100px;width:200px;height:50px">next</a>
<a href="{external}" style="position:absolute;left:100px;top:300px;width:200px;height:50px">external</a>
<a href="/next" target="_blank"
   style="position:absolute;left:100px;top:500px;width:200px;height:50px">new tab</a>
<input id="q" style="position:absolute;left:100px;top:700px;width:300px;height:40px">
</body></html>"""
NEXT = "<!doctype html><html><body><h1>next page</h1></body></html>"


@pytest.fixture(scope="module")
def site():
    """Serve HOME and NEXT on localhost. The external link points to 127.0.0.1: same server,
    different host, so it counts as off-domain when only "localhost" is allowed."""
    pages: dict[str, str] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = pages.get(self.path, NEXT).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    pages["/"] = HOME.format(external=f"http://127.0.0.1:{port}/elsewhere")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://localhost:{port}"
    server.shutdown()


@pytest.fixture
def browser(site):
    with Browser({"localhost"}, width=1000, height=1000) as b:
        b.reset(site + "/")
        yield b


def click(x: int, y: int) -> ClickArgs:
    return ClickArgs(tool_name="click", element="test", x=x, y=y)


def test_click_on_a_link_navigates(browser):
    assert browser.execute(click(200, 125)) == "clicked (200, 125); url changed to /next"


def test_click_on_nothing_reports_page_unchanged(browser):
    assert browser.execute(click(900, 900)) == "clicked (900, 900); page unchanged"


def test_off_domain_link_is_blocked_and_undone(browser, site):
    assert browser.execute(click(200, 325)) == "blocked: off-domain (127.0.0.1), went back"
    assert browser.page.url == site + "/"


def test_new_tab_is_followed(browser):
    result = browser.execute(click(200, 525))
    assert result == "clicked (200, 525); opened a new tab; url changed to /next"
    assert browser.page.url.endswith("/next")


def test_write_types_into_the_focused_field(browser):
    browser.execute(click(250, 720))  # focus the input
    result = browser.execute(WriteArgs(tool_name="write", content="Holo"))
    assert browser.page.input_value("#q") == "Holo"
    assert result == "typed 'Holo'"  # the text is visible, so the page changed


def test_write_without_focus_types_nothing(browser):
    result = browser.execute(WriteArgs(tool_name="write", content="Holo"))
    assert result == "nothing typed: no text field has focus, click one first"
    assert browser.page.input_value("#q") == ""


def test_scroll_moves_the_page(browser):
    assert browser.execute(ScrollArgs(tool_name="scroll", direction="down")) == "scrolled down"
    assert browser.page.evaluate("window.scrollY") == pytest.approx(750, abs=5)
    browser.execute(ScrollArgs(tool_name="scroll", direction="up"))
    assert browser.page.evaluate("window.scrollY") == 0


def test_screenshot_is_viewport_sized(browser, tmp_path):
    shot = browser.screenshot(tmp_path / "s.png")
    assert (shot.width, shot.height) == (1000, 1000)  # viewport only, not the 3000 px page
