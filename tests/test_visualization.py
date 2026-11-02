"""Real browser scenarios for the offline presentation."""
# ruff: noqa: F811 -- pytest fixtures imported for discovery, arguments request them.

from pathlib import Path

from PIL import Image
from test_reports import artifact_dir, built_reports  # noqa: F401

from alfabetizacao_ml.visualization import prepare_deck_frames
from alfabetizacao_ml.visualization.reports import _browser


def test_native_button_space_and_fullscreen_toolbar(built_reports):
    with _browser() as browser:
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        page.goto(Path(built_reports["deck_html"]).as_uri())
        page.locator("#fullscreen").focus()
        page.keyboard.press("Space")
        page.mouse.move(1000, 600)
        page.wait_for_function(
            "document.documentElement.classList.contains('presenting') && getComputedStyle(document.querySelector('.toolbar')).opacity === '0'"
        )
        assert page.url.endswith("#slide-1")
        assert page.locator(".toolbar").evaluate("el => getComputedStyle(el).opacity") == "0"
        page.screenshot(
            path=str(Path(built_reports["deck_html"]).parent / "qa-fullscreen-hidden.png")
        )
        page.locator("#fullscreen").focus()
        assert page.locator(".toolbar").evaluate("el => getComputedStyle(el).opacity") == "1"
        page.keyboard.press("Space")
        page.wait_for_function("!document.fullscreenElement")
        assert page.url.endswith("#slide-1")
        page.set_viewport_size({"width": 390, "height": 844})
        page.evaluate("window.fase3Deck.show(7)")
        marker = page.locator(".slide.active .fixture-mark").bounding_box()
        counter = page.locator(".slide.active .slide-number").bounding_box()
        assert marker["y"] + marker["height"] <= counter["y"]
        page.screenshot(
            path=str(Path(built_reports["deck_html"]).parent / "qa-mobile-390.png"), full_page=True
        )


def test_frame_layout_stays_inside_canvas(built_reports):
    with _browser() as browser:
        page = browser.new_page(viewport={"width": 1920, "height": 1080}, reduced_motion="reduce")
        page.goto(Path(built_reports["deck_html"]).as_uri())
        page.evaluate("document.documentElement.dataset.mode='frame'")
        for index in range(10):
            page.evaluate("i => window.fase3Deck.show(i)", index)
            rectangles = page.locator(".slide.active [data-evidence]").evaluate_all(
                "nodes => nodes.map(n => {const r=n.getBoundingClientRect();return {x:r.x,y:r.y,right:r.right,bottom:r.bottom,text:n.textContent.slice(0,50)}})"
            )
            for rectangle in rectangles:
                assert rectangle["x"] >= 0 and rectangle["y"] >= 0, (index, rectangle)
                assert rectangle["right"] <= 1920 and rectangle["bottom"] <= 1080, (
                    index,
                    rectangle,
                )
            body = page.locator(".slide.active .slide-body").bounding_box()
            footer = page.locator(".slide.active .source").bounding_box()
            assert body["y"] + body["height"] <= footer["y"], (index, body, footer)


def test_keyboard_navigation_and_mobile_reflow(built_reports):
    with _browser() as browser:
        page = browser.new_page(viewport={"width": 1280, "height": 720})
        page.goto(Path(built_reports["deck_html"]).as_uri())
        page.keyboard.press("ArrowRight")
        assert page.url.endswith("#slide-2")
        assert page.locator("#title-2").evaluate("el => el === document.activeElement")
        page.keyboard.press("End")
        assert page.url.endswith("#slide-10")
        assert page.locator("#next").is_disabled()
        page.keyboard.press("Home")
        assert page.url.endswith("#slide-1")
        for width, height in ((390, 844), (640, 360), (768, 1024)):
            page.set_viewport_size({"width": width, "height": height})
            for index in range(10):
                page.evaluate("i => window.fase3Deck.show(i)", index)
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), (
                    width,
                    index,
                )


def test_exported_frames_are_full_hd(built_reports, tmp_path):
    frames = prepare_deck_frames(built_reports["deck_html"], tmp_path / "frames")
    assert len(frames) == 10
    for index, frame in enumerate(frames, 1):
        assert frame.name == f"slide-{index:02d}.png"
        assert Image.open(frame).size == (1920, 1080)
