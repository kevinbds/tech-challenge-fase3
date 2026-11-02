(() => {
  const slides = [...document.querySelectorAll('.slide')];
  const previous = document.getElementById('previous');
  const next = document.getElementById('next');
  const status = document.getElementById('position');
  let index = 0;
  function resize() {
    if (document.documentElement.dataset.mode === 'frame' || innerWidth <= 900) return;
    const height = Math.max(100, innerHeight - (document.fullscreenElement ? 0 : document.querySelector('.toolbar').offsetHeight));
    const scale = Math.min(innerWidth / 1920, height / 1080);
    document.querySelector('.stage').style.transform = `scale(${scale})`;
    document.querySelector('.stage-shell').style.width = `${1920 * scale}px`;
    document.querySelector('.stage-shell').style.height = `${1080 * scale}px`;
  }
  function show(number, focus = false) {
    index = Math.max(0, Math.min(slides.length - 1, number));
    slides.forEach((slide, i) => { slide.classList.toggle('active', i === index); slide.setAttribute('aria-hidden', String(i !== index)); });
    previous.disabled = index === 0; next.disabled = index === slides.length - 1;
    status.textContent = `${index + 1} / ${slides.length}`;
    history.replaceState(null, '', `#slide-${index + 1}`);
    if (focus) slides[index].querySelector('h1').focus({preventScroll: true});
    resize();
  }
  function fromHash() { const match = location.hash.match(/^#slide-(\d+)$/); show(match ? Number(match[1]) - 1 : 0); }
  previous.addEventListener('click', () => show(index - 1, true));
  next.addEventListener('click', () => show(index + 1, true));
  document.getElementById('fullscreen').addEventListener('click', async () => {
    try { if (document.fullscreenElement) await document.exitFullscreen(); else await document.documentElement.requestFullscreen(); } catch { status.textContent = 'Tela cheia indisponível; navegação permanece ativa.'; }
  });
  addEventListener('keydown', event => {
    if (event.defaultPrevented || event.target.closest('button,a,input,textarea,select,[contenteditable="true"],[role="button"]')) return;
    if (['ArrowRight','PageDown',' '].includes(event.key)) { event.preventDefault(); show(index + 1, true); }
    if (['ArrowLeft','PageUp'].includes(event.key)) { event.preventDefault(); show(index - 1, true); }
    if (event.key === 'Home') { event.preventDefault(); show(0, true); }
    if (event.key === 'End') { event.preventDefault(); show(slides.length - 1, true); }
  });
  document.addEventListener('fullscreenchange', () => {
    document.documentElement.classList.toggle('presenting', Boolean(document.fullscreenElement));
    if (document.fullscreenElement) slides[index].querySelector('h1').focus({preventScroll: true});
    resize();
  });
  addEventListener('hashchange', fromHash); addEventListener('resize', resize);
  window.fase3Deck = {show, slides: slides.length}; fromHash();
})();
