"""Small server-rendered demo using the same router and validation as the API."""

from html import escape
from importlib.resources import files
from string import Template

from intentops.router import Router
from intentops.schemas import Prediction

EXAMPLES = {
    "pin": "How do I change my PIN?",
    "card": "I am still waiting for my card",
    "unclear": "I need help",
}


def render_demo(
    router: Router | None,
    text: str = "",
    prediction: Prediction | None = None,
    error: str | None = None,
) -> str:
    """Escape all dynamic content; never include submitted text in URLs or logs."""
    result = ""
    if prediction is not None:
        state = "Human review" if prediction.needs_review else "Automatically accepted"
        rows = "".join(
            f"<tr><td>{escape(item.intent)}</td><td>{item.probability:.1%}</td></tr>"
            for item in prediction.top_k
        )
        result = (
            '<section class="result" aria-live="polite"><div class="eyebrow">Routing decision</div>'
            f"<h2>{escape(prediction.intent.replace('_', ' '))}</h2>"
            f'<p class="decision {"review" if prediction.needs_review else "accepted"}">{state}</p>'
            '<div class="stats">'
            f"<div><strong>{prediction.confidence:.1%}</strong>"
            "<span>Classifier confidence</span></div>"
            f"<div><strong>{prediction.latency_ms:.2f} ms</strong>"
            "<span>Routing time</span></div></div>"
            f'<p class="muted">Decision source: {escape(prediction.source)}. '
            "Routing time excludes HTTP transport and page rendering.</p>"
            "<h3>Classifier alternatives</h3><table><thead><tr><th>Intent</th>"
            f"<th>Probability</th></tr></thead><tbody>{rows}</tbody></table>"
            '<p class="muted">These probabilities are not calibrated. The suggested intent '
            "is still shown when a person needs to review the request.</p></section>"
        )
    threshold = f"{router.settings.review_threshold:.0%}" if router else "unavailable"
    version = router.baseline.version if router else "Model unavailable"
    template = Template(files("intentops").joinpath("templates/demo.html").read_text())
    return template.substitute(
        form_text=escape(text),
        model_version=escape(version),
        threshold=threshold,
        error_block=f'<p role="alert" class="error">{escape(error)}</p>' if error else "",
        result_block=result,
    )
