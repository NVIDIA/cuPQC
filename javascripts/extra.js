document.addEventListener("DOMContentLoaded", function () {
  var path = window.location.pathname;

  if (path.match(/\/(cuPQC\/)?(index\.html?)?$/) || path.endsWith("/cuPQC")) {
    document.body.classList.add("cupqc-page-home");
  } else if (path.includes("get-started")) {
    document.body.classList.add("cupqc-page-getstarted");
  } else if (path.includes("applications")) {
    document.body.classList.add("cupqc-section-apps");
  } else if (path.includes("blogs") || path.includes("releases")) {
    document.body.classList.add("cupqc-section-blog");
  } else if (path.includes("docs-hub")) {
    document.body.classList.add("cupqc-page-docs");
  }

  document.querySelectorAll('a[href^="http"]:not([target="_blank"])').forEach(function (link) {
    link.setAttribute("target", "_blank");
    link.setAttribute("rel", "noopener noreferrer");
  });

  document.querySelectorAll(".cupqc-copy-btn").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var text = (btn.getAttribute("data-copy") || "").replace(/&#10;/g, "\n");
      var label = btn.getAttribute("data-label") || btn.textContent;
      var copied = btn.getAttribute("data-copied") || "Copied";
      navigator.clipboard.writeText(text).then(function () {
        btn.textContent = copied;
        setTimeout(function () { btn.textContent = label; }, 1500);
      }).catch(function () {});
    });
  });

  var chart = document.querySelector(".cupqc-bench-chart, .cupqc-hash-chart");
  if (chart) {
    requestAnimationFrame(function () {
      chart.classList.add("is-animated");
    });
  }
});
