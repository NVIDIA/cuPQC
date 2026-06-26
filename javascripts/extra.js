document.addEventListener("DOMContentLoaded", function () {
  var path = window.location.pathname;

  if (path === "/" || path.match(/\/(cuPQC\/)?(index\.html?)?$/) || path.endsWith("/cuPQC")) {
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

  var mobileNavToggle = document.getElementById("cupqc-mobile-nav");
  var mobileNavButton = document.querySelector(".cupqc-topbar__menu");
  var mobileNavOverlay = document.querySelector(".cupqc-mobile-nav__overlay");

  function setMobileNavOpen(isOpen) {
    if (!mobileNavToggle) return;
    mobileNavToggle.checked = isOpen;
    document.body.classList.toggle("cupqc-mobile-nav-open", isOpen);
    if (mobileNavButton) {
      mobileNavButton.setAttribute("aria-expanded", isOpen ? "true" : "false");
    }
  }

  if (mobileNavToggle && mobileNavButton) {
    mobileNavButton.addEventListener("click", function () {
      setMobileNavOpen(!mobileNavToggle.checked);
    });

    mobileNavToggle.addEventListener("change", function () {
      setMobileNavOpen(mobileNavToggle.checked);
    });

    if (mobileNavOverlay) {
      mobileNavOverlay.addEventListener("click", function () {
        setMobileNavOpen(false);
      });
    }

    document.querySelectorAll(".cupqc-mobile-nav a").forEach(function (link) {
      link.addEventListener("click", function () {
        setMobileNavOpen(false);
      });
    });

    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && mobileNavToggle.checked) {
        setMobileNavOpen(false);
      }
    });
  }

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
