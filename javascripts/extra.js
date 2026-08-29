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

  initFacetFilters();
});

/*
  Sidebar facet filters on the Applications and Blog overview pages. The filter
  buttons and their counts are rendered by overrides/partials/facet-nav.html
  from data computed in hooks/facets.py; this only wires up the filtering.

  Markup contract:
    panel  <nav data-facets>
    group  <div data-facet-group="year">
    option <button data-facet-value="2026">
    item   <a data-facet-item data-facet-year="2026" ...>
    section  optional wrapper hidden when it holds no visible items
    empty    optional message shown when nothing matches
*/
function initFacetFilters() {
  function toArray(list) {
    return Array.prototype.slice.call(list);
  }

  var panel = document.querySelector("[data-facets]");
  var items = toArray(document.querySelectorAll("[data-facet-item]"));
  if (!panel || !items.length) return;

  var groups = toArray(panel.querySelectorAll("[data-facet-group]"));
  var sections = toArray(document.querySelectorAll("[data-facet-section]"));
  var empty = document.querySelector("[data-facet-empty]");
  var ALL = "__all__";
  var selection = {};

  function keyOf(group) {
    return group.getAttribute("data-facet-group");
  }

  function optionsOf(group) {
    return toArray(group.querySelectorAll("[data-facet-value]"));
  }

  function valueOf(item, key) {
    return item.getAttribute("data-facet-" + key) || "";
  }

  function matches(item) {
    return groups.every(function (group) {
      var key = keyOf(group);
      return selection[key] === ALL || selection[key] === valueOf(item, key);
    });
  }

  /* Detail pages link back here as ?type=Tutorials, so the overview opens
     with that filter already applied. */
  function requested(group) {
    var found = new RegExp("[?&]" + keyOf(group) + "=([^&]*)").exec(location.search);
    if (!found) return ALL;
    var value = decodeURIComponent(found[1].replace(/\+/g, " "));
    var known = optionsOf(group).some(function (option) {
      return option.getAttribute("data-facet-value") === value;
    });
    return known ? value : ALL;
  }

  function apply() {
    var shown = 0;
    items.forEach(function (item) {
      var visible = matches(item);
      item.hidden = !visible;
      if (visible) shown++;
    });
    sections.forEach(function (section) {
      section.hidden = !section.querySelector("[data-facet-item]:not([hidden])");
    });
    if (empty) empty.hidden = shown > 0;
    groups.forEach(function (group) {
      var selected = selection[keyOf(group)];
      optionsOf(group).forEach(function (option) {
        var isActive = option.getAttribute("data-facet-value") === selected;
        option.classList.toggle("is-active", isActive);
        option.setAttribute("aria-pressed", isActive ? "true" : "false");
      });
    });
  }

  groups.forEach(function (group) {
    var key = keyOf(group);
    selection[key] = requested(group);
    optionsOf(group).forEach(function (option) {
      option.addEventListener("click", function () {
        selection[key] = option.getAttribute("data-facet-value");
        apply();
      });
    });
  });

  apply();
}
