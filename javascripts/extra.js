// Make external links in navigation open in new tab
document.addEventListener('DOMContentLoaded', function() {
  // Select all navigation links that start with http
  const externalLinks = document.querySelectorAll('.md-nav__link[href^="http"]');
  
  externalLinks.forEach(link => {
    link.setAttribute('target', '_blank');
    link.setAttribute('rel', 'noopener noreferrer');
  });
});

