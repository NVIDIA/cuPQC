// Make all external links open in new tab
document.addEventListener('DOMContentLoaded', function() {
  // Select all links that start with http or https (external links)
  // Exclude links that already have target="_blank" set
  const externalLinks = document.querySelectorAll('a[href^="http"]:not([target="_blank"])');
  
  externalLinks.forEach(link => {
    link.setAttribute('target', '_blank');
    link.setAttribute('rel', 'noopener noreferrer');
  });
});
