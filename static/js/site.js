document.addEventListener("DOMContentLoaded", () => {
  const button = document.getElementById("copy-bibtex");
  const citation = document.getElementById("bibtex-code");
  if (!button || !citation) return;

  button.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(citation.textContent);
      const label = button.querySelector("span");
      if (label) label.textContent = "Copied";
      window.setTimeout(() => {
        if (label) label.textContent = "Copy";
      }, 1600);
    } catch (error) {
      console.error("Could not copy citation", error);
    }
  });
});
