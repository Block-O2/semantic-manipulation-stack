"use strict";

const mediaBlocks = document.querySelectorAll("[data-demo-media]");

mediaBlocks.forEach((block) => {
  const video = block.querySelector(".demo-video");
  const videoFallback = block.querySelector("[data-video-fallback]");
  const videoPath = block.dataset.video;

  if (video && videoFallback && videoPath) {
    const showVideo = () => {
      video.hidden = false;
      videoFallback.hidden = true;
    };

    const keepVideoFallback = () => {
      video.hidden = true;
      videoFallback.hidden = false;
      video.removeAttribute("src");
    };

    video.addEventListener("loadedmetadata", showVideo, { once: true });
    video.addEventListener("error", keepVideoFallback, { once: true });
    video.src = videoPath;
    video.load();
  }

  const image = block.querySelector(".scene-image");
  const imageFallback = block.querySelector("[data-image-fallback]");
  const imagePath = block.dataset.image;

  if (image && imageFallback && imagePath) {
    const probe = new Image();

    probe.addEventListener(
      "load",
      () => {
        image.src = imagePath;
        image.hidden = false;
        imageFallback.hidden = true;
      },
      { once: true },
    );

    probe.addEventListener(
      "error",
      () => {
        image.hidden = true;
        imageFallback.hidden = false;
      },
      { once: true },
    );

    probe.src = imagePath;
  }
});
