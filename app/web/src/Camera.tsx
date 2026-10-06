import { useEffect, useRef, useState } from "react";

interface Props {
  onCapture: (photo: Blob) => void;
  onClose: () => void;
}

// A live camera view with one button to take the photo. The stream never leaves the page: frames
// are only read when Take photo is pressed, and every track stops when the camera closes.
export default function Camera({ onCapture, onClose }: Props) {
  const video = useRef<HTMLVideoElement>(null);
  const captureButton = useRef<HTMLButtonElement>(null);
  const closeButton = useRef<HTMLButtonElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let stream: MediaStream | null = null;
    let cancelled = false;
    navigator.mediaDevices
      .getUserMedia({ video: { facingMode: "environment" }, audio: false })
      .then((s) => {
        if (cancelled) {
          s.getTracks().forEach((t) => t.stop());
          return;
        }
        stream = s;
        if (video.current) video.current.srcObject = s;
        setReady(true);
      })
      .catch(() => {
        if (!cancelled) {
          setError(
            "Lumen couldn't open the camera. Check that this browser may use it " +
              "(System Settings, Privacy & Security, Camera), or choose a photo instead.",
          );
        }
      });
    return () => {
      cancelled = true;
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  // The button that opened the camera is gone: keep keyboard focus inside the camera controls,
  // and move it to the shutter once it can be pressed.
  useEffect(() => {
    (ready ? captureButton : closeButton).current?.focus();
  }, [ready]);

  function capture() {
    const v = video.current;
    if (!v || !v.videoWidth) return;
    const canvas = document.createElement("canvas");
    canvas.width = v.videoWidth;
    canvas.height = v.videoHeight;
    canvas.getContext("2d")?.drawImage(v, 0, 0);
    canvas.toBlob((blob) => blob && onCapture(blob), "image/jpeg", 0.92);
  }

  return (
    <div className="camera">
      {error ? (
        <p role="alert" className="error">
          {error}
        </p>
      ) : (
        <p role="status">
          {ready ? "Camera on. Point it at the object, then press Take photo." : "Opening the camera…"}
        </p>
      )}
      {/* The view is for sighted helpers; the photo itself is what Lumen answers about. */}
      <video ref={video} autoPlay playsInline muted aria-hidden="true" className="camera-view" />
      <div className="row">
        <button ref={captureButton} type="button" onClick={capture} disabled={!ready}>
          Take photo
        </button>
        <button ref={closeButton} type="button" onClick={onClose}>
          Close camera
        </button>
      </div>
    </div>
  );
}
