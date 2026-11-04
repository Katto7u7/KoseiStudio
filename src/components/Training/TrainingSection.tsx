import { useState } from "react";
import StepUpload from "./StepUpload";
import StepCaption from "./StepCaption";
import StepConfig from "./StepConfig";
import StepProgress from "./StepProgress";
import type { CaptionEntry } from "../../types";

type Step = 0 | 1 | 2 | 3;

interface UploadState {
  jobId: string;
  filenames: string[];
  previewUrls: string[];
}

const STEPS = ["Imágenes", "Captions", "Config", "Progreso"];

export default function TrainingSection() {
  const [step, setStep] = useState<Step>(0);
  const [upload, setUpload] = useState<UploadState | null>(null);
  const [captions, setCaptions] = useState<CaptionEntry[]>([]);
  const [tokenPrefix, setTokenPrefix] = useState("ohwxsubject");
  const [activeJobId, setActiveJobId] = useState<string | null>(null);

  const goTo = (s: Step) => setStep(s);

  return (
    <div className="section-content">
      <div className="panel stagger-1">
        <div className="panel-header">
          <span className="panel-header-title">LoRA Training</span>
          <span className="panel-header-sub">Pipeline automático con kohya_ss + BLIP</span>
        </div>

        <div className="stepper">
          {STEPS.map((label, i) => {
            const state = i < step ? "done" : i === step ? "active" : "";
            return (
              <div key={i} className={`step ${state}`}>
                <span className="step-num">{i < step ? "✓" : i + 1}</span>
                <span className="step-label" style={{ fontSize: 12 }}>{label}</span>
              </div>
            );
          })}
        </div>

        {step === 0 && (
          <StepUpload
            onComplete={(jobId, filenames, previewUrls) => {
              setUpload({ jobId, filenames, previewUrls });
              goTo(1);
            }}
          />
        )}

        {step === 1 && upload && (
          <StepCaption
            jobId={upload.jobId}
            filenames={upload.filenames}
            previewUrls={upload.previewUrls}
            tokenPrefix={tokenPrefix}
            setTokenPrefix={setTokenPrefix}
            captions={captions}
            setCaptions={setCaptions}
            onBack={() => goTo(0)}
            onNext={() => goTo(2)}
          />
        )}

        {step === 2 && upload && (
          <StepConfig
            jobId={upload.jobId}
            tokenPrefix={tokenPrefix}
            captions={captions}
            imageCount={upload.filenames.length}
            onBack={() => goTo(1)}
            onStarted={(jobId) => { setActiveJobId(jobId); goTo(3); }}
          />
        )}

        {step === 3 && activeJobId && (
          <StepProgress
            jobId={activeJobId}
            onReset={() => {
              setStep(0);
              setUpload(null);
              setCaptions([]);
              setActiveJobId(null);
            }}
          />
        )}
      </div>
    </div>
  );
}
