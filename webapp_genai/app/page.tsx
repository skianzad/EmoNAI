import { Editor } from "@/components/Editor";
import { Phone } from "@/components/Phone";

export default function Page() {
  return (
    <main className="stage">
      <Phone>
        <Editor />
      </Phone>
      <p className="caption">
        GenAI Image Editor
        <span>HMI Lab</span>
      </p>
    </main>
  );
}
